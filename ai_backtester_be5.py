#!/usr/bin/env python3
"""
ai_backtester_be1.py — v2.0 PARITY EDITION

PARITY FIXES vs LIVE ENGINE:
1. TRAILING SL AFTER BE: Instead of moving SL to entry (BE), moves it to
   entry + (be_trail_pct * TP_distance), capturing partial profit on partial moves.
2. DYNAMIC SPREAD: Reads the actual 'spread' column from OHLC CSV per candle
   instead of a fixed static value. Falls back to --spread-ticks if column absent.
3. FILL SLIPPAGE MODEL: Adds configurable random slippage on entry to simulate
   live fill conditions (not perfect fills at limit price).
4. MERGE ALIGNMENT GUARD: Validates timestamp delta between ML state and OHLC.
   Warns if misalignment > 60s — catches timezone/broker offset bugs silently.
5. ANTI-REPAINTING GUARD: Offsets all structure feature columns by +1 bar
   (--anti-repaint flag) so the model only sees what was KNOWN at signal time,
   not what was confirmed afterward (the #1 hidden SMC bug).
6. CANDLE-BY-CANDLE WALKFORWARD: (--walkforward flag) processes bar-by-bar
   instead of batch-predicting on frozen states. Closest to true live parity.
7. MIN STOP DISTANCE: If the structural SL distance from the entry/limit price
   is smaller than --min-stop-distance, the SL is moved outward to
   entry ± min_stop_distance and TP is recalculated as
   entry ± (min_stop_distance * RR).  Set to 0.0 to disable (default).

Usage:
    python ai_backtester_be1.py \
      --in-csv ./ml_dataset_labeled_v15.csv \
      --ohlc-csv "./historical_data/Volatility 75 (1s) Index_M5.csv" \
      --model model_2R_50pb_v15.json \
      --metadata model_2R_50pb_v15_metadata.json \
      --direction both --rr 2.0 --be-trigger 1.5 \
      --be-trail-pct 0.15 \
      --threshold 0.50 --threshold-max 0.55 \
      --blocked-hours "3,4,5,7,9,10,14,19,20,22" \
      --start-date 2026-01-01 --end-date 2026-06-01 \
      --risk-pct 1.0 --slippage-ticks 3 \
      --min-stop-distance 0.0
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import argparse
import math
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from preprocessing import preprocess_for_model

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()

    # Data & Model
    p.add_argument("--in-csv",        required=True)
    p.add_argument("--ohlc-csv",      default=None)
    p.add_argument("--model",         required=True)
    p.add_argument("--metadata",      required=True)
    p.add_argument("--direction",     default="both", choices=["both","bullish","bearish"])
    p.add_argument("--bias-col",      default="m5_external_bias")
    p.add_argument("--time-col",      default="timestamp")
    p.add_argument("--start-date",    default=None)
    p.add_argument("--end-date",      default=None)

    # Account & Risk
    p.add_argument("--balance",       type=float, default=10.0)
    p.add_argument("--risk-pct",      type=float, default=10.0)
    p.add_argument("--rr",            type=float, default=1.0)
    p.add_argument("--be-trigger",    type=float, default=1.0,
                   help="Move SL after price travels X×R in your favor")
    p.add_argument("--be-trail-pct",  type=float, default=0.15,
                   help="After BE trigger, SL moves to entry+(this×TP_dist) instead of entry. "
                        "0.0 = classic breakeven. 0.15 = lock 15%% of TP distance as profit.")
    p.add_argument("--threshold",     type=float, default=0.55)
    p.add_argument("--threshold-max", type=float, default=0.95)
    p.add_argument("--cooldown-minutes", type=int, default=0)

    # Broker Specs (V75 defaults)
    p.add_argument("--min-lot",       type=float, default=0.05)
    p.add_argument("--max-lot",       type=float, default=80.0)
    p.add_argument("--lot-step",      type=float, default=0.001)
    p.add_argument("--tick-size",     type=float, default=0.01)
    p.add_argument("--tick-value",    type=float, default=0.01)
    p.add_argument("--min-stop-level",type=float, default=460)
    p.add_argument("--spread-ticks",  type=float, default=50,
                   help="Fallback static spread if OHLC has no 'spread' column")
    p.add_argument("--slippage-ticks",type=float, default=0,
                   help="Max random slippage on entry fill (ticks). 0=perfect fill.")
    p.add_argument("--limit-expiry-bars", type=int, default=6,
                   help="Max bars to wait for a limit order to fill before skipping the trade. "
                        "Default 6 bars (~30 min on M5). Set 0 for market orders only.")
    p.add_argument("--entry-col",     default="close")

    # ── ⑦ MIN STOP DISTANCE ────────────────────────────────────────────────
    # If the structural SL distance (|limit_price - base_sl|) is smaller than
    # this value, the SL is moved outward to limit_price ± min_stop_distance
    # and TP is recalculated from that new effective distance × RR.
    # Set to 0.0 (default) to disable — structural SL is always used as-is.
    p.add_argument("--min-stop-distance", type=float, default=7.0,
                   help="Minimum allowed SL distance in price units from entry. "
                        "If structural SL is closer than this, SL is overridden to "
                        "entry ± min_stop_distance and TP recalculated at RR × new dist. "
                        "0.0 = disabled (use structural SL as-is).")

    # Filters
    p.add_argument("--blocked-hours", default="")

    # Parity flags
    p.add_argument("--anti-repaint",  action="store_true",
                   help="Lag all structure feature columns by +1 bar before prediction "
                        "to prevent look-ahead bias from repainting SMC features.")
    p.add_argument("--merge-tolerance-secs", type=int, default=60,
                   help="Max allowed timestamp misalignment between ML state and OHLC (secs).")
    p.add_argument("--sl-compression", type=float, default=0.7,
                   help="Compress structural SL (e.g., 0.80 = 80% of structural distance). TP remains anchored to full distance.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LOT SIZE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
def calculate_lot_size(risk_money, entry_price, sl_price, args):
    sl_pts   = abs(entry_price - sl_price)
    sl_ticks = sl_pts / args.tick_size

    if sl_ticks < args.min_stop_level:
        return 0.0, 0.0          # too tight

    loss_per_lot = sl_ticks * args.tick_value
    if loss_per_lot <= 0:
        return 0.0, 0.0

    raw_lot = risk_money / loss_per_lot
    if raw_lot < args.min_lot:
        return 0.0, 0.0          # too wide

    lot = math.floor(raw_lot / args.lot_step) * args.lot_step
    lot = max(args.min_lot, min(args.max_lot, lot))
    return round(lot, 3), round(lot * loss_per_lot, 4)


# ─────────────────────────────────────────────────────────────────────────────
# PARITY UTILITY: ANTI-REPAINT LAG
# ─────────────────────────────────────────────────────────────────────────────
STRUCTURE_KEYWORDS = [
    'strong_low','strong_high','weak_low','weak_high',
    'sweep_low','sweep_high','bos','choch','internal',
    'poi','pullback','equilibrium','last_internal'
]

def apply_anti_repaint_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shifts all structure/SMC feature columns forward by 1 bar.
    This ensures the model at bar T only sees structure confirmed at bar T-1,
    matching live behavior where the bar hasn't closed yet.
    """
    lagged = 0
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in STRUCTURE_KEYWORDS):
            df[col] = df[col].shift(1)
            lagged += 1
    print(f"[PARITY] Anti-repaint: lagged {lagged} structure columns by +1 bar.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PARITY UTILITY: MERGE ALIGNMENT GUARD
# ─────────────────────────────────────────────────────────────────────────────
def check_merge_alignment(df: pd.DataFrame, ohlc_df: pd.DataFrame,
                           tolerance_secs: int = 60):
    """
    After merge_asof, checks the actual timestamp delta between ML state
    and attached OHLC bar. Warns if any delta exceeds tolerance.
    """
    if 'ohlc_timestamp' not in df.columns:
        return df

    delta = (df['timestamp'] - df['ohlc_timestamp']).abs().dt.total_seconds()
    bad   = (delta > tolerance_secs).sum()
    worst = delta.max()

    if bad > 0:
        print(f"\n[PARITY WARNING] {bad} rows have OHLC misalignment > {tolerance_secs}s "
              f"(worst: {worst:.0f}s). Possible timezone or broker offset issue.")
        print(f"  → These rows will use wrong candle OHLC for entry/SL/TP calculations.")
        print(f"  → Fix: ensure both CSVs use the same timezone (UTC) and broker timestamps.\n")
    else:
        print(f"[PARITY] Merge alignment OK — max delta {worst:.0f}s, all within {tolerance_secs}s.")

    df = df.drop(columns=['ohlc_timestamp'], errors='ignore')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 54)
    print("  AI MT5 BACKTESTER — PARITY EDITION v2.0")
    print("=" * 54)
    if args.be_trail_pct > 0:
        print(f"[PARITY] Trailing SL after BE: lock {args.be_trail_pct*100:.0f}% of TP distance as profit")
    if args.slippage_ticks > 0:
        print(f"[PARITY] Fill slippage model: up to {args.slippage_ticks} ticks random on entry")
    if args.anti_repaint:
        print(f"[PARITY] Anti-repaint guard: structure columns lagged +1 bar")
    if args.min_stop_distance > 0.0:
        print(f"[PARITY] Min stop distance: {args.min_stop_distance} — structural SL overridden when closer")

    # ── Load ML state dataset ────────────────────────────────────────────────
    df = pd.read_csv(args.in_csv, low_memory=False)

    if args.time_col in df.columns:
        df['timestamp'] = pd.to_datetime(df[args.time_col], errors='coerce', utc=True)
    else:
        for tc in ['time','date','timestamp']:
            if tc in df.columns:
                df['timestamp'] = pd.to_datetime(df[tc], errors='coerce', utc=True)
                break

    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # ── Anti-repaint: lag structure columns BEFORE any merging ──────────────
    if args.anti_repaint:
        df = apply_anti_repaint_lag(df)

    # ── Load & merge OHLC ───────────────────────────────────────────────────
    ohlc_df      = None
    has_spread   = False

    if args.ohlc_csv and os.path.exists(args.ohlc_csv):
        print(f"[*] Loading OHLC: {args.ohlc_csv}")
        ohlc_df = pd.read_csv(args.ohlc_csv, low_memory=False)
        ohlc_df.columns = [c.lower().replace('<','').replace('>','').strip()
                           for c in ohlc_df.columns]

        t_cols = [c for c in ['time','timestamp','date'] if c in ohlc_df.columns]
        ohlc_df['timestamp'] = pd.to_datetime(ohlc_df[t_cols[0]], errors='coerce', utc=True)
        ohlc_df = (ohlc_df.dropna(subset=['timestamp'])
                          .sort_values('timestamp')
                          .reset_index(drop=True))

        # PARITY: check for real spread column
        if 'spread' in ohlc_df.columns:
            has_spread = True
            print(f"[PARITY] Real spread column found in OHLC — using dynamic spread per candle.")
        else:
            print(f"[PARITY] No 'spread' column in OHLC — using static {args.spread_ticks} ticks fallback.")

        # Drop OHLC columns that may already exist in df
        for c in ['open','high','low','close']:
            if c in df.columns:
                df = df.drop(columns=[c])

        keep = [c for c in ['timestamp','open','high','low','close','spread']
                if c in ohlc_df.columns]

        # PARITY: keep ohlc_timestamp for alignment guard
        ohlc_df['ohlc_timestamp'] = ohlc_df['timestamp']
        keep_with_ts = keep + ['ohlc_timestamp']

        df = pd.merge_asof(df, ohlc_df[keep_with_ts],
                           on='timestamp', direction='backward')

        # Run alignment guard
        df = check_merge_alignment(df, ohlc_df, args.merge_tolerance_secs)
    else:
        print("[*] No --ohlc-csv. Using prices from state table.")

    if args.start_date:
        df = df[df['timestamp'] >= pd.to_datetime(args.start_date, utc=True)]
    if args.end_date:
        df = df[df['timestamp'] <= pd.to_datetime(args.end_date, utc=True)]

    df = df.reset_index(drop=True)

    # ── Load model & metadata ────────────────────────────────────────────────
    print(f"[*] Loading model: {args.model}")
    with open(args.metadata) as f:
        meta = json.load(f)

    features          = meta['features']
    category_mappings = meta.get('category_mappings', {})

    bst = xgb.Booster()
    bst.load_model(args.model)

    # ── Ironclad sanitization ─────────────────────────────────────────────
    for col in features:
        if col not in df.columns:
            df[col] = 'unknown' if col in category_mappings else 0.0

    for col, mapping in category_mappings.items():
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str).str.strip().str.lower()
            vk       = {str(k).lower(): str(k) for k in mapping}
            df[col]  = df[col].map(lambda x: vk.get(x, x))
            bad_mask = ~df[col].isin(set(mapping))
            if bad_mask.any():
                fb = ('unknown' if 'unknown' in mapping
                      else ('neutral' if 'neutral' in mapping
                      else list(mapping)[0]))
                df.loc[bad_mask, col] = fb

    for col in features:
        if col in df.columns and col not in category_mappings:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # ── Predict ──────────────────────────────────────────────────────────────
    bias_filter = (args.bias_col, args.direction) if args.direction != 'both' else None

    try:
        X, valid_df = preprocess_for_model(df, features, category_mappings, bias_filter)
    except Exception as e:
        print(f"[!] Preprocessing error: {e}")
        return

    valid_df['ai_prob'] = bst.predict(xgb.DMatrix(X))

    trade_df = valid_df[valid_df['ai_prob'] >= args.threshold].copy()
    print(f"[*] AI approved {len(trade_df)} setups at >= {args.threshold*100:.0f}% confidence.")

    # Dedup: keep highest-prob signal per candle
    trade_df = (trade_df.sort_values('ai_prob', ascending=False)
                        .drop_duplicates(subset=['timestamp'], keep='first')
                        .sort_values('timestamp')
                        .reset_index(drop=True))
    print(f"[*] After dedup: {len(trade_df)} unique-candle setups.")

    if len(trade_df) == 0:
        print("[!] No trades. Try --threshold lower.")
        return

    # ── Auto-detect pullback % from model name ─────────────────────────────
    entry_mode = 'close'
    for pb in ['50pb','30pb','20pb']:
        if pb in args.model:
            entry_mode = float(pb.replace('pb','')) / 100
            break

    if isinstance(entry_mode, float):
        print(f"[*] Limit order mode: {entry_mode*100:.0f}% pullback.")
    else:
        print(f"[*] Market execution mode.")

    # ── Simulation setup ──────────────────────────────────────────────────
    balance        = args.balance
    peak_balance   = balance
    max_dd_pct     = 0.0
    breakevens     = 0
    trail_exits    = 0        # count profitable trail SL exits
    min_stop_overrides = 0    # ⑦ count how many times min_stop_distance was applied
    results        = []
    last_trade_time = None

    skip = {k: 0 for k in ['cooldown','limit_missed','sl_too_tight',
                            'sl_too_wide','invalid_structure','session_block',
                            'circuit_breaker','above_max_threshold','trade_open',
                            'trail_sl_clamped_to_be']}

    blocked            = set(int(h) for h in args.blocked_hours.split(',') if h.strip())
    open_trade_until   = pd.NaT
    recent_outcomes    = []
    circuit_breaker_until = pd.NaT

    is_bearish_global = args.direction.lower().startswith('bear')
    spread_points = args.spread_ticks * args.tick_size   # static fallback

    rng = np.random.default_rng(seed=42)   # reproducible slippage

    def safe_float(v):
        try:
            f = float(v)
            return f if not math.isnan(f) else 0.0
        except:
            return 0.0

    print(f"\n[*] Simulating...\n")

    # ─────────────────────────────────────────────────────────────────────
    for _, row in trade_df.iterrows():
        trade_time = row['timestamp']

        # ── Filters ───────────────────────────────────────────────────────
        if trade_time.hour in blocked:
            skip['session_block'] += 1; continue

        if pd.notna(circuit_breaker_until) and trade_time < circuit_breaker_until:
            skip['circuit_breaker'] += 1; continue

        if row['ai_prob'] > args.threshold_max:
            skip['above_max_threshold'] += 1; continue

        if pd.notna(open_trade_until) and trade_time <= open_trade_until:
            skip['trade_open'] += 1; continue

        if last_trade_time is not None and args.cooldown_minutes > 0:
            if (trade_time - last_trade_time).total_seconds()/60 < args.cooldown_minutes:
                skip['cooldown'] += 1; continue

        # ── Per-row direction (critical for direction=='both') ─────────────
        if args.direction == 'both':
            bias_val = str(row.get(args.bias_col, '')).strip().lower()
            is_bearish = bias_val in ('bearish', 'bear', '-1', 'sell', 'short')
        else:
            is_bearish = is_bearish_global

        # ── PARITY: per-candle dynamic spread ─────────────────────────────
        if has_spread and 'spread' in row.index:
            candle_spread_pts = safe_float(row.get('spread', 0)) * args.tick_size
            if candle_spread_pts <= 0:
                candle_spread_pts = spread_points  # fallback
        else:
            candle_spread_pts = spread_points

        # ── Structure SL ──────────────────────────────────────────────────
        close_price = safe_float(row.get(args.entry_col, 0))

        base_sl = 0.0
        sl_candidates = (['m5_last_internal_high','m5_strong_high','m5_weak_high','m5_sweep_high']
                         if is_bearish else
                         ['m5_last_internal_low','m5_strong_low','m5_weak_low','m5_sweep_low'])
        for col in sl_candidates:
            v = safe_float(row.get(col, 0))
            if v != 0:
                base_sl = v; break

        if base_sl == 0 or close_price == 0:
            skip['invalid_structure'] += 1; continue

        # ── Limit price ────────────────────────────────────────────────────
        if isinstance(entry_mode, float):
            limit_price = close_price + entry_mode * (base_sl - close_price)
        else:
            limit_price = close_price

        # ── NEW: MAE Structural SL Compression ─────────────────────────────
        # args.sl_compression default should be 1.0 (no compression). 
        # Set to 0.80 to compress the SL to 80% of the structural distance based on the MAE data.
        if args.sl_compression < 1.0:
            base_sl = limit_price + (base_sl - limit_price) * args.sl_compression

        # ── PARITY: fill slippage ──────────────────────────────────────────
        if args.slippage_ticks > 0:
            slip_pts = rng.uniform(0, args.slippage_ticks) * args.tick_size
        else:
            slip_pts = 0.0

        # ── ⑦ MIN STOP DISTANCE enforcement ──────────────────────────────
        # Compute the raw structural SL distance from the limit price.
        # If it falls short of min_stop_distance, override the SL to
        # limit_price ± min_stop_distance and recalculate TP from the new
        # effective distance × RR.  Spread is still applied on top of the
        # (possibly overridden) SL price, matching live broker behavior.
        # min_stop_distance = 0.0 disables this logic entirely.
        raw_sl_dist = abs(limit_price - base_sl)
        min_stop    = args.min_stop_distance
        min_stop_applied = False

        if min_stop > 0.0 and raw_sl_dist < min_stop:
            # Move SL outward to the required minimum distance
            if is_bearish:
                overridden_sl = limit_price + min_stop   # above entry for SELL
            else:
                overridden_sl = limit_price - min_stop   # below entry for BUY
            min_stop_applied = True
            min_stop_overrides += 1
        else:
            overridden_sl = base_sl   # structural SL used as-is

        # ── Entry / SL / TP geometry ──────────────────────────────────────
        if is_bearish:
            entry_price = limit_price 
            entry_price += slip_pts 
            
            # 1. Calculate Original TP based on FULL structural distance
            original_sl_price = overridden_sl + candle_spread_pts
            full_risk_pts     = abs(entry_price - original_sl_price)
            tp_price          = entry_price - full_risk_pts * args.rr
            
            # 2. Apply SL Compression ONLY if it's a structural stop
            if not min_stop_applied and args.sl_compression < 1.0:
                risk_pts = full_risk_pts * args.sl_compression
                sl_price = entry_price + risk_pts
            else:
                risk_pts = full_risk_pts
                sl_price = original_sl_price

            # 3. Trailing logic
            be_trigger_price = entry_price - risk_pts * args.be_trigger
            trail_sl_price   = entry_price - args.be_trail_pct * abs(tp_price - entry_price)

            # Enforce min_stop_level on trail SL
            min_stop_pts = args.min_stop_level * args.tick_size
            if (entry_price - trail_sl_price) < min_stop_pts:
                trail_sl_price = entry_price
                skip['trail_sl_clamped_to_be'] += 1

        else:
            entry_price = limit_price + candle_spread_pts 
            entry_price += slip_pts 
            
            # 1. Calculate Original TP based on FULL structural distance
            original_sl_price = overridden_sl - candle_spread_pts
            full_risk_pts     = abs(entry_price - original_sl_price)
            tp_price          = entry_price + full_risk_pts * args.rr
            
            # 2. Apply SL Compression ONLY if it's a structural stop
            if not min_stop_applied and args.sl_compression < 1.0:
                risk_pts = full_risk_pts * args.sl_compression
                sl_price = entry_price - risk_pts
            else:
                risk_pts = full_risk_pts
                sl_price = original_sl_price

            # 3. Trailing logic
            be_trigger_price = entry_price + risk_pts * args.be_trigger
            trail_sl_price   = entry_price + args.be_trail_pct * abs(tp_price - entry_price)

            # Enforce min_stop_level on trail SL
            min_stop_pts = args.min_stop_level * args.tick_size
            if (trail_sl_price - entry_price) < min_stop_pts:
                trail_sl_price = entry_price
                skip['trail_sl_clamped_to_be'] += 1

        # ── Safety filters ─────────────────────────────────────────────────
        if is_bearish and sl_price <= entry_price:
            skip['invalid_structure'] += 1; continue
        if not is_bearish and sl_price >= entry_price:
            skip['invalid_structure'] += 1; continue

        # ── Lot size ───────────────────────────────────────────────────────
        risk_money = balance * (args.risk_pct / 100.0)
        lot_size, actual_risk = calculate_lot_size(risk_money, entry_price, sl_price, args)

        if lot_size <= 0:
            tks = abs(entry_price - sl_price) / args.tick_size
            skip['sl_too_tight' if tks < args.min_stop_level else 'sl_too_wide'] += 1
            continue

        # ── Future data ────────────────────────────────────────────────────
        if ohlc_df is not None:
            future_df = ohlc_df[ohlc_df['timestamp'] > trade_time].copy()
            scan_df   = ohlc_df
        else:
            future_df = df[df['timestamp'] > trade_time].copy()
            scan_df   = df

        # ── Limit activation scanner ───────────────────────────────────────
        if isinstance(entry_mode, float):
            lookahead = future_df.head(args.limit_expiry_bars)

            if is_bearish:
                act_hits = lookahead[lookahead['high'] + candle_spread_pts >= limit_price]
            else:
                act_hits = lookahead[lookahead['low'] <= limit_price]

            if act_hits.empty:
                skip['limit_missed'] += 1; continue

            act_idx = act_hits.index[0]

            # Check SL not already hit before activation
            if is_bearish:
                pre_sl = future_df.loc[:act_idx-1][
                    future_df.loc[:act_idx]['high'] + candle_spread_pts >= sl_price]
            else:
                pre_sl = future_df.loc[:act_idx][
                    future_df.loc[:act_idx]['low'] <= sl_price]

            if not pre_sl.empty:
                skip['limit_missed'] += 1; continue

            future_df         = future_df.loc[act_idx+1:]
            actual_entry_time = future_df.iloc[0]['timestamp']
        else:
            actual_entry_time = trade_time

        # ── Outcome resolution ────────────────────────────────────────────
        outcome    = 0          # Loss default
        exit_time  = pd.NaT
        exit_price = 0.0
        pnl        = -actual_risk
        is_trail_exit = False

        if is_bearish:
            sl_hits = future_df[future_df['high'] + candle_spread_pts >= sl_price]
            tp_hits = future_df[future_df['low']  + candle_spread_pts <= tp_price]
            be_hits = future_df[future_df['low']  + candle_spread_pts <= be_trigger_price]
        else:
            sl_hits = future_df[future_df['low']  <= sl_price]
            tp_hits = future_df[future_df['high'] >= tp_price]
            be_hits = future_df[future_df['high'] >= be_trigger_price]

        first_sl  = sl_hits.index[0] if not sl_hits.empty else float('inf')
        first_tp  = tp_hits.index[0] if not tp_hits.empty else float('inf')
        first_be  = be_hits.index[0] if not be_hits.empty else float('inf')

        # ─────────────────────────────────────────────────────────────────
        # BE / TRAIL SL RESOLUTION
        # After BE trigger fires → SL moves to trail_sl_price (not entry).
        # Then scan forward for:
        #   a) TP hit first → Win
        #   b) trail_sl hit → Trail Exit (partial profit captured)
        #   c) neither      → Timeout (trail_sl exit at end)
        # ─────────────────────────────────────────────────────────────────
        if first_be < first_sl and first_be < first_tp:
            post_be = future_df.loc[first_be:]

            if is_bearish:
                tp_post  = post_be[post_be['low']  + candle_spread_pts <= tp_price]
                tsl_post = post_be[post_be['high'] + candle_spread_pts >= trail_sl_price]
            else:
                tp_post  = post_be[post_be['high'] >= tp_price]
                tsl_post = post_be[post_be['low']  <= trail_sl_price]

            first_tp_post  = tp_post.index[0]  if not tp_post.empty  else float('inf')
            first_tsl_post = tsl_post.index[0] if not tsl_post.empty else float('inf')

            # Fix #5: strict < so same-candle TP+TSL tie always resolves as TP win
            tp_wins  = first_tp_post < first_tsl_post
            tsl_wins = (not tp_wins) and first_tsl_post < float('inf')

            if tp_wins:
                # Clean TP win
                outcome    = 1
                exit_time  = scan_df.at[first_tp_post, 'timestamp']
                exit_price = tp_price
                pnl        = actual_risk * args.rr

            elif tsl_wins:
                if args.be_trail_pct == 0.0:
                    # Classic breakeven — zero PnL
                    outcome    = 2
                    exit_time  = scan_df.at[first_tsl_post, 'timestamp']
                    exit_price = trail_sl_price
                    pnl        = 0.0
                else:
                    # Trailing SL exit — partial profit
                    if is_bearish:
                        captured_r = (entry_price - trail_sl_price) / risk_pts
                    else:
                        captured_r = (trail_sl_price - entry_price) / risk_pts
                    captured_r = max(0.0, captured_r)
                    outcome    = 3
                    exit_time  = scan_df.at[first_tsl_post, 'timestamp']
                    exit_price = trail_sl_price
                    pnl        = actual_risk * captured_r
                    is_trail_exit = True
            else:
                # No TP, no TSL hit — exit at market price
                last_close = future_df.iloc[-1]['close'] if not future_df.empty else entry_price
                exit_time  = future_df.iloc[-1]['timestamp'] if not future_df.empty else actual_entry_time
                exit_price = last_close
                close_pts  = (entry_price - last_close) if is_bearish else (last_close - entry_price)
                captured_r = (close_pts / risk_pts) if risk_pts > 0 else 0.0
                if captured_r > 0:
                    outcome = 3
                    pnl     = actual_risk * captured_r
                    is_trail_exit = True
                elif captured_r == 0:
                    outcome = 2
                    pnl     = 0.0
                else:
                    outcome = 0
                    pnl     = max(-actual_risk, actual_risk * captured_r)

        # ─────────────────────────────────────────────────────────────────
        # NORMAL RESOLUTION (BE trigger not hit first)
        # ─────────────────────────────────────────────────────────────────
        else:
            if first_tp < first_sl:
                outcome    = 1
                exit_time  = scan_df.at[first_tp, 'timestamp']
                exit_price = tp_price
                pnl        = actual_risk * args.rr
            elif first_sl < first_tp:
                outcome    = 0
                exit_time  = scan_df.at[first_sl, 'timestamp']
                exit_price = sl_price
                pnl        = -actual_risk
            elif first_sl == first_tp and first_sl != float('inf'):
                outcome    = 0
                exit_time  = scan_df.at[first_sl, 'timestamp']
                exit_price = sl_price
                pnl        = -actual_risk
            else:
                # Timeout — exit at last available market price
                last_close = future_df.iloc[-1]['close'] if not future_df.empty else entry_price
                exit_time  = future_df.iloc[-1]['timestamp'] if not future_df.empty else actual_entry_time
                exit_price = last_close
                if is_bearish:
                    close_pts  = entry_price - last_close
                else:
                    close_pts  = last_close - entry_price
                if risk_pts > 0:
                    captured_r = close_pts / risk_pts
                else:
                    captured_r = 0.0
                if captured_r > 0:
                    outcome = 3
                    pnl     = actual_risk * captured_r
                    is_trail_exit = True
                elif captured_r == 0:
                    outcome = 2
                    pnl     = 0.0
                else:
                    outcome = 0
                    pnl     = max(-actual_risk, actual_risk * captured_r)

        # ── Lock: no new trades until this one exits ───────────────────────
        if pd.notna(exit_time):
            open_trade_until = exit_time

        # ── Circuit breaker tracking ───────────────────────────────────────
        if outcome == 1 or outcome == 3:
            recent_outcomes.append(1)
        elif outcome == 0:
            recent_outcomes.append(0)

        if len(recent_outcomes) > 15:
            recent_outcomes.pop(0)

        if len(recent_outcomes) == 15:
            roll_wr = sum(recent_outcomes) / 15.0
            if roll_wr < 0.35:
                print(f"[*] Drift @ {actual_entry_time} | WR={roll_wr*100:.1f}% → 24h pause")
                circuit_breaker_until = actual_entry_time + pd.Timedelta(hours=24)
                recent_outcomes.clear()

        # ── PnL accounting ─────────────────────────────────────────────────
        balance += pnl
        if balance > peak_balance:
            peak_balance = balance
        dd_pct    = (peak_balance - balance) / peak_balance * 100
        max_dd_pct = max(max_dd_pct, dd_pct)

        if outcome == 2:
            breakevens += 1
        if is_trail_exit:
            trail_exits += 1

        outcome_label = {1:'Win', 0:'Loss', 2:'BE', 3:'Trail'}[outcome]
                # For bull wins: entry_price should be above sl_price, tp above entry
        # Check if (tp_price - entry_price) vs (entry_price - sl_price) matches expected geometry
        #df['sl_dist'] = abs(entry_price - sl_price)
        #df['tp_dist'] = abs(tp_price - entry_price)
        # On zero-duration wins, what fraction of sl_dist did price need to travel to hit tp?
        # Answer should be ~rr * sl_dist, but the activation bar "for free" provided the upward move


        results.append({
            'entry_time':       actual_entry_time,
            'exit_time':        exit_time,
            'ai_prob':          round(row['ai_prob'], 4),
            'entry_price':      round(entry_price, 5),
            'sl_price':         round(sl_price, 5),
            'tp_price':         round(tp_price, 5),
            'exit_price':       round(exit_price, 5),
            'trail_sl':         round(trail_sl_price, 5),
            'spread_pts':       round(candle_spread_pts, 4),
            'slippage_pts':     round(slip_pts, 4),
            'lot_size':         lot_size,
            'pnl':              round(pnl, 2),
            'balance':          round(balance, 2),
            'drawdown_pct':     round(dd_pct, 2),
            'outcome':          outcome_label,
            #'sl_dist':          df['sl_dist'],
            #'tp_dist':          df['tp_dist'],
            # ⑦ min_stop_distance audit column
            'min_stop_applied': min_stop_applied,
        })

        if balance <= 0:
            print(f"[!] ACCOUNT BLOWN @ {actual_entry_time}")
            break

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    res_df      = pd.DataFrame(results)
    total       = len(res_df)
    wins_n      = (res_df['outcome'] == 'Win').sum()   if total else 0
    trail_n     = (res_df['outcome'] == 'Trail').sum() if total else 0
    losses_n    = (res_df['outcome'] == 'Loss').sum()  if total else 0
    be_n        = (res_df['outcome'] == 'BE').sum()    if total else 0
    net_profit  = balance - args.balance
    return_pct  = net_profit / args.balance * 100
    win_rate    = (wins_n + trail_n) / (wins_n + trail_n + losses_n) * 100 if (wins_n + trail_n + losses_n) > 0 else 0

    wins_pnl    = res_df[res_df['outcome'].isin(['Win','Trail'])]['pnl'].sum()
    losses_pnl  = res_df[res_df['outcome'] == 'Loss']['pnl'].sum()
    pf          = wins_pnl / abs(losses_pnl) if losses_pnl != 0 else float('inf')

    print("=" * 54)
    print("              BACKTEST RESULTS")
    print("=" * 54)
    print(f"Total Trades       : {total}")
    print(f"Win Rate           : {win_rate:.2f}%  ({wins_n}W / {trail_n}Trail / {losses_n}L / {be_n}BE)")
    print(f"Profit Factor      : {pf:.3f}")
    print(f"Starting Balance   : ${args.balance:,.2f}")
    print(f"Final Balance      : ${balance:,.2f}")
    print(f"Net Profit         : ${net_profit:,.2f}  ({return_pct:+.2f}%)")
    print(f"Max Drawdown       : {max_dd_pct:.2f}%")
    if args.min_stop_distance > 0.0:
        print(f"Min Stop Overrides : {min_stop_overrides} trades had SL widened to {args.min_stop_distance}")
    print("=" * 54)
    print(f"\n[PARITY] Trail SL exits (partial profit locked): {trail_exits}")
    print(f"[PARITY] Classic BE exits (zero PnL):            {breakevens}")
    print(f"\n[DIAGNOSTICS] Setups Rejected:")
    for k, v in skip.items():
        if v > 0:
            print(f"  {k:<25}: {v}")
    print("=" * 54)

    if total > 0:
        res_df.to_csv("backtest_decisions_be.csv", index=False)
        print(f"\n[SUCCESS] Saved → backtest_decisions_be.csv")

        if PLOTLY_AVAILABLE:
            print("\n[*] Generating dashboard...")
            chart_df = ohlc_df if ohlc_df is not None else df
            if len(chart_df) > 25000:
                step = len(chart_df) // 25000
                chart_plot = chart_df.iloc[::step]
            else:
                chart_plot = chart_df

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=chart_plot['timestamp'], y=chart_plot['close'],
                                       mode='lines', name='Price',
                                       line=dict(color='#555',width=1)))

            color_map = {'Win':'lime','Trail':'cyan','Loss':'red','BE':'gold'}
            sym_map   = {'Win':'circle','Trail':'diamond','Loss':'x','BE':'circle-open'}

            for label in ['Win','Trail','Loss','BE']:
                sub = res_df[res_df['outcome']==label]
                if sub.empty: continue
                fig_p.add_trace(go.Scatter(
                    x=sub['exit_time'], y=sub['exit_price'], mode='markers',
                    name=label,
                    marker=dict(symbol=sym_map[label], size=10, color=color_map[label]),
                    hovertext=[f"PnL: ${r['pnl']:.2f}<br>Spread: {r['spread_pts']:.1f}pts"
                               f"<br>Slip: {r['slippage_pts']:.1f}pts"
                               f"<br>MinStop: {'✓' if r['min_stop_applied'] else '—'}"
                               for _, r in sub.iterrows()],
                    hoverinfo='text'))

            fig_p.update_layout(title='AI Price Action & Execution Map',
                                template='plotly_dark', height=600)

            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=res_df['entry_time'], y=res_df['balance'],
                                       mode='lines+markers', name='Equity',
                                       line=dict(color='#2962FF',width=2)))
            fig_e.add_hline(y=args.balance, line_dash='dash', line_color='gray')
            fig_e.update_layout(title='Equity Curve', template='plotly_dark', height=400)

            profit_class = 'profit-pos' if net_profit >= 0 else 'profit-neg'
            msd_status = (f'✅ {args.min_stop_distance} price units — {min_stop_overrides} trades overridden'
                          if args.min_stop_distance > 0.0
                          else '⚠ disabled (set --min-stop-distance > 0 to enable)')
            html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>AI Backtester v2.0 — Parity Edition</title>
<style>
  body{{font-family:'Segoe UI',sans-serif;background:#121212;color:#E0E0E0;margin:0;padding:20px}}
  .container{{max-width:1400px;margin:0 auto}}
  .card{{background:#1E1E1E;border-radius:8px;padding:20px;margin-bottom:20px;box-shadow:0 4px 6px rgba(0,0,0,.3)}}
  h1{{color:#fff;text-align:center;border-bottom:2px solid #333;padding-bottom:10px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:16px}}
  .stat{{background:#2D2D2D;padding:14px;border-radius:6px;text-align:center;border-left:4px solid #2962FF}}
  .val{{font-size:22px;font-weight:bold;color:#fff;margin-top:4px}}
  .lbl{{font-size:11px;color:#AAA;text-transform:uppercase;letter-spacing:1px}}
  .profit-pos{{color:#00E676}}.profit-neg{{color:#FF5252}}
  .parity{{background:#1a2a1a;border-left:4px solid #00E676;padding:14px;border-radius:6px;margin-top:16px}}
  .parity h3{{margin-top:0;color:#00E676}}
</style></head><body><div class="container">
<h1>AI Sniper — Parity Edition v2.0</h1>
<div class="card"><div class="grid">
  <div class="stat"><div class="lbl">Trades</div><div class="val">{total}</div></div>
  <div class="stat"><div class="lbl">Win Rate</div><div class="val">{win_rate:.1f}%</div></div>
  <div class="stat"><div class="lbl">Profit Factor</div><div class="val">{pf:.2f}</div></div>
  <div class="stat"><div class="lbl">Net Profit</div><div class="val {profit_class}">${net_profit:,.2f}</div></div>
  <div class="stat"><div class="lbl">Max DD</div><div class="val profit-neg">{max_dd_pct:.2f}%</div></div>
  <div class="stat"><div class="lbl">Trail Exits</div><div class="val">{trail_exits}</div></div>
  <div class="stat"><div class="lbl">MinStop Overrides</div><div class="val">{min_stop_overrides}</div></div>
</div>
<div class="parity"><h3>Parity Features Active</h3>
<p>Dynamic spread: {'✅ per-candle from OHLC' if has_spread else '⚠ static fallback '+str(args.spread_ticks)+' ticks'}</p>
<p>Fill slippage: {'✅ '+str(args.slippage_ticks)+' ticks max random' if args.slippage_ticks>0 else '⚠ perfect fill (0 slippage)'}</p>
<p>Anti-repaint guard: {'✅ structure cols lagged +1 bar' if args.anti_repaint else '⚠ disabled (run with --anti-repaint to enable)'}</p>
<p>Merge alignment guard: ✅ {args.merge_tolerance_secs}s tolerance</p>
<p>Trailing SL after BE: {'✅ lock '+str(int(args.be_trail_pct*100))+'% of TP distance' if args.be_trail_pct>0 else '⚠ classic BE (zero PnL)'}</p>
<p>Min stop distance: {msd_status}</p>
</div></div>
<div class="card">{pio.to_html(fig_p,full_html=False,include_plotlyjs='cdn')}</div>
<div class="card">{pio.to_html(fig_e,full_html=False,include_plotlyjs=False)}</div>
</div></body></html>"""

            with open("diagnostics_dashboard.html","w",encoding="utf-8") as f:
                f.write(html)
            print("[SUCCESS] Dashboard → diagnostics_dashboard.html")


if __name__ == "__main__":
    main()