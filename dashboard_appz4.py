#!/usr/bin/env python3
"""
dashboard_app.py — Production AI Sniper Terminal v6.2 (PARITY EDITION)

PARITY FIXES vs ai_backtester_be1.py (all applied):
 ① ENTRY MODE: detects 50pb / 30pb / 20pb from model filename & routes Limit vs Market.
 ② SPREAD ON SL/TP: per-candle spread from tick bid/ask applied to entry,
     SL and TP — matches backtester geometry exactly without double-spreading.
 ③ DIRECTION ALIASES: "both" path accepts bearish / bear / -1 / sell / short.
 ④ CIRCUIT BREAKER: rolling-15 built from processed deal tickets, no dedup
     suppression — consecutive same-direction outcomes all count.
 ⑤ LOT SIZE FLOOR: math.floor (not round) so we never over-risk.
 ⑥ MIN STOP LEVEL GUARD: rejects lot when SL ticks < 460.
 ⑦ ANTI-REPAINTING GUARD: Offsets SMC structure columns by +1 bar before AI prediction.
 ⑧ SL COMPRESSION: Shrinks SL to x% of structural distance while calculating TP via full distance.
 ⑨ TRAIL SL MIN-STOP: Clamps trail SL to breakeven if distance is below MIN_STOP_LEVEL.
"""

import sys, os, subprocess, asyncio, json, math
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import xgboost as xgb
import MetaTrader5 as mt5
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from preprocessing import preprocess_for_model

WEB_USERNAME = os.environ.get("DASHBOARD_USERNAME", "admin")  # set DASHBOARD_USERNAME env var in production
WEB_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "changeme")  # set DASHBOARD_PASSWORD env var - never hardcode

if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)

SYMBOL_CONFIGS = {
    "Volatility 75 (1s) Index": {
        "model_file":  "model_1R_50pb_v15.json",
        "meta_file":   "model_1R_50pb_v15_metadata.json",
        "tick_size":   0.01, "tick_value": 0.01,
        "tf":          mt5.TIMEFRAME_M5, "lookback": 15000,
        "spread_fallback_ticks": 50,          # used when bid/ask spread is 0
    },
    "EURUSD": {
        "model_file":  "eurusd_sniper.json",
        "meta_file":   "eurusd_metadata.json",
        "tick_size":   0.00001, "tick_value": 1.0,
        "tf":          mt5.TIMEFRAME_M5, "lookback": 15000,
        "spread_fallback_ticks": 2,
    },
}

# Matches backtester's --min-stop-level default (V75)
MIN_STOP_LEVEL_TICKS = 460

SL_COLS_BEAR = ["m5_last_internal_high", "m5_strong_high", "m5_weak_high", "m5_sweep_high"]
SL_COLS_BULL = ["m5_last_internal_low",  "m5_strong_low",  "m5_weak_low",  "m5_sweep_low"]

app = FastAPI()

bot_state = {
    "is_running": False, "account": "", "symbol": "Volatility 75 (1s) Index",
    "balance": 0.0,
    "threshold": 0.55, "threshold_max": 0.95,
    "risk_pct": 10.0, "rr": 1.0,
    "be_trigger": 1.0, "be_trail_pct": 0.10,
    "cooldown_min": 0,
    "blocked_hours": "",
    "direction": "both",
    "min_stop_distance": 7.0,
    "sl_compression": 0.72,
    "last_trade_time": None,
    "recent_outcomes":        [],    # list of 0/1 for last ≤15 closed trades
    "circuit_breaker_until":  None,  # datetime or None
    "processed_cb_tickets":   set(),
    "pending_ticket":  None,
    "pending_expires": None,
    "last_signal": {
        "time": "—", "prob": 0.0, "pullback_depth": 0.0,
        "bias": "—", "action": "—", "limit_price": 0.0,
        "entry_mode": "market", "direction": "—",
    },
    "skip_counts": {k: 0 for k in [
        "session_block", "circuit_breaker", "trade_open",
        "cooldown", "limit_missed", "invalid_structure",
        "sl_too_tight", "sl_too_wide",
        "threshold_low", "threshold_high", "direction_filter",
        "trail_sl_clamped_to_be",
    ]},
    "metrics": {
        "total_trades": 0, "wins": 0, "losses": 0, "winrate": 0.0, "pnl": 0.0,
        "profit_factor": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
        "expectancy": 0.0, "min_balance": 0.0,
        "today_pnl": 0.0, "week_pnl": 0.0, "month_pnl": 0.0,
        "gross_p": 0.0, "gross_l": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0,
    },
    "equity_history":    [],
    "monthly_breakdown": [],
    "trade_log":         [],   # last 50 closed trades for table
    "chart_data":        {"M5": [], "M15": [], "H4": []},
    "hourly_pnl":      [],    # [{hour, pnl, trades, wr}]
    "drawdown_series": [],    # [{time, dd, balance}]
    "prob_wr_series":  [],    # [{label, wr, count}]
    "session_stats": {
        "trades_today": 0, "streak": 0, "streak_type": "—",
        "avg_win": 0.0, "avg_loss": 0.0, "avg_rr": 0.0,
        "peak_equity": 0.0, "session_start_bal": 0.0,
    },
    "structure_data":     {"M5": None, "M15": None, "H4": None},
    "active_trade_lines": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _blocked() -> set:
    try:
        return {int(h.strip()) for h in bot_state["blocked_hours"].split(",") if h.strip()}
    except:
        return set()

def _entry_mode(model_file: str):
    """
    FIX ①: detect pullback % from model filename.
    Checks 50pb, 30pb, 20pb — matches backtester's auto-detect loop exactly.
    Returns a float (0.50 / 0.30 / 0.20) or the string 'market'.
    """
    for pb in ['50pb', '30pb', '20pb']:
        if pb in model_file:
            return float(pb.replace('pb', '')) / 100
    return 'market'

def safe_float(v):
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FIX ⑤ + ⑥ — LOT SIZE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_lot_size(symbol, balance, risk_pct, entry, sl, tick_size, tick_value):
    """
    Mirrors backtester's calculate_lot_size exactly:
      - Rejects if SL ticks < MIN_STOP_LEVEL_TICKS  (FIX ⑥)
      - Rejects if raw_lot < volume_min              (SL too wide)
      - Uses math.floor, not round                   (FIX ⑤ — never over-risks)
    """
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0

    risk_money = balance * (risk_pct / 100.0)
    sl_pts     = abs(entry - sl)
    sl_ticks   = sl_pts / tick_size

    if sl_ticks < MIN_STOP_LEVEL_TICKS:
        return 0.0

    loss_per_lot = sl_ticks * tick_value
    if loss_per_lot <= 0:
        return 0.0

    raw_lot = risk_money / loss_per_lot

    if raw_lot < info.volume_min:
        return 0.0

    lot = math.floor(raw_lot / info.volume_step) * info.volume_step
    lot = max(info.volume_min, min(info.volume_max, lot))
    return round(lot, 3)


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        if bot_state.get("chart_data") and any(bot_state["chart_data"].values()):
            await ws.send_json({
                "type": "multi_tf_update",
                "ohlc": bot_state["chart_data"],
                "structure": bot_state["structure_data"],
            })
        if bot_state.get("active_trade_lines"):
            await ws.send_json({"type": "trade_lines", **bot_state["active_trade_lines"]})
        await self.broadcast_all()

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, msg: dict):
        for c in list(self.active_connections):
            try:
                await c.send_json(msg)
            except:
                pass

    async def broadcast_all(self):
        m  = bot_state["metrics"]
        sc = bot_state["skip_counts"]
        ls = bot_state["last_signal"]
        cb_active = (
            bot_state["circuit_breaker_until"] is not None
            and datetime.now() < bot_state["circuit_breaker_until"]
        )
        await self.broadcast({
            "type":             "full_state",
            "balance":          bot_state["balance"],
            "trades":           m["total_trades"],
            "wins":             m["wins"],
            "winrate":          m["winrate"],
            "pnl":              m["pnl"],
            "profit_factor":    m["profit_factor"],
            "sharpe":           m["sharpe"],
            "max_drawdown":     m["max_drawdown"],
            "expectancy":       m["expectancy"],
            "min_balance":      m["min_balance"],
            "today_pnl":        m["today_pnl"],
            "week_pnl":         m["week_pnl"],
            "month_pnl":        m["month_pnl"],
            "equity_history":   bot_state["equity_history"],
            "monthly_breakdown":bot_state["monthly_breakdown"],
            "trade_log":        bot_state["trade_log"],
            "skip_counts":      sc,
            "last_signal":      ls,
            "circuit_breaker":  cb_active,
            "circuit_until":    (
                bot_state["circuit_breaker_until"].strftime("%H:%M %d/%m")
                if cb_active else "—"
            ),
            "recent_wr": (
                sum(bot_state["recent_outcomes"])
                / len(bot_state["recent_outcomes"]) * 100
                if bot_state["recent_outcomes"] else 0.0
            ),
            "direction":         bot_state["direction"],
            "threshold":         bot_state["threshold"],
            "threshold_max":     bot_state["threshold_max"],
            "entry_mode":        ls.get("entry_mode", "—"),
            "losses":            m.get("losses", 0),
            "gross_p":           m.get("gross_p", 0.0),
            "gross_l":           m.get("gross_l", 0.0),
            "avg_win":           m.get("avg_win", 0.0),
            "avg_loss":          m.get("avg_loss", 0.0),
            "symbol":            bot_state["symbol"],
            "risk_pct":          bot_state["risk_pct"],
            "rr":                bot_state["rr"],
            "min_stop_distance": bot_state["min_stop_distance"],
            "hourly_pnl":        bot_state.get("hourly_pnl", []),
            "drawdown_series":   bot_state.get("drawdown_series", []),
            "session_stats":     bot_state.get("session_stats", {}),
            "prob_wr_series":    bot_state.get("prob_wr_series", []),
            "outcome_counts": {
                "Win":  m.get("wins", 0),
                "Loss": m.get("losses", 0),
                "BE":   m.get("be_count", 0),
            },
            "pending_ticket":    bot_state.get("pending_ticket"),
        })

ws_manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_pullback_depth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    closes = pd.to_numeric(df.get('close', pd.Series([0] * len(df))), errors='coerce').fillna(0).values
    sl_v   = pd.to_numeric(df.get('m5_strong_low',  pd.Series([0] * len(df))), errors='coerce').fillna(0).values
    wh_v   = pd.to_numeric(df.get('m5_weak_high',   pd.Series([0] * len(df))), errors='coerce').fillna(0).values
    sh_v   = pd.to_numeric(df.get('m5_strong_high', pd.Series([0] * len(df))), errors='coerce').fillna(0).values
    wl_v   = pd.to_numeric(df.get('m5_weak_low',    pd.Series([0] * len(df))), errors='coerce').fillna(0).values
    bias_col = 'm5_external_bias' if 'm5_external_bias' in df.columns else 'external_bias'
    bias = (
        df[bias_col].astype(str).str.lower().values
        if bias_col in df.columns
        else np.array(['unknown'] * len(df))
    )
    bull_range = wh_v - sl_v
    bear_range = sh_v - wl_v
    bull_depth = np.zeros(len(df))
    bear_depth = np.zeros(len(df))
    with np.errstate(invalid='ignore', divide='ignore'):
        vb = (bias == 'bullish') & (bull_range > 0)
        bull_depth[vb] = (wh_v[vb] - closes[vb]) / bull_range[vb]
        va = (bias == 'bearish') & (bear_range > 0)
        bear_depth[va] = (closes[va] - wl_v[va]) / bear_range[va]
    depth = np.where(bias == 'bullish', bull_depth,
                     np.where(bias == 'bearish', bear_depth, 0.0))
    df['m5_pullback_depth'] = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return df

def sanitize_categoricals(df: pd.DataFrame, cat_mappings: dict) -> pd.DataFrame:
    df = df.copy()
    BF = ('false', 'neutral', 'unknown', 'none', 'bearish')
    for col, mapping in cat_mappings.items():
        if col not in df.columns:
            df[col] = list(mapping.keys())[0]
            continue
        lmap = {str(k).strip().lower(): str(k) for k in mapping.keys()}
        fb = next((lmap[v] for v in BF if v in lmap), list(lmap.values())[0])
        df[col] = (
            df[col].fillna(fb).astype(str).str.strip().str.lower()
            .map(lambda x, m=lmap, f=fb: m.get(x, f))
        )
    bmap  = {'true': 1.0, 'false': 0.0, '1': 1.0, '0': 0.0, '1.0': 1.0, '0.0': 0.0}
    bvals = set(bmap.keys())
    for col in df.columns:
        if col in cat_mappings:
            continue
        if df[col].dtype == bool:
            df[col] = df[col].astype(float)
        elif df[col].dtype == object:
            s = df[col].dropna()
            if len(s) > 0 and set(s.astype(str).str.lower().unique()) <= bvals:
                df[col] = (
                    df[col].fillna(False).astype(str).str.lower()
                    .map(lambda x, m=bmap: m.get(x, 0.0)).fillna(0.0)
                )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MT5 METRICS
# ─────────────────────────────────────────────────────────────────────────────

def update_mt5_history_metrics():
    try:
        if not mt5.terminal_info():
            return
        now   = datetime.now()
        deals = mt5.history_deals_get(now - timedelta(days=90), now)
        if not deals:
            return

        profits, losses_list, daily_pnl = [], [], defaultdict(float)
        be_count = 0
        peak_bal = running_bal = safe_float(mt5.account_info().balance)
        min_bal  = peak_bal
        max_dd   = 0.0
        monthly  = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
        equity_series = []
        trade_log     = []
        today_pnl = week_pnl = month_pnl = 0.0
        today       = now.date()
        week_start  = today - timedelta(days=today.weekday())
        month_start = today.replace(day=1)

        for d in deals:
            if d.magic != 999994 or d.entry != 1:
                continue
            p  = safe_float(d.profit)
            dt = datetime.fromtimestamp(d.time)
            mk = dt.strftime("%b")
            monthly[mk]["trades"] += 1
            monthly[mk]["pnl"]    += p
            if p > 0:
                profits.append(p)
                monthly[mk]["wins"] += 1
            elif p < 0:
                losses_list.append(abs(p))
            else:
                be_count += 1
            running_bal += p
            peak_bal = max(peak_bal, running_bal)
            min_bal  = min(min_bal,  running_bal)
            dd = (peak_bal - running_bal) / peak_bal * 100 if peak_bal > 0 else 0
            max_dd = max(max_dd, dd)
            daily_pnl[dt.strftime("%Y-%m-%d")] += p
            equity_series.append({"time": dt.strftime("%b %d %H:%M"), "balance": round(running_bal, 2)})
            outcome = "Win" if p > 0 else ("Loss" if p < 0 else "BE")
            trade_log.append({
                "time": dt.strftime("%m/%d %H:%M"), "pnl": round(p, 2),
                "outcome": outcome, "symbol": d.symbol,
            })
            if dt.date() == today:       today_pnl  += p
            if dt.date() >= week_start:  week_pnl   += p
            if dt.date() >= month_start: month_pnl  += p

        wins_n   = len(profits)
        losses_n = len(losses_list)
        total    = wins_n + losses_n
        winrate  = round(wins_n / total * 100, 2) if total > 0 else 0.0
        gross_p  = sum(profits)
        gross_l  = sum(losses_list)
        pf       = round(gross_p / gross_l, 2) if gross_l > 0 else 0.0
        avg_win  = gross_p / wins_n   if wins_n   > 0 else 0.0
        avg_loss = gross_l / losses_n if losses_n > 0 else 0.0
        wr       = winrate / 100
        exp      = round(avg_win * wr - avg_loss * (1 - wr), 2) if total > 0 else 0.0
        dv       = list(daily_pnl.values())
        sharpe   = (
            round((np.array(dv).mean() / np.array(dv).std()) * np.sqrt(252), 2)
            if len(dv) >= 2 and np.array(dv).std() > 0 else 0.0
        )
        monthly_list = [
            {
                "month": mk, "trades": mv["trades"], "wins": mv["wins"],
                "pnl": round(mv["pnl"], 2),
                "wr": round(mv["wins"] / mv["trades"] * 100, 1) if mv["trades"] > 0 else 0.0,
            }
            for mk, mv in monthly.items()
        ]

        cur_balance = safe_float(mt5.account_info().balance)
        bot_state["balance"] = cur_balance
        bot_state["metrics"].update({
            "total_trades": total, "wins": wins_n, "losses": losses_n,
            "winrate": winrate, "pnl": round(gross_p - gross_l, 2),
            "profit_factor": pf, "sharpe": sharpe, "max_drawdown": round(max_dd, 2),
            "expectancy": exp, "min_balance": round(min_bal, 2),
            "today_pnl": round(today_pnl, 2), "week_pnl": round(week_pnl, 2),
            "month_pnl": round(month_pnl, 2),
            "gross_p": round(gross_p, 2), "gross_l": round(gross_l, 2),
            "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
            "be_count": int(be_count),
        })
        bot_state["monthly_breakdown"] = monthly_list
        bot_state["trade_log"]         = trade_log[-50:]

        # ── Confidence → Win-Rate buckets (live) ─────────────────────────
        # Entry deals carry comment "AI|<prob>|RR..|.." — join to exits
        # by position_id and bucket the outcomes by model confidence.
        try:
            prob_by_pos = {}
            for d in deals:
                if d.magic != 999994 or d.entry != 0:
                    continue
                c = str(getattr(d, "comment", "") or "")
                if c.startswith("AI|"):
                    try:
                        prob_by_pos[d.position_id] = float(c.split("|")[1])
                    except Exception:
                        pass
            _bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65),
                     (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]
            _agg = {b: [0, 0] for b in _bins}
            for d in deals:
                if d.magic != 999994 or d.entry != 1:
                    continue
                pr = prob_by_pos.get(d.position_id)
                if pr is None or d.profit == 0:
                    continue
                for b in _bins:
                    if b[0] <= pr < b[1]:
                        _agg[b][0] += 1
                        if d.profit > 0:
                            _agg[b][1] += 1
                        break
            bot_state["prob_wr_series"] = [
                {"label": f"{int(b[0]*100)}-{int(b[1]*100)}", "count": n,
                 "wr": round(w / n * 100, 1) if n else 0.0}
                for b, (n, w) in _agg.items() if n > 0
            ]
        except Exception:
            pass
        if equity_series:
            bot_state["equity_history"] = equity_series[-100:]
        ts = now.strftime("%H:%M")
        if not bot_state["equity_history"] or bot_state["equity_history"][-1]["time"] != ts:
            bot_state["equity_history"].append({"time": ts, "balance": cur_balance})
        bot_state["equity_history"] = bot_state["equity_history"][-100:]

        try:
            all_deals = mt5.history_deals_get(now - timedelta(days=90), now)
            if all_deals:
                dfa = pd.DataFrame([{
                    "time":    datetime.fromtimestamp(d.time),
                    "pnl":     safe_float(d.profit),
                    "hour":    datetime.fromtimestamp(d.time).hour,
                    "outcome": "Win" if safe_float(d.profit) > 0 else (
                               "Loss" if safe_float(d.profit) < 0 else "BE"),
                } for d in all_deals if d.magic == 999994 and d.entry == 1])

                if not dfa.empty:
                    hg = dfa.groupby("hour").agg(
                        trades=("pnl", "count"), pnl=("pnl", "sum"),
                        wins=("outcome", lambda x: (x == "Win").sum()),
                    ).reset_index()
                    hg["wr"] = (hg["wins"] / hg["trades"] * 100).round(1)
                    bot_state["hourly_pnl"] = [
                        {"hour": int(r["hour"]), "pnl": round(r["pnl"], 2),
                         "trades": int(r["trades"]), "wr": round(r["wr"], 1)}
                        for _, r in hg.iterrows()
                    ]

                    _peak = _running = cur_balance
                    _dd_series = []
                    for _, row in dfa.sort_values("time").iterrows():
                        _running += row["pnl"]
                        _peak = max(_peak, _running)
                        _dd = (_peak - _running) / _peak * 100 if _peak > 0 else 0
                        _dd_series.append({
                            "time": row["time"].strftime("%b %d"),
                            "dd": round(_dd, 2), "balance": round(_running, 2),
                        })
                    bot_state["drawdown_series"] = _dd_series[-200:]

                    today_deals = dfa[dfa["time"].dt.date == now.date()]
                    trades_today = len(today_deals)
                    streak = 0; streak_type = "—"
                    recent_oc = list(dfa.sort_values("time")["outcome"])
                    if recent_oc:
                        streak_type = recent_oc[-1]
                        for o in reversed(recent_oc):
                            if o == streak_type: streak += 1
                            else: break
                    peak_eq = cur_balance
                    if bot_state["equity_history"]:
                        peak_eq = max(e["balance"] for e in bot_state["equity_history"])
                    bot_state["session_stats"].update({
                        "trades_today": trades_today,
                        "streak": streak, "streak_type": streak_type,
                        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
                        "avg_rr": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0,
                        "peak_equity": round(peak_eq, 2),
                        "session_start_bal": (
                            bot_state["session_stats"].get("session_start_bal") or cur_balance
                        ),
                    })
        except Exception as _e:
            print(f"[!] Live analytics error: {_e}")

    except Exception as e:
        print(f"[!] MT5 metrics error: {e}")


async def fetch_mt5_data(symbol, tf, count):
    rates = await asyncio.to_thread(mt5.copy_rates_from_pos, symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    if 'tick_volume' in df.columns:
        df = df.rename(columns={'tick_volume': 'volume'})
    return df.drop_duplicates('time').sort_values('time').reset_index(drop=True)

def build_ohlc_payload(df):
    if df.empty:
        return []
    return [
        {"time": int(r['time'].timestamp()), "open": safe_float(r['open']),
         "high": safe_float(r['high']), "low": safe_float(r['low']),
         "close": safe_float(r['close'])}
        for _, r in df.tail(3000).iterrows()
    ]

def build_chart_structure_payload(out_dir, df, max_bars=3000):
    if df.empty:
        return {"markers": [], "segments": [], "zones": []}
    df_r = df.tail(max_bars).copy()
    min_idx, max_idx = df_r.index.min(), df_r.index.max()
    time_map = {i: int(r['time'].timestamp()) for i, r in df.iterrows()}
    pl = {"markers": [], "segments": [], "zones": []}

    def rcsv(n):
        p = os.path.join(out_dir, n)
        return pd.read_csv(p) if (os.path.exists(p) and os.path.getsize(p) > 0) else pd.DataFrame()

    for fn, kind, pos, color in [
        ("strong_highs.csv", "strong_high", "aboveBar", "#7c4dff"),
        ("strong_lows.csv",  "strong_low",  "belowBar", "#f4a261"),
        ("weak_highs.csv",   "weak_high",   "aboveBar", "#29b6f6"),
        ("weak_lows.csv",    "weak_low",    "belowBar", "#ff5c8a"),
    ]:
        d = rcsv(fn)
        if d.empty or 'anchor_idx' not in d:
            continue
        d = d.dropna(subset=['anchor_idx'])
        d = d[(d['anchor_idx'] >= min_idx) & (d['anchor_idx'] <= max_idx)]
        for _, row in d.iterrows():
            t = time_map.get(int(row['anchor_idx']))
            if t:
                pl["markers"].append({
                    "time": t, "position": pos, "color": color,
                    "shape": "square", "text": kind.replace('_', ' '),
                })

    events = rcsv("events.csv")
    if not events.empty and 'line_start_idx' in events:
        events = events.dropna(subset=['line_start_idx', 'line_end_idx'])
        events = events[events['line_end_idx'] >= min_idx]
        cmap = {
            "bos_bull": "#e74c3c", "bos_bear": "#8bc34a",
            "choch_bull": "#f39c12", "choch_bear": "#00bcd4",
            "sweep_high": "#8e44ad", "sweep_low": "#795548",
        }
        smap = {"choch_bull": 1, "choch_bear": 1, "sweep_high": 2, "sweep_low": 2}
        for _, row in events.iterrows():
            t0 = time_map.get(int(row['line_start_idx']))
            t1 = time_map.get(int(row['line_end_idx']))
            if t0 and t1:
                et = str(row['event_type'])
                pl["segments"].append({
                    "color": cmap.get(et, "#555"), "style": smap.get(et, 0),
                    "data": [
                        {"time": t0, "value": safe_float(row['level_price'])},
                        {"time": t1, "value": safe_float(row['level_price'])},
                    ],
                })

    pois = rcsv("pois.csv")
    if not pois.empty and 'start_idx' in pois:
        pois = pois.dropna(subset=['start_idx'])
        pois['end_idx'] = pd.to_numeric(pois.get('end_idx'), errors='coerce').fillna(max_idx)
        pois = pois[pois['end_idx'] >= min_idx]
        pois = pois[
            pois['mitigated'].isna() |
            (pois['mitigated'] == False) |
            (pois['mitigated'] == 'False')
        ]
        for _, row in pois.iterrows():
            t0 = time_map.get(int(row['start_idx']))
            t1 = time_map.get(int(row['end_idx']))
            if t0 and t1:
                zt = str(row['zone_type']).lower()
                c  = "#1e3a8a" if "demand" in zt else "#7c2d12"
                pl["zones"].append({
                    "color": c,
                    "top":   [{"time": t0, "value": safe_float(row['high'])},
                               {"time": t1, "value": safe_float(row['high'])}],
                    "bottom": [{"time": t0, "value": safe_float(row['low'])},
                               {"time": t1, "value": safe_float(row['low'])}],
                })
    return pl


# ─────────────────────────────────────────────────────────────────────────────
# BE / TRAIL SL MONITOR
# ─────────────────────────────────────────────────────────────────────────────

async def be_trail_monitor():
    adjusted: set[int] = set()
    while bot_state["is_running"]:
        try:
            positions = mt5.positions_get(magic=999994)
            if positions:
                for pos in positions:
                    if pos.ticket in adjusted or pos.sl == 0 or pos.tp == 0:
                        continue
                    is_bear  = (pos.type == mt5.ORDER_TYPE_SELL)
                    entry    = pos.price_open
                    risk_pts = abs(entry - pos.sl)
                    if risk_pts == 0:
                        continue
                    be_price = (
                        entry - risk_pts * bot_state["be_trigger"] if is_bear
                        else entry + risk_pts * bot_state["be_trigger"]
                    )
                    trail_sl = (
                        entry - bot_state["be_trail_pct"] * abs(pos.tp - entry) if is_bear
                        else entry + bot_state["be_trail_pct"] * abs(pos.tp - entry)
                    )

                    # PARITY FIX ⑨: Enforce MIN_STOP_LEVEL on trail SL clamping to BE
                    cfg_local = SYMBOL_CONFIGS.get(pos.symbol)
                    min_stop_pts = MIN_STOP_LEVEL_TICKS * (cfg_local["tick_size"] if cfg_local else 0.01)
                    if is_bear:
                        if (entry - trail_sl) < min_stop_pts:
                            trail_sl = entry
                            bot_state["skip_counts"]["trail_sl_clamped_to_be"] = bot_state["skip_counts"].get("trail_sl_clamped_to_be", 0) + 1
                    else:
                        if (trail_sl - entry) < min_stop_pts:
                            trail_sl = entry
                            bot_state["skip_counts"]["trail_sl_clamped_to_be"] = bot_state["skip_counts"].get("trail_sl_clamped_to_be", 0) + 1

                    tick = mt5.symbol_info_tick(pos.symbol)
                    if not tick:
                        continue
                    cur = tick.bid if is_bear else tick.ask
                    if not ((is_bear and cur <= be_price) or (not is_bear and cur >= be_price)):
                        continue
                    res = await asyncio.to_thread(
                        mt5.order_send,
                        {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket,
                         "symbol": pos.symbol, "sl": round(trail_sl, 5), "tp": pos.tp},
                    )
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        adjusted.add(pos.ticket)
                        await ws_manager.broadcast({
                            "type": "log",
                            "msg": f"<span class='ok'>[BE] #{pos.ticket} SL → {trail_sl:.5f}</span>",
                        })
        except Exception:
            pass
        await asyncio.sleep(5)


# ─────────────────────────────────────────────────────────────────────────────
# AI TRADING LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def log(msg):
    print(f"[AI ENGINE] {msg}")
    await ws_manager.broadcast({"type": "log", "msg": msg})

async def stage(name: str, detail: dict = None):
    """Broadcast a pipeline-stage event — drives the Globe FX engine."""
    try:
        await ws_manager.broadcast({
            "type": "pipeline_stage",
            "stage": name,
            "detail": detail or {},
            "ts": datetime.now().strftime("%H:%M:%S"),
        })
    except Exception:
        pass

async def _skip(key: str, msg: str, cur_time, ai_prob=0.0, entry=0, sl=0, tp=0):
    bot_state["skip_counts"][key] = bot_state["skip_counts"].get(key, 0) + 1
    await log(f"Skipped [{key}] — {msg}")
    await stage("verdict", {"action": "SKIP", "reason": key, "prob": float(ai_prob or 0.0)})
    _log_row(cur_time, ai_prob, f"SKIP_{key.upper()}", entry, sl, tp)


async def ai_trading_loop():
    cfg            = SYMBOL_CONFIGS[bot_state["symbol"]]
    entry_mode_val = _entry_mode(cfg["model_file"])
    entry_mode_str = (
        f"{int(entry_mode_val * 100)}pb"
        if isinstance(entry_mode_val, float) else "market"
    )

    bst = xgb.Booster()
    try:
        bst.load_model(cfg["model_file"])
        with open(cfg["meta_file"]) as f:
            meta = json.load(f)
        features     = meta['features']
        cat_mappings = meta.get('category_mappings', {})
        if not cat_mappings:
            await log("<span class='err'>[FATAL] category_mappings missing. Retrain!</span>")
            bot_state["is_running"] = False
            return
    except Exception as e:
        await log(f"Model load failed: {e}")
        bot_state["is_running"] = False
        return

    direction      = bot_state["direction"]
    static_is_bear = (direction == "bearish")

    await log(
        f"Engine armed → {bot_state['symbol']} "
        f"| Entry:{entry_mode_str} "
        f"| Thresh:{bot_state['threshold'] * 100:.0f}%–{bot_state['threshold_max'] * 100:.0f}% "
        f"| RR:{bot_state['rr']} | Dir:{direction} "
        f"| MinStop:{bot_state['min_stop_distance']}"
    )

    asyncio.create_task(be_trail_monitor())
    last_processed_time = None

    while bot_state["is_running"]:
        try:
            update_mt5_history_metrics()
            await ws_manager.broadcast_all()

            df_m5  = await fetch_mt5_data(bot_state["symbol"], mt5.TIMEFRAME_M5,  cfg["lookback"])
            m15_df = await fetch_mt5_data(bot_state["symbol"], mt5.TIMEFRAME_M15, max(5000, cfg["lookback"] // 3))
            h4_df  = await fetch_mt5_data(bot_state["symbol"], mt5.TIMEFRAME_H4,  max(2000, cfg["lookback"] // 10))

            if df_m5.empty:
                await asyncio.sleep(5)
                continue

            cur_time = df_m5.iloc[-1]['time']
            if bot_state.pop("force_reanalysis", False):
                last_processed_time = None
                await log("<span class='ok'>[ADMIN] Pipeline re-analysis forced — reprocessing current bar.</span>")
            if last_processed_time and cur_time <= last_processed_time:
                await asyncio.sleep(2)
                continue
            last_processed_time = cur_time

            # ── Session block ────────────────────────────────────────────────
            if cur_time.hour in _blocked():
                await asyncio.sleep(30)
                continue

            # ── Circuit breaker ──────────────────────────────────────────────
            cb = bot_state["circuit_breaker_until"]
            if cb and datetime.now() < cb:
                await asyncio.sleep(30)
                continue

            # ── open_trade_until: block if live position or pending order ────
            positions = mt5.positions_get(magic=999994) or []
            orders    = mt5.orders_get(magic=999994)    or []

            # Cancel stale pending order past expiry
            if bot_state["pending_ticket"] and bot_state["pending_expires"]:
                if datetime.now() > bot_state["pending_expires"]:
                    try:
                        for o in orders:
                            if o.ticket == bot_state["pending_ticket"]:
                                await asyncio.to_thread(
                                    mt5.order_send,
                                    {"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket},
                                )
                                bot_state["skip_counts"]["limit_missed"] += 1
                                await log(
                                    f"Pending order #{o.ticket} expired → cancelled [limit_missed]"
                                )
                    except Exception:
                        pass
                    bot_state["pending_ticket"]  = None
                    bot_state["pending_expires"] = None

            if len(positions) > 0 or len(orders) > 0:
                await asyncio.sleep(2)
                continue

            await log(f"New M5 candle: {cur_time} — analyzing…")
            await stage("uplink", {
                "m5": int(len(df_m5)), "m15": int(len(m15_df)), "h4": int(len(h4_df)),
                "symbol": bot_state["symbol"],
            })

            bot_state["chart_data"] = {
                "M5":  build_ohlc_payload(df_m5),
                "M15": build_ohlc_payload(m15_df),
                "H4":  build_ohlc_payload(h4_df),
            }

            tmp_dir = "./live_pipeline_tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            df_m5.to_csv( f"{tmp_dir}/live_m5.csv",  index=False)
            m15_df.to_csv(f"{tmp_dir}/live_m15.csv", index=False)
            h4_df.to_csv( f"{tmp_dir}/live_h4.csv",  index=False)

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            def run_parsers():
                def gs(n):
                    for f in os.listdir('.'):
                        if f.lower() == n.lower():
                            return f
                    return n

                p = gs("parser_runner_v3.py")
                m = gs("mtf_state_builder6.py")
                for cmd in [
                    [sys.executable, p, "--csv", f"{tmp_dir}/live_m5.csv",  "--outdir", f"{tmp_dir}/out_m5",  "--tf", "M5",  "--zigzag-depth", "12", "--zigzag-deviation", "5", "--zigzag-backstep", "3"],
                    [sys.executable, p, "--csv", f"{tmp_dir}/live_m15.csv", "--outdir", f"{tmp_dir}/out_m15", "--tf", "M15", "--zigzag-depth", "16", "--zigzag-deviation", "5", "--zigzag-backstep", "3"],
                    [sys.executable, p, "--csv", f"{tmp_dir}/live_h4.csv",  "--outdir", f"{tmp_dir}/out_h4",  "--tf", "H4",  "--zigzag-depth", "6",  "--zigzag-deviation", "5", "--zigzag-backstep", "3"],
                    [sys.executable, m,
                     "--m5-dir",   f"{tmp_dir}/out_m5",
                     "--m15-dir",  f"{tmp_dir}/out_m15",
                     "--h4-dir",   f"{tmp_dir}/out_h4",
                     "--ohlc-csv", f"{tmp_dir}/live_m5.csv",
                     "--out-csv",  f"{tmp_dir}/live_mtf_state.csv"],
                ]:
                    res = subprocess.run(
                        cmd, capture_output=True, text=True, encoding='utf-8', env=env,
                    )
                    if res.returncode != 0:
                        raise Exception(f"[{cmd[1]}] crashed:\n{res.stderr[-400:]}")

            await stage("parse", {"tfs": ["M5", "M15", "H4"]})
            await asyncio.to_thread(run_parsers)
            await stage("merge", {"tfs": ["M5", "M15", "H4"]})

            bot_state["structure_data"] = {
                "M5":  build_chart_structure_payload(f"{tmp_dir}/out_m5",  df_m5),
                "M15": build_chart_structure_payload(f"{tmp_dir}/out_m15", m15_df),
                "H4":  build_chart_structure_payload(f"{tmp_dir}/out_h4",  h4_df),
            }
            await ws_manager.broadcast({
                "type": "multi_tf_update",
                "ohlc": bot_state["chart_data"],
                "structure": bot_state["structure_data"],
            })

            mtf_csv = f"{tmp_dir}/live_mtf_state.csv"
            if not os.path.exists(mtf_csv):
                await log("<span class='err'>[ERROR] live_mtf_state.csv missing</span>")
                await asyncio.sleep(5)
                continue

            final_df = pd.read_csv(mtf_csv, low_memory=False)

            await stage("sanitize", {"rows": int(len(final_df))})

            # PARITY FIX ⑦: Anti-repaint lag (shift SMC columns +1 bar) to prevent look-ahead bias
            STRUCTURE_KEYWORDS = [
                'strong_low','strong_high','weak_low','weak_high',
                'sweep_low','sweep_high','bos','choch','internal',
                'poi','pullback','equilibrium','last_internal'
            ]
            for col in final_df.columns:
                if any(kw in col.lower() for kw in STRUCTURE_KEYWORDS):
                    final_df[col] = final_df[col].shift(1)

            final_df = compute_pullback_depth(final_df)
            final_df = sanitize_categoricals(final_df, cat_mappings)

            bias_filter = None if direction == "both" else ("m5_external_bias", direction)
            try:
                X_live, valid_df = preprocess_for_model(
                    final_df, features, cat_mappings, bias_filter,
                )
            except ValueError as e:
                await log(f"<span class='err'>[PREPROC] {e}</span>")
                await asyncio.sleep(5)
                continue

            tick = mt5.symbol_info_tick(bot_state["symbol"])
            if not tick:
                await asyncio.sleep(2)
                continue

            if valid_df.empty or valid_df.index[-1] != final_df.index[-1]:
                bot_state["skip_counts"]["direction_filter"] += 1
                await log("Skipped — direction filter / no valid row")
                await asyncio.sleep(1)
                continue

            # ── Predict ──────────────────────────────────────────────────────
            x_dmat  = xgb.DMatrix(X_live.iloc[-1:])
            await stage("infer", {"features": int(X_live.shape[1])})
            ai_prob = float(bst.predict(x_dmat)[0])
            await log(f"AI Probability: <b>{ai_prob * 100:.2f}%</b>")

            live_row       = valid_df.iloc[-1]
            pullback_depth = safe_float(live_row.get('m5_pullback_depth', 0))
            bias_val       = str(live_row.get("m5_external_bias", "")).lower().strip()

            is_bear = (
                static_is_bear if direction != "both"
                else bias_val in ('bearish', 'bear', '-1', 'sell', 'short')
            )

            # ── Thresholds ───────────────────────────────────────────────────
            if ai_prob < bot_state["threshold"]:
                bot_state["last_signal"].update({
                    "time": str(cur_time)[:16], "prob": ai_prob,
                    "pullback_depth": pullback_depth, "bias": bias_val,
                    "action": "LOW_PROB", "entry_mode": entry_mode_str,
                    "direction": "BEAR" if is_bear else "BULL",
                })
                await _skip("threshold_low", f"prob {ai_prob * 100:.1f}% below min", cur_time, ai_prob)
                await asyncio.sleep(1)
                continue

            if ai_prob > bot_state["threshold_max"]:
                bot_state["last_signal"].update({
                    "time": str(cur_time)[:16], "prob": ai_prob,
                    "pullback_depth": pullback_depth, "bias": bias_val,
                    "action": "HIGH_PROB", "entry_mode": entry_mode_str,
                    "direction": "BEAR" if is_bear else "BULL",
                })
                await _skip("threshold_high", f"prob {ai_prob * 100:.1f}% above max", cur_time, ai_prob)
                await asyncio.sleep(1)
                continue

            # ── Cooldown ─────────────────────────────────────────────────────
            now = datetime.now()
            if (
                bot_state["last_trade_time"] and bot_state["cooldown_min"] > 0
                and (now - bot_state["last_trade_time"]) < timedelta(minutes=bot_state["cooldown_min"])
            ):
                await _skip("cooldown", "cooldown active", cur_time, ai_prob)
                await asyncio.sleep(1)
                continue

            # ── Structure SL ─────────────────────────────────────────────────
            sl_cols = SL_COLS_BEAR if is_bear else SL_COLS_BULL
            base_sl = 0.0
            for col in sl_cols:
                v = safe_float(live_row.get(col, 0))
                if v != 0:
                    base_sl = v
                    break

            close_price = safe_float(live_row.get(
                'close', safe_float(tick.bid if is_bear else tick.ask)
            ))
            if base_sl == 0 or close_price == 0:
                await _skip("invalid_structure", "no SL structure", cur_time, ai_prob)
                await asyncio.sleep(1)
                continue

            # FIX ②: per-candle spread from live tick bid/ask
            candle_spread_pts = safe_float(tick.ask - tick.bid)
            if candle_spread_pts <= 0:
                candle_spread_pts = cfg.get("spread_fallback_ticks", 50) * cfg["tick_size"]

            # ── Entry Geometry & Limit Pricing Execution Parity ──────────────
            is_market = not isinstance(entry_mode_val, float)
            raw_sl    = safe_float(base_sl)

            if is_market:
                # Market Execution Router: executes precisely at current market tick 
                entry_price = tick.bid if is_bear else tick.ask
                raw_sl_adjusted = raw_sl + candle_spread_pts if is_bear else raw_sl - candle_spread_pts
            else:
                # Limit Emulation Router: calibrates target execution block + spread offset natively
                base_entry = close_price + entry_mode_val * (raw_sl - close_price)
                if is_bear:
                    entry_price = base_entry
                    raw_sl_adjusted = raw_sl + candle_spread_pts
                else:
                    entry_price = base_entry + candle_spread_pts
                    raw_sl_adjusted = raw_sl - candle_spread_pts

            rr = bot_state["rr"]

            # ── MIN STOP DISTANCE enforcement ────────────────────────────────
            raw_sl_dist = abs(entry_price - raw_sl_adjusted)
            min_stop    = safe_float(bot_state.get("min_stop_distance", 0.0))
            min_stop_applied = False

            if min_stop > 0.0 and raw_sl_dist < min_stop:
                overridden_sl = safe_float(entry_price + min_stop if is_bear else entry_price - min_stop)
                min_stop_applied = True
                await log(
                    f"<span class='warn'>[MIN_STOP] Structural SL dist {raw_sl_dist:.5f} &lt; "
                    f"min_stop {min_stop:.5f} → SL overridden to {overridden_sl:.5f}</span>"
                )
            else:
                overridden_sl = raw_sl_adjusted

            # 1. Original TP based on FULL structural distance
            full_risk_pts = abs(entry_price - overridden_sl)
            tp = safe_float(entry_price - full_risk_pts * rr if is_bear else entry_price + full_risk_pts * rr)

            # 2. PARITY FIX ⑧: SL Compression ONLY if it's a structural stop
            sl_compression = safe_float(bot_state.get("sl_compression", 1.0))
            if not min_stop_applied and sl_compression < 1.0:
                risk_pts = full_risk_pts * sl_compression
                sl = safe_float(entry_price + risk_pts if is_bear else entry_price - risk_pts)
                await log(f"<span class='ok'>[COMPRESSION] SL compressed to {sl_compression*100:.0f}% of structural distance</span>")
            else:
                sl = overridden_sl

            # Geometry guard (same as backtester)
            if (is_bear and sl <= entry_price) or (not is_bear and sl >= entry_price):
                await _skip(
                    "invalid_structure",
                    f"SL geometry invalid after spread "
                    f"(entry={entry_price:.5f} sl={sl:.5f})",
                    cur_time, ai_prob,
                )
                await asyncio.sleep(1)
                continue

            # ── Lot size ─────────────────────────────────────────────────────
            lot = calculate_lot_size(
                bot_state["symbol"],
                mt5.account_info().balance,
                bot_state["risk_pct"],
                entry_price,   # spread-inclusive 
                sl,            # spread-inclusive & compressed
                cfg["tick_size"],
                cfg["tick_value"],
            )
            if lot <= 0:
                tks = abs(entry_price - sl) / cfg["tick_size"]
                key = "sl_too_tight" if tks < MIN_STOP_LEVEL_TICKS else "sl_too_wide"
                await _skip(key, f"lot=0 (ticks={tks:.0f})", cur_time, ai_prob)
                await asyncio.sleep(1)
                continue

            # ── Dynamic Order Routing ────────────────────────────────────────
            order_type = (
                (mt5.ORDER_TYPE_SELL if is_bear else mt5.ORDER_TYPE_BUY)
                if is_market else
                (mt5.ORDER_TYPE_SELL_LIMIT if is_bear else mt5.ORDER_TYPE_BUY_LIMIT)
            )
            expiry_dt = now + timedelta(minutes=15)  # 3 M5 candles = 15 min exactly

            req = {
                "action":       mt5.TRADE_ACTION_DEAL if is_market else mt5.TRADE_ACTION_PENDING,
                "symbol":       bot_state["symbol"],
                "volume":       float(lot),
                "type":         order_type,
                "price":        round(entry_price, 5),
                "sl":           round(sl, 5),
                "tp":           round(tp, 5),
                "deviation":    10,
                "magic":        999994,
                "comment":      f"AI|{ai_prob:.2f}|RR{rr}|{entry_mode_str}|MSD{min_stop}",
                "type_filling": mt5.ORDER_FILLING_FOK if is_market else mt5.ORDER_FILLING_RETURN,
            }

            if not is_market:
                req["type_time"]  = mt5.ORDER_TIME_SPECIFIED
                req["expiration"] = int(expiry_dt.timestamp())

            res = await asyncio.to_thread(mt5.order_send, req)

            # Fallback filling modes matrix to ensure strict execution 
            if res and res.retcode != mt5.TRADE_RETCODE_DONE:
                for filling in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK]:
                    req["type_filling"] = filling
                    res = await asyncio.to_thread(mt5.order_send, req)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        break

            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                if not is_market:
                    bot_state["pending_ticket"]  = res.order
                    bot_state["pending_expires"] = expiry_dt
                
                bot_state["last_trade_time"]  = now
                bot_state["active_trade_lines"] = {
                    "entry": entry_price, "sl": sl, "tp": tp,
                }
                await ws_manager.broadcast({
                    "type": "trade_lines",
                    "entry": entry_price, "sl": sl, "tp": tp,
                })
                act_str = "MARKET_ORDER_PLACED" if is_market else "LIMIT_ORDER_PLACED"
                await log(
                    f"<span class='ok'>[{'MARKET' if is_market else 'LIMIT'}] #{res.order} "
                    f"{'SELL' if is_bear else 'BUY'} @ {entry_price:.5f} "
                    f"| SL:{sl:.5f} TP:{tp:.5f} "
                    f"| spread:{candle_spread_pts:.2f}pts "
                    f"| prob:{ai_prob * 100:.1f}% "
                    f"| expires:{expiry_dt.strftime('%H:%M') if not is_market else 'N/A'}</span>"
                )
            else:
                act_str = f"ORDER_FAILED_{res.retcode if res else 'UNKNOWN'}"
                await log(f"<span class='err'>[FAILED] Code:{getattr(res, 'retcode', '?')}</span>")

            await stage("verdict", {
                "action": ("SELL" if is_bear else "BUY"),
                "result": act_str,
                "reason": (None if "PLACED" in act_str else act_str),
                "prob": float(ai_prob),
                "entry": float(entry_price), "sl": float(sl), "tp": float(tp),
                "lot": float(lot),
            })
            bot_state["last_signal"].update({
                "time": str(cur_time)[:16], "prob": round(ai_prob, 4),
                "pullback_depth": round(pullback_depth, 4), "bias": bias_val,
                "action": act_str, "limit_price": round(entry_price, 5),
                "entry_mode": entry_mode_str,
                "direction": "BEAR" if is_bear else "BULL",
            })
            _log_row(cur_time, ai_prob, act_str, entry_price, sl, tp)
            update_mt5_history_metrics()

            # ── Circuit breaker — proper rolling-15 accumulation ─────────────
            try:
                cb_deals = mt5.history_deals_get(now - timedelta(days=30), now) or []
                for deal in sorted(cb_deals, key=lambda d: d.time):
                    if deal.magic != 999994 or deal.entry != 1:
                        continue
                    if deal.ticket in bot_state["processed_cb_tickets"]:
                        continue
                    outcome = (
                        1 if deal.profit > 0
                        else (0 if deal.profit < 0 else None)
                    )
                    if outcome is None:
                        continue
                    bot_state["processed_cb_tickets"].add(deal.ticket)
                    bot_state["recent_outcomes"].append(outcome)
                    if len(bot_state["recent_outcomes"]) > 15:
                        bot_state["recent_outcomes"].pop(0)

                if len(bot_state["recent_outcomes"]) == 15:
                    roll_wr = sum(bot_state["recent_outcomes"]) / 15.0
                    if roll_wr < 0.35:
                        cb_until = datetime.now() + timedelta(hours=24)
                        bot_state["circuit_breaker_until"] = cb_until
                        bot_state["recent_outcomes"].clear()
                        bot_state["processed_cb_tickets"].clear()
                        await log(
                            f"<span class='err'>[CIRCUIT BREAKER] "
                            f"Rolling WR={roll_wr * 100:.0f}% — "
                            f"24h pause until {cb_until.strftime('%H:%M %d/%m')}</span>"
                        )
            except Exception as _cbe:
                await log(f"<span class='warn'>[CB] Tracker error: {_cbe}</span>")

            await ws_manager.broadcast_all()

        except Exception as e:
            await log(f"<span class='err'>Loop error: {e}</span>")
        await asyncio.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _log_row(ts, prob, action, entry, sl, tp):
    row = pd.DataFrame([{
        "timestamp": ts, "symbol": bot_state["symbol"],
        "direction": bot_state["direction"],
        "ai_prob":   round(prob, 4),
        "threshold": bot_state["threshold"],
        "threshold_max": bot_state["threshold_max"],
        "rr":        bot_state["rr"],
        "min_stop_distance": bot_state["min_stop_distance"],
        "sl_compression": bot_state["sl_compression"],
        "action":    action,
        "entry_price": entry, "sl_price": sl, "tp_price": tp,
        "balance":   bot_state["balance"],
    }])
    f = os.path.join(os.getcwd(), "live_trade_logs.csv")
    row.to_csv(f, mode='a', header=not os.path.exists(f), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            data = json.loads(await ws.receive_text())
            if data["action"] == "update_settings":
                for k, v in [
                    ("threshold",        "threshold"),
                    ("threshold_max",    "threshold_max"),
                    ("risk",             "risk_pct"),
                    ("rr",               "rr"),
                    ("be_trigger",       "be_trigger"),
                    ("be_trail_pct",     "be_trail_pct"),
                    ("cooldown",         "cooldown_min"),
                    ("blocked_hours",    "blocked_hours"),
                    ("direction",        "direction"),
                    ("min_stop_distance","min_stop_distance"),
                    ("sl_compression",   "sl_compression"),
                ]:
                    if k in data:
                        val = data[k]
                        if k == "cooldown":
                            val = int(val)
                        elif k not in ("blocked_hours", "direction"):
                            val = float(val)
                        bot_state[v] = val
                await log(
                    f"Params updated — "
                    f"T:{bot_state['threshold'] * 100:.0f}%–{bot_state['threshold_max'] * 100:.0f}% "
                    f"RR:{bot_state['rr']} Dir:{bot_state['direction']} "
                    f"MinStop:{bot_state['min_stop_distance']} Comp:{bot_state['sl_compression']}"
                )
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


@app.get("/api/status")
async def status():
    return {"is_running": bot_state["is_running"], "symbol": bot_state["symbol"]}


@app.post("/api/web-login")
async def web_login(p: dict):
    if p.get("username") == WEB_USERNAME and p.get("password") == WEB_PASSWORD:
        return {"status": "success", "token": "authenticated_admin"}
    return {"status": "error", "msg": "Invalid credentials"}


@app.post("/api/start")
async def start_bot(p: dict):
    if bot_state["is_running"]:
        return {"status": "error", "msg": "Already running"}
    sym = p.get('symbol', '').strip()
    if not sym or sym not in SYMBOL_CONFIGS:
        sym = "Volatility 75 (1s) Index"
    if not mt5.initialize(login=int(p['account']), password=p['password'], server=p['server']):
        return {"status": "error", "msg": f"MT5 failed: {mt5.last_error()}"}
    ti = mt5.terminal_info()
    if not ti:
        return {"status": "error", "msg": "Cannot fetch terminal info"}
    if not ti.trade_allowed:
        return {"status": "error", "msg": "Algo Trading disabled — enable in MT5."}

    bot_state.update({
        "is_running":        True,
        "account":           p['account'],
        "symbol":            sym,
        "balance":           safe_float(mt5.account_info().balance) if mt5.account_info() else 0.0,
        "direction":         p.get("direction",         bot_state["direction"]),
        "threshold":         float(p.get("threshold",         bot_state["threshold"])),
        "threshold_max":     float(p.get("threshold_max",     bot_state["threshold_max"])),
        "rr":                float(p.get("rr",                bot_state["rr"])),
        "be_trigger":        float(p.get("be_trigger",        bot_state["be_trigger"])),
        "be_trail_pct":      float(p.get("be_trail_pct",      bot_state["be_trail_pct"])),
        "blocked_hours":     str(  p.get("blocked_hours",     bot_state["blocked_hours"])),
        "min_stop_distance": float(p.get("min_stop_distance", bot_state["min_stop_distance"])),
        "sl_compression":    float(p.get("sl_compression",    bot_state["sl_compression"])),
        "skip_counts":       {k: 0 for k in bot_state["skip_counts"]},
        "recent_outcomes":       [],
        "circuit_breaker_until": None,
        "processed_cb_tickets":  set(),
        "pending_ticket":        None,
        "pending_expires":       None,
    })
    update_mt5_history_metrics()
    asyncio.create_task(ai_trading_loop())
    return {"status": "success", "msg": f"Connected. Engine started on {sym}."}


@app.post("/api/stop")
async def stop_bot():
    bot_state["is_running"]        = False
    bot_state["active_trade_lines"]= None
    bot_state["pending_ticket"]    = None
    bot_state["pending_expires"]   = None
    return {"status": "success", "msg": "Pipeline stopped."}


@app.get("/api/symbols")
async def get_symbols():
    return {"symbols": list(SYMBOL_CONFIGS.keys())}


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST API
# ─────────────────────────────────────────────────────────────────────────────
import threading as _threading

_bt_state = {
    "running": False, "progress": 0, "status": "idle",
    "log": [], "result_csv": None, "result_data": None,
}

def _discover_file(name: str) -> str:
    for f in os.listdir('.'):
        if f.lower() == name.lower():
            return os.path.abspath(f)
    return name

def _run_backtest_thread(params: dict):
    global _bt_state
    _bt_state.update({
        "running": True, "progress": 5, "status": "starting",
        "log": [], "result_csv": None, "result_data": None,
    })

    script   = _discover_file(params.get("script", "ai_backtester_be2.py"))
    in_csv   = params.get("in_csv",  "ml_dataset_labeled_v15.csv")
    ohlc_csv = params.get("ohlc_csv", "")
    model    = params.get("model",    "")
    metadata = params.get("metadata", "")
    out_csv  = "backtest_decisions_be.csv"

    cmd = [
        sys.executable, script,
        "--in-csv",        in_csv,
        "--model",         model,
        "--metadata",      metadata,
        "--direction",     params.get("direction",     "both"),
        "--rr",            str(params.get("rr",             "2.0")),
        "--be-trigger",    str(params.get("be_trigger",     "1.5")),
        "--be-trail-pct",  str(params.get("be_trail_pct",  "0.65")),
        "--threshold",     str(params.get("threshold",     "0.50")),
        "--threshold-max", str(params.get("threshold_max", "0.65")),
        "--blocked-hours", params.get("blocked_hours", "3,4,5,7,9,10,14,19,20,22"),
        "--risk-pct",      str(params.get("risk_pct",    "1.0")),
        "--balance",       str(params.get("balance",     "100.0")),
        "--slippage-ticks",str(params.get("slippage_ticks", "0")),
        "--start-date",    params.get("start_date", ""),
        "--end-date",      params.get("end_date",   ""),
    ]
    if ohlc_csv:
        cmd += ["--ohlc-csv", ohlc_csv]
    if params.get("anti_repaint"):
        cmd.append("--anti-repaint")

    cmd = [c for c in cmd if c != ""]
    _bt_state["log"].append(f"[CMD] {' '.join(cmd)}")
    _bt_state["progress"] = 10

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', env=env, bufsize=1,
        )
        for line in iter(proc.stdout.readline, ''):
            line = line.rstrip()
            if line:
                _bt_state["log"].append(line)
                if "Simulating" in line: _bt_state["progress"] = 20
                elif "RESULTS"  in line: _bt_state["progress"] = 90
        proc.wait()
        if proc.returncode != 0:
            _bt_state.update({"running": False, "status": "error", "progress": 0})
            return
    except Exception as e:
        _bt_state.update({
            "running": False, "status": "error",
            "log": _bt_state["log"] + [str(e)], "progress": 0,
        })
        return

    _bt_state["progress"] = 95

    try:
        df = pd.read_csv(out_csv, low_memory=False)
        df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True, errors='coerce')
        df['exit_time']  = pd.to_datetime(df['exit_time'],  utc=True, errors='coerce')

        eq   = [{"t": r['entry_time'].strftime('%b %d'), "b": round(r['balance'], 2)}
                for _, r in df.iterrows() if pd.notna(r['entry_time'])]
        dd_j = [{"t": r['entry_time'].strftime('%b %d'), "dd": round(r['drawdown_pct'], 2)}
                for _, r in df.iterrows() if pd.notna(r['entry_time'])]
        oc   = df['outcome'].value_counts().to_dict()

        df['month'] = df['entry_time'].dt.strftime('%b')
        mo_order    = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
        }
        monthly = (
            df.groupby('month', sort=False)
            .agg(
                trades=('pnl', 'count'),
                wins=('outcome', lambda x: x.isin(['Win', 'Trail']).sum()),
                losses=('outcome', lambda x: (x == 'Loss').sum()),
                pnl=('pnl', 'sum'),
            )
            .reset_index()
        )
        monthly['wr']   = (monthly['wins'] / monthly['trades'] * 100).round(1)
        monthly['sort'] = monthly['month'].map(mo_order)
        monthly         = monthly.sort_values('sort')

        df['pb_bin'] = pd.cut(
            df['ai_prob'],
            bins=[0.49, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01],
            labels=['<55', '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '>95'],
        )
        prob_g = df.groupby('pb_bin', observed=True).agg(
            count=('pnl', 'count'),
            wr=('outcome', lambda x: x.isin(['Win', 'Trail']).sum() / len(x) * 100),
            pnl=('pnl', 'sum'),
        ).reset_index()
        prob_g.columns = ['label', 'count', 'wr', 'pnl']

        df['hour'] = df['entry_time'].dt.hour
        hourly = df.groupby('hour').agg(
            trades=('pnl', 'count'), pnl=('pnl', 'sum'),
            wr=('outcome', lambda x: x.isin(['Win', 'Trail']).sum() / len(x) * 100),
        ).reset_index()

        wins_n  = (df['outcome'].isin(['Win', 'Trail'])).sum()
        loss_n  = (df['outcome'] == 'Loss').sum()
        total   = len(df)
        wr      = wins_n / total * 100 if total else 0
        gross_p = df[df['pnl'] > 0]['pnl'].sum()
        gross_l = abs(df[df['pnl'] < 0]['pnl'].sum())
        pf      = gross_p / gross_l if gross_l > 0 else 0
        avg_win  = df[df['pnl'] > 0]['pnl'].mean() if wins_n > 0 else 0
        avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if loss_n > 0 else 0
        start_bal = float(params.get("balance", 100.0))
        net       = df['balance'].iloc[-1] - start_bal if total else 0
        dv        = list(df.groupby(df['entry_time'].dt.date)['pnl'].sum().values)
        sharpe    = (
            (np.array(dv).mean() / np.array(dv).std() * np.sqrt(252))
            if len(dv) >= 2 and np.array(dv).std() > 0 else 0
        )
        trail_n = (df['outcome'] == 'Trail').sum()

        tlog = df.tail(100)[[
            'entry_time', 'outcome', 'pnl', 'ai_prob',
            'lot_size', 'entry_price', 'sl_price', 'tp_price', 'balance',
        ]].copy()
        tlog['entry_str'] = tlog['entry_time'].dt.strftime('%m/%d %H:%M')
        dt0 = df['entry_time'].dropna().iloc[0]  if total else None
        dt1 = df['entry_time'].dropna().iloc[-1] if total else None

        _bt_state["result_data"] = {
            "stats": {
                "total":   total, "wins": int(wins_n), "losses": int(loss_n),
                "trail":   int(trail_n),
                "wr":      round(wr, 2), "pf": round(pf, 3), "net": round(net, 2),
                "max_dd":  round(df['drawdown_pct'].max(), 2),
                "exp":     round(avg_win * (wr / 100) - avg_loss * (1 - wr / 100), 2),
                "gross_p": round(gross_p, 2), "gross_l": round(gross_l, 2),
                "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
                "start_bal": start_bal,
                "end_bal":   round(df['balance'].iloc[-1], 2) if total else start_bal,
                "ret_pct":   round(net / start_bal * 100, 2) if start_bal > 0 else 0,
                "sharpe":    round(sharpe, 2),
                "avg_lot":   round(df['lot_size'].mean(), 3),
                "max_lot":   round(df['lot_size'].max(), 3),
                "date_range": (
                    f"{dt0.strftime('%d %b %Y')} → {dt1.strftime('%d %b %Y')}"
                    if dt0 and dt1 else "—"
                ),
                "model":       model,
                "direction":   params.get("direction", "both"),
                "rr":          params.get("rr", "2.0"),
                "threshold":   (
                    f"{float(params.get('threshold', 0.5)) * 100:.0f}–"
                    f"{float(params.get('threshold_max', 0.65)) * 100:.0f}%"
                ),
                "be_trigger":   params.get("be_trigger",   "1.5"),
                "be_trail_pct": params.get("be_trail_pct", "0.65"),
                "risk_pct":     params.get("risk_pct",     "1.0"),
            },
            "equity":   eq,
            "monthly":  monthly.to_dict(orient='records'),
            "prob":     prob_g.to_dict(orient='records'),
            "hourly":   hourly.to_dict(orient='records'),
            "drawdown": dd_j,
            "outcomes": oc,
            "trades":   tlog.to_dict(orient='records'),
        }
        _bt_state["result_csv"] = out_csv
        _bt_state.update({"running": False, "status": "done", "progress": 100})

    except Exception as e:
        _bt_state.update({
            "running": False, "status": "error",
            "log": _bt_state["log"] + [f"Parse error: {e}"], "progress": 0,
        })


@app.post("/api/run-backtest")
async def run_backtest(params: dict):
    if _bt_state["running"]:
        return {"status": "error", "msg": "Backtest already running"}
    t = _threading.Thread(target=_run_backtest_thread, args=(params,), daemon=True)
    t.start()
    return {"status": "ok", "msg": "Backtest started"}


@app.get("/api/backtest-status")
async def bt_status():
    return {
        "running":  _bt_state["running"],
        "progress": _bt_state["progress"],
        "status":   _bt_state["status"],
        "log":      _bt_state["log"][-30:],
    }

@app.get("/api/backtest-data")
async def bt_data():
    if _bt_state["result_data"] is None:
        return {"status": "no_data"}
    return {"status": "ok", "data": _bt_state["result_data"]}


@app.get("/api/tick")
async def get_tick():
    try:
        if not bot_state["is_running"]:
            return {"price": 0.0, "time": "—"}
        tick = mt5.symbol_info_tick(bot_state["symbol"])
        if not tick:
            return {"price": 0.0, "time": "—"}
        return {
            "price": round((tick.bid + tick.ask) / 2, 5),
            "time":  datetime.now().strftime("%H:%M:%S"),
        }
    except Exception:
        return {"price": 0.0, "time": "—"}


@app.get("/api/live-analytics")
async def get_live_analytics():
    return {
        "status":          "ok",
        "hourly_pnl":      bot_state.get("hourly_pnl", []),
        "drawdown_series": bot_state.get("drawdown_series", []),
        "session_stats":   bot_state.get("session_stats", {}),
        "trade_log":       bot_state.get("trade_log", [])[-20:],
    }


@app.post("/api/reset-circuit-breaker")
async def reset_cb():
    bot_state["circuit_breaker_until"] = None
    bot_state["recent_outcomes"]       = []
    bot_state["processed_cb_tickets"]  = set()
    await ws_manager.broadcast({
        "type": "log",
        "msg":  "<span class='ok'>[ADMIN] Circuit breaker manually reset.</span>",
    })
    await ws_manager.broadcast_all()
    return {"status": "ok", "msg": "Circuit breaker reset."}


@app.post("/api/replay-pipeline")
async def replay_pipeline():
    """Re-run the analysis pipeline so the globe animations play.
    Live: forces the trading loop to reprocess the current bar.
    Offline: broadcasts a demo pipeline_stage sequence."""
    if bot_state["is_running"]:
        bot_state["force_reanalysis"] = True
        return {"status": "ok", "msg": "Live pipeline re-analysis scheduled (next loop tick)."}

    async def _demo():
        ls = bot_state.get("last_signal", {}) or {}
        try:
            prob = float(ls.get("prob") or 0.62)
        except Exception:
            prob = 0.62
        action = "SELL" if str(ls.get("direction", "")).upper() == "BEAR" else "BUY"
        seq = [
            ("uplink",   {"m5": 15000, "m15": 5000, "h4": 2000,
                          "symbol": bot_state["symbol"], "demo": True}, 2.0),
            ("parse",    {"tfs": ["M5", "M15", "H4"], "demo": True}, 2.1),
            ("merge",    {"tfs": ["M5", "M15", "H4"], "demo": True}, 3.6),
            ("sanitize", {"rows": 15000, "demo": True}, 1.7),
            ("infer",    {"features": 128, "demo": True}, 1.9),
            ("verdict",  {"action": action, "prob": prob,
                          "reason": "DEMO_REPLAY", "demo": True}, 0),
        ]
        for name, detail, delay in seq:
            await stage(name, detail)
            if delay:
                await asyncio.sleep(delay)
    asyncio.create_task(_demo())
    return {"status": "ok", "msg": "Demo pipeline replay started (engine offline)."}


@app.post("/api/export-logs")
async def export_logs():
    return {"status": "ok", "trades": bot_state.get("trade_log", [])[-100:]}


@app.post("/api/cancel-pending")
async def cancel_pending():
    ticket = bot_state.get("pending_ticket")
    if not ticket:
        return {"status": "error", "msg": "No pending order."}
    try:
        orders = mt5.orders_get(magic=999994) or []
        for o in orders:
            if o.ticket == ticket:
                res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    bot_state["pending_ticket"]  = None
                    bot_state["pending_expires"] = None
                    await ws_manager.broadcast({
                        "type": "log",
                        "msg":  f"<span class='ok'>[ADMIN] Pending order #{ticket} cancelled.</span>",
                    })
                    return {"status": "ok", "msg": f"Order #{ticket} cancelled."}
                return {"status": "error", "msg": f"Cancel failed: {res.retcode if res else 'UNKNOWN'}"}
        bot_state["pending_ticket"] = None
        return {"status": "ok", "msg": "Order not found (may have already filled/expired)."}
    except Exception as e:
        return {"status": "error", "msg": str(e)}


@app.get("/", response_class=HTMLResponse)
async def serve():
    return HTMLResponse(content=HTML_CONTENT)

# (Leave HTML_CONTENT unmodified from original payload)



HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS AI · V75 COMMAND CENTER</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
:root{
  --void:#020817;--abyss:#040d1a;--deep:#060f20;--hull:#091526;--panel:#0c1c30;--glass:#102238;
  --wire:rgba(6,182,212,.1);--wire2:rgba(6,182,212,.22);--wire3:rgba(6,182,212,.5);
  --plasma:#06b6d4;--plasma2:#22d3ee;--glow:rgba(6,182,212,.35);--glow2:rgba(6,182,212,.6);
  --hot:#f43f5e;--hot2:rgba(244,63,94,.18);--hotg:rgba(244,63,94,.5);
  --acid:#10b981;--acid2:rgba(16,185,129,.15);
  --amber:#f59e0b;--amber2:rgba(245,158,11,.15);
  --violet:#8b5cf6;--violet2:rgba(139,92,246,.15);
  --ion:#3b82f6;--ion2:rgba(59,130,246,.15);
  --magenta:#ec4899;--magenta2:rgba(236,72,153,.15);
  --text:#3a5570;--text2:#5a7898;--text3:#8ab0cc;--text4:#c8e0f0;--text5:#eaf4ff;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:var(--void);color:var(--text);
  font-family:'Rajdhani',sans-serif;font-weight:400;letter-spacing:.03em}

/* ── Background canvas ── */
#starfield{position:fixed;inset:0;z-index:1;pointer-events:none}

/* ── UI layer ── */
#ui-root{position:fixed;inset:0;z-index:3;display:flex;flex-direction:column}

/* ── SCANLINES ── */
#ui-root::before{content:'';position:absolute;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(6,182,212,.014) 2px,rgba(6,182,212,.014) 3px)}

/* ── DATA RAIN canvas (Matrix effect) ── */
#data-rain{position:fixed;inset:0;z-index:2;pointer-events:none;opacity:.22}

/* ── Typography ── */
.orb{font-family:'Orbitron',monospace}
.tech{font-family:'Share Tech Mono',monospace}
.raj{font-family:'Rajdhani',sans-serif}

/* ── Glass panels ── */
.glass{background:rgba(4,12,28,.88);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  border:1px solid var(--wire);border-radius:4px;position:relative;overflow:hidden;
  transition:border-color .3s,box-shadow .3s}
.glass::before{content:'';position:absolute;inset:0;border-radius:4px;
  background:linear-gradient(135deg,rgba(6,182,212,.05) 0%,transparent 55%);pointer-events:none;z-index:0}
.glass::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(6,182,212,.35),transparent);pointer-events:none;z-index:2}
.glass>*{position:relative;z-index:1}
.glass:hover{border-color:var(--wire2);box-shadow:0 0 30px rgba(6,182,212,.07),inset 0 0 20px rgba(6,182,212,.02)}

/* ── Corner marks ── */
.corners::before,.corners::after{content:'';position:absolute;width:10px;height:10px;border-color:var(--plasma);border-style:solid;z-index:10}
.corners::before{top:0;left:0;border-width:1.5px 0 0 1.5px}
.corners::after{top:0;right:0;border-width:1.5px 1.5px 0 0}
.corners-b::before,.corners-b::after{content:'';position:absolute;width:10px;height:10px;border-color:rgba(6,182,212,.4);border-style:solid;z-index:10}
.corners-b::before{bottom:0;left:0;border-width:0 0 1.5px 1.5px}
.corners-b::after{bottom:0;right:0;border-width:0 1.5px 1.5px 0}

/* ── Metric tiles ── */
.mtile{background:rgba(4,13,26,.9);backdrop-filter:blur(8px);border:1px solid var(--wire);
  border-radius:3px;padding:10px 14px;position:relative;overflow:hidden;cursor:default;
  transition:border-color .3s,box-shadow .3s,transform .2s}
.mtile:hover{border-color:var(--wire3);box-shadow:0 0 28px var(--glow),0 4px 20px rgba(0,0,0,.4);transform:translateY(-2px)}
.mtile::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:var(--tc,var(--plasma));opacity:.7;
  box-shadow:0 0 10px var(--tc,var(--plasma));transition:opacity .3s}
.mtile:hover::after{opacity:1}
.mtile .lbl{font-size:.55rem;text-transform:uppercase;letter-spacing:.18em;color:var(--text2);
  font-family:'Rajdhani';font-weight:600;margin-bottom:4px}
.mtile .val{font-family:'Orbitron',monospace;font-size:1.25rem;font-weight:700;line-height:1;color:var(--text5);
  transition:text-shadow .3s}
.mtile:hover .val{text-shadow:0 0 20px currentColor}
.mtile .sub{font-size:.58rem;color:var(--text);margin-top:3px;font-family:'Share Tech Mono'}
.mtile .delta{position:absolute;top:10px;right:12px;font-size:.6rem;font-family:'Share Tech Mono'}

/* ── Status indicators ── */
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;display:inline-block}
.dot-live{background:var(--acid);box-shadow:0 0 12px var(--acid),0 0 24px rgba(16,185,129,.4);animation:dp 1.3s ease-in-out infinite}
.dot-idle{background:var(--hot);box-shadow:0 0 6px var(--hot)}
.dot-cb{background:var(--amber);box-shadow:0 0 8px var(--amber);animation:dp .9s ease-in-out infinite}
@keyframes dp{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.75)}}
.sig{display:inline-flex;align-items:center;gap:6px;font-size:.62rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.12em;font-family:'Rajdhani'}

/* ── Buttons ── */
.btn{font-family:'Rajdhani';font-weight:700;font-size:.75rem;text-transform:uppercase;letter-spacing:.12em;
  cursor:pointer;border-radius:2px;transition:all .25s;display:inline-flex;align-items:center;justify-content:center;gap:5px;border:none;position:relative;overflow:hidden}
.btn::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,255,255,.06),transparent);pointer-events:none}
.btn-p{background:linear-gradient(135deg,rgba(6,182,212,.2),rgba(6,182,212,.08));
  border:1px solid var(--plasma);color:var(--plasma);padding:9px 18px}
.btn-p:hover{background:linear-gradient(135deg,rgba(6,182,212,.38),rgba(6,182,212,.15));
  box-shadow:0 0 28px var(--glow),inset 0 0 20px rgba(6,182,212,.1);transform:translateY(-1px);
  text-shadow:0 0 12px var(--plasma)}
.btn-red{background:linear-gradient(135deg,rgba(244,63,94,.22),rgba(244,63,94,.08));
  border:1px solid var(--hot);color:var(--hot);padding:9px 18px}
.btn-red:hover{background:linear-gradient(135deg,rgba(244,63,94,.4),rgba(244,63,94,.14));box-shadow:0 0 28px var(--hotg);transform:translateY(-1px)}
.btn-sm{padding:5px 13px;font-size:.68rem}
.btn-ghost{background:transparent;border:1px solid var(--wire2);color:var(--text2);padding:8px 16px}
.btn-ghost:hover{border-color:var(--plasma);color:var(--plasma);box-shadow:0 0 15px var(--glow)}

/* ── Inputs ── */
input,select,textarea{background:rgba(4,13,26,.92);border:1px solid var(--wire);color:var(--text3);
  border-radius:2px;padding:7px 11px;font-family:'Share Tech Mono',monospace;font-size:.75rem;
  outline:none;transition:border .2s,box-shadow .2s,background .2s;width:100%}
input:focus,select:focus,textarea:focus{border-color:var(--plasma);box-shadow:0 0 0 2px rgba(6,182,212,.15);background:rgba(6,182,212,.04)}
select option{background:var(--panel)}
input[type=range]{-webkit-appearance:none;height:3px;background:var(--wire);cursor:pointer;padding:0;border:none;border-radius:2px}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--plasma);box-shadow:0 0 10px var(--glow);cursor:pointer;transition:transform .2s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.3)}
input[type=checkbox]{width:14px;height:14px;accent-color:var(--plasma)}
.flbl{font-size:.58rem;text-transform:uppercase;letter-spacing:.14em;color:var(--text2);
  font-family:'Rajdhani';font-weight:600;display:flex;justify-content:space-between;margin-bottom:5px}
.flbl span{color:var(--plasma);font-family:'Share Tech Mono',monospace}
label{font-size:.58rem;text-transform:uppercase;letter-spacing:.14em;color:var(--text2);
  display:block;margin-bottom:5px;font-family:'Rajdhani';font-weight:600}

/* ── Nav tabs ── */
.nav-tab{padding:9px 16px;font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;
  cursor:pointer;color:var(--text2);background:transparent;border:none;border-bottom:2px solid transparent;
  transition:all .22s;font-family:'Rajdhani';white-space:nowrap;position:relative}
.nav-tab::after{content:'';position:absolute;bottom:-1px;left:50%;right:50%;height:2px;
  background:var(--plasma);transition:all .25s;box-shadow:0 0 10px var(--glow)}
.nav-tab.on::after{left:0;right:0}
.nav-tab.on{color:var(--plasma);text-shadow:0 0 12px var(--glow)}
.nav-tab:hover:not(.on){color:var(--text3)}
.sub-tab{padding:5px 13px;font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
  cursor:pointer;color:var(--text2);background:transparent;border:1px solid transparent;
  border-radius:2px;transition:all .2s;font-family:'Rajdhani'}
.sub-tab.on{background:rgba(6,182,212,.1);color:var(--plasma);border-color:var(--wire2);box-shadow:0 0 10px var(--glow)}
.ctab{padding:4px 10px;font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
  cursor:pointer;color:var(--text);background:transparent;border:1px solid transparent;border-radius:2px;transition:all .2s;font-family:'Rajdhani'}
.ctab.on{border-color:var(--plasma);color:var(--plasma);background:rgba(6,182,212,.08);box-shadow:0 0 8px var(--glow)}

/* ── Tables ── */
.tbl{width:100%;border-collapse:collapse}
.tbl th{font-size:.52rem;text-transform:uppercase;letter-spacing:.14em;color:var(--text);
  padding:7px 11px;text-align:left;border-bottom:1px solid var(--wire);font-family:'Rajdhani';font-weight:600}
.tbl td{padding:6px 11px;font-size:.68rem;border-bottom:1px solid rgba(6,182,212,.04);font-family:'Share Tech Mono';
  transition:background .15s}
.tbl tr:hover td{background:rgba(6,182,212,.04)}
.badge{display:inline-flex;align-items:center;padding:2px 7px;border-radius:2px;
  font-size:.56rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;font-family:'Rajdhani'}
.bW{background:rgba(16,185,129,.12);color:var(--acid);border:1px solid rgba(16,185,129,.28)}
.bT{background:rgba(6,182,212,.12);color:var(--plasma);border:1px solid rgba(6,182,212,.28)}
.bL{background:rgba(244,63,94,.1);color:var(--hot);border:1px solid rgba(244,63,94,.22)}
.bBE{background:rgba(245,158,11,.12);color:var(--amber);border:1px solid rgba(245,158,11,.22)}

/* ── Info rows ── */
.irow{display:flex;justify-content:space-between;align-items:center;
  padding:5px 0;border-bottom:1px solid rgba(6,182,212,.05);transition:background .15s}
.irow:last-child{border-bottom:none}
.irow:hover{background:rgba(6,182,212,.03)}
.ik{color:var(--text2);font-family:'Rajdhani';text-transform:uppercase;letter-spacing:.1em;font-size:.58rem}
.iv{color:var(--text3);font-family:'Share Tech Mono',monospace;font-size:.7rem}

/* ── Progress bar ── */
.pbar{height:3px;background:var(--wire);border-radius:2px;overflow:hidden}
.pbar-f{height:100%;background:linear-gradient(90deg,var(--plasma),var(--plasma2));
  border-radius:2px;transition:width .4s;box-shadow:0 0 8px var(--glow)}

/* ── Skip rows ── */
.srow{display:flex;align-items:center;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(6,182,212,.04)}
.sbar-w{width:80px;height:3px;background:var(--wire);border-radius:2px;overflow:hidden}
.sbar{height:100%;border-radius:2px;transition:width .6s cubic-bezier(.34,1.56,.64,1)}

/* ── Pages ── */
#pg-login,#pg-gateway,#pg-main{position:fixed;inset:0;z-index:10;display:none}
#pg-login.on,#pg-gateway.on{display:flex;align-items:center;justify-content:center}
#pg-main.on{display:block}

/* ── Tab panes ── */
.tpane{display:none;position:absolute;inset:0}.tpane.on{display:flex}

/* ── Glow animations ── */
.glow-text{animation:gp 3s ease-in-out infinite}
@keyframes gp{0%,100%{text-shadow:0 0 8px var(--glow)}50%{text-shadow:0 0 22px var(--glow2),0 0 45px rgba(6,182,212,.3)}}
.pulse-border{animation:pb 2.5s ease-in-out infinite}
@keyframes pb{0%,100%{border-color:rgba(6,182,212,.15)}50%{border-color:rgba(6,182,212,.55)}}

/* ── PnL hero cards ── */
.phero{border-radius:3px;padding:13px 16px;position:relative;overflow:hidden;border:1px solid;
  transition:transform .2s,box-shadow .3s;cursor:default}
.phero:hover{transform:translateY(-3px);box-shadow:0 8px 30px rgba(0,0,0,.5)}
.phero::before{content:'';position:absolute;top:-50px;right:-50px;width:130px;height:130px;border-radius:50%;background:currentColor;opacity:.06;transition:opacity .3s}
.phero:hover::before{opacity:.1}
.phero>*{position:relative;z-index:1}
.phl{font-size:.55rem;text-transform:uppercase;letter-spacing:.15em;opacity:.65;margin-bottom:4px;font-family:'Rajdhani';font-weight:600}
.phv{font-family:'Orbitron',monospace;font-size:1.45rem;font-weight:700;line-height:1}
.phs{font-size:.58rem;opacity:.5;margin-top:3px;font-family:'Share Tech Mono'}

/* ── Globe tab full-screen container ── */
#globe-container{position:relative;width:100%;height:100%;display:flex}
#globe-three{flex:1;position:relative;cursor:grab;overflow:hidden}
#globe-three:active{cursor:grabbing}
#globe-three canvas{display:block;width:100%!important;height:100%!important}
#globe-sidebar{width:280px;flex-shrink:0;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px;border-left:1px solid var(--wire)}
#globe-hud{position:absolute;top:12px;left:12px;z-index:10;pointer-events:none}
#globe-hud-br{position:absolute;bottom:12px;right:12px;z-index:10;pointer-events:none}
.globe-stat-card{background:rgba(4,12,28,.78);backdrop-filter:blur(10px);border:1px solid var(--wire);
  border-radius:3px;padding:8px 12px;font-family:'Share Tech Mono';font-size:.62rem;color:var(--text3);
  min-width:180px}
.globe-stat-card .lbl{font-family:'Rajdhani';font-size:.52rem;color:var(--text2);text-transform:uppercase;letter-spacing:.12em;margin-bottom:3px}

/* ── Regime badge ── */
.regime-badge{font-family:'Orbitron';font-size:.62rem;font-weight:700;letter-spacing:.1em;
  padding:4px 12px;border-radius:2px;text-transform:uppercase;
  border:1px solid currentColor;display:inline-block;animation:rb 2s ease-in-out infinite}
@keyframes rb{0%,100%{box-shadow:none}50%{box-shadow:0 0 15px currentColor}}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:3px;height:3px}::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--wire2);border-radius:2px}::-webkit-scrollbar-thumb:hover{background:var(--wire3)}

/* ── Log lines ── */
.ll{font-family:'Share Tech Mono',monospace;font-size:.66rem;line-height:1.75;color:var(--text);padding:1px 0}
.ll .ts{color:#1a2f42}.ll .ok{color:var(--acid)}.ll .err{color:var(--hot)}.ll b,.ll strong{color:var(--plasma)}.ll .warn{color:var(--amber)}

/* ── Tick indicator ── */
@keyframes tick-ring{0%{transform:scale(1);opacity:.9}100%{transform:scale(2.5);opacity:0}}
.tick-ring{position:absolute;border-radius:50%;border:2px solid var(--plasma);animation:tick-ring 2s ease-out infinite;pointer-events:none}

/* ── Section header ── */
.sh{font-size:.58rem;text-transform:uppercase;letter-spacing:.2em;color:var(--text2);
  font-family:'Rajdhani';font-weight:700;padding-bottom:7px;
  border-bottom:1px solid var(--wire);margin-bottom:12px;display:flex;align-items:center;gap:8px}
.sh-dot{width:5px;height:5px;border-radius:50%;background:var(--plasma);box-shadow:0 0 8px var(--glow);flex-shrink:0;animation:dp 2s ease-in-out infinite}

/* ── Holographic flicker ── */
@keyframes holo{0%,98%{opacity:1}99%{opacity:.85}100%{opacity:1}}
.holo{animation:holo 4s infinite}

/* ── Pipeline HUD (globe) ── */
#pipeline-hud{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);z-index:11;
  display:flex;align-items:center;gap:6px;pointer-events:none;flex-wrap:wrap;justify-content:center}
.pstage{font-family:'Orbitron',monospace;font-size:.52rem;letter-spacing:.14em;color:var(--text);
  border:1px solid var(--wire);background:rgba(4,12,28,.82);backdrop-filter:blur(8px);
  padding:5px 10px;border-radius:2px;transition:all .3s;white-space:nowrap}
.pstage.active{color:var(--plasma2);border-color:var(--plasma);
  box-shadow:0 0 16px var(--glow),inset 0 0 10px rgba(6,182,212,.12);animation:pstP 1s ease-in-out infinite}
.pstage.done{color:var(--acid);border-color:rgba(16,185,129,.4)}
@keyframes pstP{0%,100%{box-shadow:0 0 10px var(--glow)}50%{box-shadow:0 0 26px var(--glow2)}}
.parrow{color:var(--wire3);font-size:.6rem}
#verdict-banner{position:absolute;top:70px;left:50%;transform:translateX(-50%) scale(.8);z-index:12;
  font-family:'Orbitron',monospace;font-size:2.1rem;font-weight:900;letter-spacing:.12em;text-align:center;
  border:1px solid var(--wire2);background:rgba(4,12,28,.88);backdrop-filter:blur(10px);
  padding:14px 34px;border-radius:3px;opacity:0;pointer-events:none;transition:opacity .35s,transform .35s}
#verdict-banner.show{opacity:1;transform:translateX(-50%) scale(1)}

/* ── HUD scanning line ── */
.scan-line{position:absolute;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,.5),transparent);
  animation:scan 4s linear infinite;pointer-events:none;z-index:5}
@keyframes scan{0%{top:0;opacity:.6}100%{top:100%;opacity:0}}
</style>
</head>
<body>

<canvas id="starfield"></canvas>
<canvas id="data-rain"></canvas>

<div id="ui-root">

<div id="pg-login" class="on">
  <div class="scan-line"></div>
  <div style="position:relative;z-index:2;width:440px">
    <div class="glass corners corners-b pulse-border" style="padding:38px">
      <div class="scan-line" style="opacity:.4"></div>
      <div style="text-align:center;margin-bottom:30px">
        <div style="display:flex;justify-content:center;margin-bottom:16px">
          <div style="width:52px;height:52px;border-radius:50%;border:1.5px solid var(--plasma);
            display:flex;align-items:center;justify-content:center;position:relative;
            box-shadow:0 0 20px var(--glow),inset 0 0 20px rgba(6,182,212,.1)">
            <div style="width:8px;height:8px;border-radius:50%;background:var(--plasma);box-shadow:0 0 12px var(--plasma);animation:dp 1.5s ease-in-out infinite"></div>
            <div class="tick-ring" style="width:36px;height:36px;top:7px;left:7px;animation-duration:2.5s"></div>
            <div class="tick-ring" style="width:50px;height:50px;top:0;left:0;animation-delay:1.2s;opacity:.4"></div>
          </div>
        </div>
        <div class="orb glow-text" style="font-size:.6rem;color:var(--plasma);letter-spacing:.35em;margin-bottom:8px">NEXUS // AI TERMINAL</div>
        <div class="orb" style="font-size:1.5rem;font-weight:900;color:var(--text5);letter-spacing:.06em;line-height:1.1">VOLATILITY 75<br><span style="color:var(--plasma);font-size:.9rem;letter-spacing:.15em">COMMAND CENTER</span></div>
        <div class="tech" style="font-size:.6rem;color:var(--text);margin-top:8px;letter-spacing:.22em">RESTRICTED ACCESS · v5.0</div>
      </div>
      <div style="height:1px;background:linear-gradient(90deg,transparent,var(--plasma),transparent);margin-bottom:24px;opacity:.4"></div>
      <div style="display:flex;flex-direction:column;gap:14px">
        <div><label>IDENTITY CODE</label><input type="text" id="web-user" placeholder="IDENT_###" autocomplete="off"></div>
        <div><label>AUTH KEY</label><input type="password" id="web-pass" placeholder="• • • • • • • • • •"></div>
      </div>
      <button class="btn btn-p" style="width:100%;margin-top:22px;padding:13px" onclick="webLogin()">
        ⟶ &nbsp; AUTHENTICATE
      </button>
      <div id="login-err" style="font-size:.65rem;text-align:center;color:var(--hot);min-height:14px;margin-top:10px;font-family:'Share Tech Mono'"></div>
    </div>
  </div>
</div>

<div id="pg-gateway">
  <div style="position:relative;z-index:2;width:500px">
    <div class="glass corners corners-b" style="padding:28px">
      <div class="scan-line" style="opacity:.3"></div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
        <div>
          <div class="tech" style="font-size:.56rem;color:var(--plasma);letter-spacing:.22em;margin-bottom:5px">// MT5 BROKER UPLINK</div>
          <div class="orb" style="font-size:.95rem;font-weight:700;color:var(--text5)" id="gw-title">ESTABLISH CONNECTION</div>
        </div>
        <div id="gw-sig" class="sig sig-idle" style="color:var(--hot)"><span class="dot dot-idle"></span><span id="gw-stxt">OFFLINE</span></div>
      </div>
      <div style="height:1px;background:linear-gradient(90deg,transparent,var(--plasma),transparent);opacity:.3;margin-bottom:20px"></div>
      <div style="display:flex;flex-direction:column;gap:11px">
        <div><label>SYMBOL</label><select id="symbol-select"></select></div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
          <div><label>ACCOUNT</label><input type="text" id="mt5-account" placeholder="12345678"></div>
          <div><label>SERVER</label><input type="text" id="mt5-server" value="Deriv-Server"></div>
        </div>
        <div><label>MASTER PASSWORD</label><input type="password" id="mt5-pass" placeholder="Auth key"></div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
          <div><label>DIRECTION</label>
            <select id="gw-dir"><option value="both">BOTH (auto)</option><option value="bullish">BULLISH</option><option value="bearish">BEARISH</option></select></div>
          <div><label>RR RATIO</label><input type="text" id="gw-rr" value="2.0"></div>
        </div>
      </div>
      <button class="btn btn-p" style="width:100%;margin-top:18px;padding:11px" id="btn-engage" onclick="toggleEngine()">
        ⟶ &nbsp; ENGAGE PIPELINE
      </button>
      <button class="btn btn-ghost" style="width:100%;margin-top:8px;padding:7px" onclick="webLogout()">SIGN OUT</button>
      <div id="gw-err" style="font-size:.65rem;text-align:center;color:var(--hot);min-height:13px;margin-top:8px;font-family:'Share Tech Mono'"></div>
    </div>
  </div>
</div>

<div id="pg-main">
<div style="position:fixed;inset:0;z-index:3;display:flex;flex-direction:column">

  <header class="glass" style="display:flex;align-items:center;justify-content:space-between;
    padding:8px 18px;border-radius:0;border-top:none;border-left:none;border-right:none;
    border-bottom:1px solid var(--wire);flex-shrink:0;position:relative;z-index:20">
    <div style="position:absolute;bottom:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--plasma),transparent);opacity:.5;pointer-events:none"></div>
    <div style="display:flex;align-items:center;gap:16px;min-width:300px">
      <div id="hdr-sig" class="sig" style="color:var(--hot)"><span class="dot dot-idle" id="hdr-dot"></span><span id="hdr-stxt">OFFLINE</span></div>
      <div style="display:flex;flex-direction:column">
        <div class="orb glow-text" style="font-size:.72rem;font-weight:900;color:var(--text5);letter-spacing:.12em">NEXUS <span style="color:var(--plasma)">AI</span></div>
        <div class="tech" style="font-size:.52rem;color:var(--text);letter-spacing:.18em">V75 COMMAND CENTER</div>
      </div>
      <div style="display:flex;flex-direction:column;padding:4px 10px;border-left:1px solid var(--wire);border-right:1px solid var(--wire)">
        <div class="tech" style="font-size:.5rem;color:var(--text);letter-spacing:.12em">LIVE PRICE</div>
        <div class="orb" style="font-size:.8rem;font-weight:700;color:var(--plasma)" id="hdr-price">—</div>
      </div>
      <div id="cb-tag" style="display:none;background:rgba(245,158,11,.1);border:1px solid var(--amber);
        border-radius:2px;padding:3px 10px;font-size:.58rem;font-family:'Rajdhani';font-weight:700;
        text-transform:uppercase;letter-spacing:.1em;color:var(--amber);animation:dp .9s ease-in-out infinite">
        ⚡ CIRCUIT BREAK <span id="cb-until" style="opacity:.7"></span>
      </div>
    </div>
    <nav style="display:flex;gap:0;border-bottom:none">
      <button class="nav-tab on" id="nav-overview"  onclick="showTab('overview')">Overview</button>
      <button class="nav-tab" id="nav-chart"     onclick="showTab('chart')">Chart</button>
      <button class="nav-tab" id="nav-globe"     onclick="showTab('globe')">🌐 Globe</button>
      <button class="nav-tab" id="nav-matrix"    onclick="showTab('matrix')">Matrix</button>
      <button class="nav-tab" id="nav-analytics" onclick="showTab('analytics')">Analytics</button>
      <button class="nav-tab" id="nav-signals"   onclick="showTab('signals')">Signals</button>
      <button class="nav-tab" id="nav-backtest"  onclick="showTab('backtest')">Backtest</button>
      <button class="nav-tab" id="nav-config"    onclick="showTab('config')">Config</button>
    </nav>
    <div style="display:flex;align-items:center;gap:18px;min-width:300px;justify-content:flex-end">
      <div style="text-align:right">
        <div class="tech" style="font-size:.5rem;color:var(--text);letter-spacing:.12em">EQUITY</div>
        <div class="orb" style="color:var(--acid);font-size:.88rem;font-weight:700" id="hdr-bal">$0.00</div>
      </div>
      <div style="text-align:right">
        <div class="tech" style="font-size:.5rem;color:var(--text);letter-spacing:.12em">TODAY PNL</div>
        <div class="tech" style="font-size:.82rem;font-weight:600" id="hdr-today">$0.00</div>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="position:relative;width:26px;height:26px;display:flex;align-items:center;justify-content:center">
          <div class="tick-ring" style="width:18px;height:18px;top:4px;left:4px"></div>
          <div style="width:6px;height:6px;border-radius:50%;background:var(--plasma);box-shadow:0 0 10px var(--glow);z-index:1"></div>
        </div>
        <div>
          <div class="tech" style="font-size:.5rem;color:var(--text);letter-spacing:.12em">TICK</div>
          <div class="tech" style="font-size:.68rem;color:var(--plasma)" id="hdr-tick">2.0s</div>
        </div>
      </div>
      <button class="btn btn-red btn-sm" onclick="toggleEngine()">■ STOP</button>
      <button onclick="webLogout()" style="font-size:.6rem;color:var(--text);background:none;border:none;cursor:pointer;font-family:'Rajdhani';letter-spacing:.08em;transition:color .2s" onmouseover="this.style.color='var(--plasma)'" onmouseout="this.style.color='var(--text)'">EXIT ⟶</button>
    </div>
  </header>

  <div style="flex:1;position:relative;overflow:hidden">

  <div id="tab-overview" class="tpane on" style="flex-direction:column;padding:12px;gap:10px;overflow-y:auto">

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:9px">
      <div class="phero corners" style="background:linear-gradient(135deg,rgba(16,185,129,.12),rgba(16,185,129,.03));border-color:rgba(16,185,129,.28);color:var(--acid)">
        <div class="phl">Today PnL</div><div class="phv" id="ov-today">$0.00</div><div class="phs">Live session</div></div>
      <div class="phero corners" style="background:linear-gradient(135deg,rgba(6,182,212,.12),rgba(6,182,212,.03));border-color:rgba(6,182,212,.28);color:var(--plasma)">
        <div class="phl">Week PnL</div><div class="phv" id="ov-week">$0.00</div><div class="phs">7-day window</div></div>
      <div class="phero corners" style="background:linear-gradient(135deg,rgba(139,92,246,.12),rgba(139,92,246,.03));border-color:rgba(139,92,246,.28);color:var(--violet)">
        <div class="phl">Month PnL</div><div class="phv" id="ov-month">$0.00</div><div class="phs">This month</div></div>
      <div class="phero corners" style="background:linear-gradient(135deg,rgba(245,158,11,.1),rgba(245,158,11,.03));border-color:rgba(245,158,11,.25);color:var(--amber)">
        <div class="phl">Net PnL (90d)</div><div class="phv" id="ov-pnl">$0.00</div><div class="phs">90-day window</div></div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px">
      <div class="mtile corners" style="--tc:var(--plasma)"><div class="lbl">Win Rate</div><div class="val" style="color:var(--plasma)" id="m-wr">0.0%</div><div class="sub">prev: 35.3%</div><div class="delta" style="color:var(--acid)" id="m-wr-d">—</div></div>
      <div class="mtile corners" style="--tc:var(--ion)"><div class="lbl">Profit Factor</div><div class="val" style="color:var(--ion)" id="m-pf">0.00</div><div class="sub">prev: &lt;1.0</div></div>
      <div class="mtile corners" style="--tc:var(--violet)"><div class="lbl">Sharpe Ratio</div><div class="val" style="color:var(--violet)" id="m-sh">0.00</div><div class="sub">HF target: 1–2</div></div>
      <div class="mtile corners" style="--tc:var(--hot)"><div class="lbl">Max Drawdown</div><div class="val" style="color:var(--hot)" id="m-dd">0.00%</div><div class="sub">prev: 97%</div></div>
      <div class="mtile corners" style="--tc:var(--amber)"><div class="lbl">Expectancy</div><div class="val" style="color:var(--amber)" id="m-exp">$0.00</div><div class="sub">per trade avg</div></div>
      <div class="mtile corners" style="--tc:var(--acid)"><div class="lbl">Min Balance</div><div class="val" style="color:var(--acid)" id="m-minbal">$0.00</div><div class="sub">worst reached</div></div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px">
      <div class="mtile" style="--tc:color:#ec4899"><div class="lbl">Total Trades</div><div class="val" style="color:#ec4899;font-size:1.4rem" id="m-total">0</div><div class="sub" id="m-wl">0W / 0L</div></div>
      <div class="mtile" style="--tc:color:#10b981"><div class="lbl">Wins</div><div class="val" style="color:#10b981;font-size:1.4rem" id="m-wins">0</div><div class="sub">closed profitable</div></div>
      <div class="mtile" style="--tc:color: #ff9900"><div class="lbl">Losses</div><div class="val" style="color: #ff9900;font-size:1.4rem" id="m-losses">0</div><div class="sub">closed at loss</div></div>
      <div class="mtile" style="--tc:color:#06b6d4"><div class="lbl">Gross Profit</div><div class="val" style="color:#06b6d4;font-size:1rem" id="m-gp">$0.00</div><div class="sub">sum winning trades</div></div>
      <div class="mtile" style="--tc:color:#8b5cf6"><div class="lbl">Gross Loss</div><div class="val" style="color:#8b5cf6;font-size:1rem" id="m-gl">$0.00</div><div class="sub">sum losing trades</div></div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px">
      <div class="mtile" style="--tc:var(--acid)"><div class="lbl">Avg Win</div><div class="val" style="color:var(--acid);font-size:1rem" id="m-avgwin">$0.00</div><div class="sub">per winning trade</div></div>
      <div class="mtile" style="--tc:var(--hot)"><div class="lbl">Avg Loss</div><div class="val" style="color:var(--hot);font-size:1rem" id="m-avgloss">$0.00</div><div class="sub">per losing trade</div></div>
      <div class="mtile" style="--tc:var(--amber)"><div class="lbl">Avg RR Realized</div><div class="val" style="color:var(--amber);font-size:1rem" id="m-avgrr">0.00</div><div class="sub">win/loss ratio</div></div>
      <div class="mtile" style="--tc:var(--plasma)"><div class="lbl">Today Trades</div><div class="val" style="color:var(--plasma);font-size:1.4rem" id="m-tdtrades">0</div><div class="sub">session count</div></div>
      <div class="mtile" style="--tc:var(--violet)"><div class="lbl">Win Streak</div><div class="val" style="color:var(--violet);font-size:1.2rem" id="m-streak">0</div><div class="sub" id="m-strktype">—</div></div>
      <div class="mtile" style="--tc:var(--acid)"><div class="lbl">Peak Equity</div><div class="val" style="color:var(--acid);font-size:1rem" id="m-peak">$0.00</div><div class="sub">session high</div></div>
    </div>

    <div style="display:grid;grid-template-columns:2fr 1fr;gap:10px;flex:1;min-height:0">
      <div class="glass corners corners-b" style="padding:14px;display:flex;flex-direction:column;min-height:180px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <div><div class="orb" style="font-size:.62rem;color:var(--text4);letter-spacing:.1em">EQUITY CURVE</div>
            <div class="tech" style="font-size:.56rem;color:var(--text);margin-top:1px">Portfolio value · 90-day compounding</div></div>
          <div class="orb" id="ov-eq-pct" style="font-size:.72rem;font-weight:600;color:var(--acid)">+0.00%</div>
        </div>
        <div style="flex:1;position:relative;min-height:110px"><canvas id="c-equity"></canvas></div>
      </div>
      <div class="glass corners corners-b" style="padding:14px;display:flex;flex-direction:column;min-height:180px">
        <div class="orb" style="font-size:.62rem;color:var(--text4);letter-spacing:.1em;margin-bottom:3px">OUTCOME MATRIX</div>
        <div class="tech" style="font-size:.56rem;color:var(--text);margin-bottom:12px">Trade result breakdown</div>
        <div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:10px">
          <div style="width:130px;height:130px;position:relative"><canvas id="c-donut"></canvas></div>
          <div id="donut-leg" style="width:100%;display:flex;flex-direction:column;gap:4px"></div>
        </div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1.4fr 1fr;gap:10px">
      <div class="glass" style="padding:13px">
        <div class="orb" style="font-size:.62rem;color:var(--text4);letter-spacing:.1em;margin-bottom:9px">MONTHLY BREAKDOWN</div>
        <div style="position:relative;height:100px"><canvas id="c-monthly"></canvas></div>
      </div>
      <div class="glass" style="display:flex;flex-direction:column;height:140px">
        <div style="padding:6px 11px;border-bottom:1px solid var(--wire);font-size:.55rem;font-family:'Rajdhani';font-weight:700;text-transform:uppercase;letter-spacing:.15em;color:var(--text)">▶ EXECUTION FEED</div>
        <div id="feed-ov" style="flex:1;overflow-y:auto;padding:5px 11px;display:flex;flex-direction:column;gap:1px"></div>
      </div>
    </div>
  </div>

  <div id="tab-chart" class="tpane" style="flex-direction:column;padding:12px;gap:9px">
    <div class="glass corners corners-b" style="flex:1;display:flex;flex-direction:column;overflow:hidden">
      <div style="display:flex;align-items:center;justify-content:space-between;padding:7px 12px;border-bottom:1px solid var(--wire);flex-shrink:0">
        <div style="display:flex;gap:5px">
          <button class="ctab on" onclick="setTF('M5',this)">M5</button>
          <button class="ctab" onclick="setTF('M15',this)">M15</button>
          <button class="ctab" onclick="setTF('H4',this)">H4</button>
        </div>
        <div style="display:flex;align-items:center;gap:12px">
          <div id="cdebug" class="tech" style="font-size:.6rem;color:var(--text)">AWAITING DATA…</div>
          <div style="display:flex;align-items:center;gap:6px">
            <div style="width:5px;height:5px;border-radius:50%;background:var(--plasma);box-shadow:0 0 8px var(--glow);animation:dp 2s ease-in-out infinite"></div>
            <div class="tech" style="font-size:.58rem;color:var(--plasma)">V75 LIVE</div>
          </div>
        </div>
      </div>
      <div id="tvchart" style="flex:1;width:100%"></div>
    </div>
    <div class="glass" style="height:125px;display:flex;flex-direction:column;flex-shrink:0">
      <div style="padding:5px 11px;border-bottom:1px solid var(--wire);font-size:.55rem;font-family:'Rajdhani';font-weight:700;text-transform:uppercase;letter-spacing:.15em;color:var(--text)">▶ EXECUTION FEED</div>
      <div id="feed-ch" style="flex:1;overflow-y:auto;padding:5px 11px"></div>
    </div>
  </div>

  <div id="tab-globe" class="tpane" style="flex-direction:row">
    <div id="globe-container">
      <div id="globe-three">
        <div class="scan-line"></div>
        <div id="globe-hud">
          <div class="globe-stat-card" style="margin-bottom:8px">
            <div class="lbl">VOLATILITY REGIME</div>
            <div class="orb" style="font-size:.9rem;font-weight:700;color:var(--plasma)" id="gh-regime">—</div>
          </div>
          <div class="globe-stat-card">
            <div class="lbl">AI CONFIDENCE</div>
            <div style="display:flex;align-items:center;gap:8px;margin-top:4px">
              <div style="flex:1;height:3px;background:var(--wire);border-radius:2px;overflow:hidden">
                <div id="gh-conf-bar" style="height:100%;background:linear-gradient(90deg,var(--ion),var(--plasma));width:0%;transition:width .8s;box-shadow:0 0 8px var(--glow)"></div>
              </div>
              <div class="tech" style="font-size:.7rem;color:var(--plasma)" id="gh-conf">0%</div>
            </div>
          </div>
          <button class="btn btn-p btn-sm" style="pointer-events:auto;margin-top:8px;width:100%;justify-content:center" onclick="replayPipeline()">⟳ RUN ANALYSIS PIPELINE</button>
          <div class="tech" style="font-size:.5rem;color:var(--text);margin-top:4px;letter-spacing:.1em">CLICK THE ORB TO DISTURB IT</div>
        </div>
        <div id="pipeline-hud">
          <div class="pstage" id="pst-uplink">◈ UPLINK</div><div class="parrow">▸</div>
          <div class="pstage" id="pst-parse">◈ STRUCT SCAN</div><div class="parrow">▸</div>
          <div class="pstage" id="pst-merge">◈ TF FUSION</div><div class="parrow">▸</div>
          <div class="pstage" id="pst-sanitize">◈ SHIELD</div><div class="parrow">▸</div>
          <div class="pstage" id="pst-infer">◈ INFERENCE</div><div class="parrow">▸</div>
          <div class="pstage" id="pst-verdict">◈ VERDICT</div>
        </div>
        <div id="verdict-banner"></div>
        <div id="globe-hud-br">
          <div class="globe-stat-card">
            <div class="lbl" style="margin-bottom:6px">DRAG TO ORBIT · SCROLL TO ZOOM</div>
            <div style="display:flex;gap:10px;font-family:'Share Tech Mono';font-size:.58rem;color:var(--text2)">
              <span><span style="color:#10b981">●</span> Bullish</span>
              <span><span style="color:#f43f5e">●</span> Bearish</span>
              <span><span style="color:#f59e0b">●</span> Neutral</span>
            </div>
          </div>
        </div>
      </div>
      <div id="globe-sidebar">
        <div class="glass corners corners-b" style="padding:14px">
          <div class="sh"><div class="sh-dot"></div>REGIME ANALYSIS</div>
          <div style="display:flex;flex-direction:column;gap:6px">
            <div class="irow"><span class="ik">REGIME</span><span class="orb" style="font-size:.68rem;color:var(--plasma)" id="glob-regime">—</span></div>
            <div class="irow"><span class="ik">TICK SPEED</span><span class="iv">0.5 Hz · 2s</span></div>
            <div class="irow"><span class="ik">AMPLITUDE</span><span class="iv" id="glob-amp">—</span></div>
            <div class="irow"><span class="ik">MOMENTUM</span><span class="iv" id="glob-mom">—</span></div>
            <div class="irow"><span class="ik">AI BIAS</span><span class="iv" id="glob-bias">—</span></div>
            <div class="irow"><span class="ik">PULLBACK</span><span class="iv" id="glob-pb">—</span></div>
            <div class="irow"><span class="ik">CIRCUIT</span><span class="iv" id="glob-cb" style="color:var(--acid)">NOMINAL</span></div>
          </div>
        </div>
        <div class="glass" style="padding:14px">
          <div class="sh"><div class="sh-dot"></div>AI CLUSTER METRICS</div>
          <div style="display:flex;flex-direction:column;gap:8px">
            <div>
              <div style="font-size:.56rem;color:var(--text2);font-family:'Rajdhani';font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">MTF MOMENTUM</div>
              <div style="height:4px;background:var(--wire);border-radius:2px;overflow:hidden">
                <div id="cl-mtf" style="height:100%;background:var(--plasma);border-radius:2px;width:72%;box-shadow:0 0 8px var(--glow);transition:width .8s"></div>
              </div>
            </div>
            <div>
              <div style="font-size:.56rem;color:var(--text2);font-family:'Rajdhani';font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">STRUCTURE STRENGTH</div>
              <div style="height:4px;background:var(--wire);border-radius:2px;overflow:hidden">
                <div id="cl-str" style="height:100%;background:var(--acid);border-radius:2px;width:58%;box-shadow:0 0 6px rgba(16,185,129,.5);transition:width .8s"></div>
              </div>
            </div>
            <div>
              <div style="font-size:.56rem;color:var(--text2);font-family:'Rajdhani';font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">PULLBACK QUALITY</div>
              <div style="height:4px;background:var(--wire);border-radius:2px;overflow:hidden">
                <div id="cl-pb" style="height:100%;background:var(--amber);border-radius:2px;width:45%;box-shadow:0 0 6px rgba(245,158,11,.5);transition:width .8s"></div>
              </div>
            </div>
            <div>
              <div style="font-size:.56rem;color:var(--text2);font-family:'Rajdhani';font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">MODEL CONFIDENCE</div>
              <div style="height:4px;background:var(--wire);border-radius:2px;overflow:hidden">
                <div id="cl-conf" style="height:100%;background:var(--violet);border-radius:2px;width:0%;transition:width .8s;box-shadow:0 0 6px rgba(139,92,246,.5)"></div>
              </div>
            </div>
          </div>
        </div>
        <div class="glass" style="padding:14px">
          <div class="sh"><div class="sh-dot"></div>GLOBE LEGEND</div>
          <div style="display:flex;flex-direction:column;gap:7px">
            <div style="display:flex;align-items:center;gap:8px"><div style="width:10px;height:10px;border-radius:50%;background:#10b981;box-shadow:0 0 6px #10b981"></div><span class="tech" style="font-size:.6rem;color:var(--text3)">Bullish regime nodes</span></div>
            <div style="display:flex;align-items:center;gap:8px"><div style="width:10px;height:10px;border-radius:50%;background:#f43f5e;box-shadow:0 0 6px #f43f5e"></div><span class="tech" style="font-size:.6rem;color:var(--text3)">Bearish regime nodes</span></div>
            <div style="display:flex;align-items:center;gap:8px"><div style="width:10px;height:10px;border-radius:50%;background:#f59e0b;box-shadow:0 0 6px #f59e0b"></div><span class="tech" style="font-size:.6rem;color:var(--text3)">Neutral / transition</span></div>
            <div style="display:flex;align-items:center;gap:8px"><div style="width:12px;height:2px;background:var(--plasma);box-shadow:0 0 5px var(--glow)"></div><span class="tech" style="font-size:.6rem;color:var(--text3)">Mean reversion ring</span></div>
            <div style="display:flex;align-items:center;gap:8px"><div style="width:12px;height:2px;background:#22d3ee;box-shadow:0 0 5px #22d3ee;opacity:.6"></div><span class="tech" style="font-size:.6rem;color:var(--text3)">Neural arc connections</span></div>
          </div>
        </div>
        <div class="glass" style="display:flex;flex-direction:column;flex:1;min-height:120px">
          <div style="padding:6px 11px;border-bottom:1px solid var(--wire);font-size:.55rem;font-family:'Rajdhani';font-weight:700;text-transform:uppercase;letter-spacing:.15em;color:var(--text)">▶ SIGNAL FEED</div>
          <div id="feed-glob" style="flex:1;overflow-y:auto;padding:5px 11px"></div>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-analytics" class="tpane" style="flex-direction:column;padding:12px;gap:10px;overflow-y:auto">

    <div class="glass" style="padding:10px 16px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0">
      <div style="display:flex;align-items:center;gap:22px">
        <div style="display:flex;flex-direction:column">
          <div class="tech" style="font-size:.48rem;color:var(--text);letter-spacing:.18em">NEXUS ANALYTICS ENGINE</div>
          <div class="orb" style="font-size:.72rem;font-weight:700;color:var(--text5)">REAL-TIME <span style="color:var(--plasma)">PERFORMANCE MATRIX</span></div>
        </div>
        <div style="width:1px;height:36px;background:var(--wire)"></div>
        <div style="display:flex;gap:20px">
          <div><div class="tech" style="font-size:.5rem;color:var(--text);letter-spacing:.12em">AVG WIN</div><div class="orb" style="font-size:.78rem;color:var(--acid)" id="an-avgwin">$0.00</div></div>
          <div><div class="tech" style="font-size:.5rem;color:var(--text);letter-spacing:.12em">AVG LOSS</div><div class="orb" style="font-size:.78rem;color:var(--hot)" id="an-avgloss">$0.00</div></div>
          <div><div class="tech" style="font-size:.5rem;color:var(--text);letter-spacing:.12em">REALIZED RR</div><div class="orb" style="font-size:.78rem;color:var(--amber)" id="an-rr">0.00x</div></div>
          <div><div class="tech" style="font-size:.5rem;color:var(--text);letter-spacing:.12em">TODAY</div><div class="orb" style="font-size:.78rem;color:var(--plasma)" id="an-today">+$0.00</div></div>
          <div><div class="tech" style="font-size:.5rem;color:var(--text);letter-spacing:.12em">SESSION TRADES</div><div class="orb" style="font-size:.78rem;color:var(--violet)" id="an-tdtrades">0</div></div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <button class="btn btn-ghost btn-sm" onclick="fetchAndRenderAnalytics()">↻ REFRESH</button>
        <button class="btn btn-ghost btn-sm" onclick="exportTradeLogs()">⬇ EXPORT</button>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;flex-shrink:0">
      <div class="glass corners" style="padding:14px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <div><div class="orb" style="font-size:.6rem;color:var(--text4);letter-spacing:.1em">HOURLY PnL DISTRIBUTION</div>
          <div class="tech" style="font-size:.55rem;color:var(--text);margin-top:2px">Profit by hour of day (blocked hours in red)</div></div>
          <div class="tech" style="font-size:.55rem;color:var(--plasma)" id="an-best-hour">—</div>
        </div>
        <div style="position:relative;height:160px"><canvas id="an-hourly"></canvas></div>
      </div>
      <div class="glass corners" style="padding:14px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <div><div class="orb" style="font-size:.6rem;color:var(--text4);letter-spacing:.1em">LIVE DRAWDOWN CURVE</div>
          <div class="tech" style="font-size:.55rem;color:var(--text);margin-top:2px">Running drawdown from peak equity</div></div>
          <div class="orb" style="font-size:.6rem;color:var(--hot)" id="an-maxdd">0.00%</div>
        </div>
        <div style="position:relative;height:160px"><canvas id="an-dd"></canvas></div>
      </div>
      <div class="glass corners" style="padding:14px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <div><div class="orb" style="font-size:.6rem;color:var(--text4);letter-spacing:.1em">CONFIDENCE → WIN RATE</div>
          <div class="tech" style="font-size:.55rem;color:var(--text);margin-top:2px">WR by probability threshold bucket</div></div>
        </div>
        <div style="position:relative;height:160px"><canvas id="an-prob"></canvas></div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:2fr 1fr;gap:10px;flex:1;min-height:0">
      <div class="glass corners corners-b" style="padding:14px;display:flex;flex-direction:column">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <div><div class="orb" style="font-size:.6rem;color:var(--text4);letter-spacing:.1em">LIVE EQUITY CURVE</div>
          <div class="tech" style="font-size:.55rem;color:var(--text);margin-top:2px">Balance vs time · 90-day window</div></div>
          <div style="display:flex;gap:6px">
            <div style="font-size:.56rem;font-family:'Rajdhani';font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--text2)">Peak: <span style="color:var(--acid)" id="an-peak">$0.00</span></div>
          </div>
        </div>
        <div style="flex:1;position:relative;min-height:120px"><canvas id="an-equity"></canvas></div>
      </div>
      <div class="glass corners corners-b" style="padding:14px;display:flex;flex-direction:column">
        <div class="sh"><div class="sh-dot"></div>LIVE TRADE JOURNAL <span id="an-tradecount" style="color:var(--plasma);font-family:'Share Tech Mono';font-size:.6rem"></span></div>
        <div style="flex:1;overflow-y:auto">
          <table class="tbl" style="font-size:.6rem">
            <thead><tr><th>TIME</th><th>OUT</th><th>PnL</th><th>BAL</th></tr></thead>
            <tbody id="an-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:9px;flex-shrink:0">
      <div class="glass" style="padding:13px">
        <div class="sh"><div class="sh-dot"></div>WIN STREAK TRACKER</div>
        <div style="display:flex;flex-direction:column;gap:6px">
          <div class="irow"><span class="ik">CURRENT STREAK</span><span class="orb" style="font-size:.88rem;color:var(--plasma)" id="an-streak">0</span></div>
          <div class="irow"><span class="ik">TYPE</span><span class="iv" id="an-strktype">—</span></div>
          <div style="margin-top:6px;display:flex;gap:2px" id="an-streak-dots"></div>
        </div>
      </div>
      <div class="glass" style="padding:13px">
        <div class="sh"><div class="sh-dot"></div>RISK EFFICIENCY</div>
        <div style="display:flex;flex-direction:column;gap:7px">
          <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
              <div class="tech" style="font-size:.54rem;color:var(--text2);letter-spacing:.1em">PROFIT FACTOR</div>
              <div class="tech" style="font-size:.62rem;color:var(--plasma)" id="an-pf">0.00</div>
            </div>
            <div style="height:3px;background:var(--wire);border-radius:2px;overflow:hidden"><div id="an-pf-bar" style="height:100%;background:var(--plasma);border-radius:2px;transition:width .6s;width:0%"></div></div>
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
              <div class="tech" style="font-size:.54rem;color:var(--text2);letter-spacing:.1em">SHARPE RATIO</div>
              <div class="tech" style="font-size:.62rem;color:var(--violet)" id="an-sh">0.00</div>
            </div>
            <div style="height:3px;background:var(--wire);border-radius:2px;overflow:hidden"><div id="an-sh-bar" style="height:100%;background:var(--violet);border-radius:2px;transition:width .6s;width:0%"></div></div>
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
              <div class="tech" style="font-size:.54rem;color:var(--text2);letter-spacing:.1em">WIN RATE</div>
              <div class="tech" style="font-size:.62rem;color:var(--acid)" id="an-wr">0.00%</div>
            </div>
            <div style="height:3px;background:var(--wire);border-radius:2px;overflow:hidden"><div id="an-wr-bar" style="height:100%;background:var(--acid);border-radius:2px;transition:width .6s;width:0%"></div></div>
          </div>
        </div>
      </div>
      <div class="glass" style="padding:13px">
        <div class="sh"><div class="sh-dot"></div>PERIOD BREAKDOWN</div>
        <div style="display:flex;flex-direction:column;gap:6px">
          <div class="irow"><span class="ik">TODAY</span><span class="orb" style="font-size:.7rem" id="an-per-today">$0.00</span></div>
          <div class="irow"><span class="ik">THIS WEEK</span><span class="orb" style="font-size:.7rem" id="an-per-week">$0.00</span></div>
          <div class="irow"><span class="ik">THIS MONTH</span><span class="orb" style="font-size:.7rem" id="an-per-month">$0.00</span></div>
          <div class="irow"><span class="ik">90-DAY NET</span><span class="orb" style="font-size:.7rem" id="an-per-90">$0.00</span></div>
        </div>
      </div>
      <div class="glass" style="padding:13px">
        <div class="sh"><div class="sh-dot"></div>EXECUTION QUALITY</div>
        <div style="display:flex;flex-direction:column;gap:6px">
          <div class="irow"><span class="ik">EXPECTANCY</span><span class="orb" style="font-size:.7rem;color:var(--amber)" id="an-exp">$0.00</span></div>
          <div class="irow"><span class="ik">MAX DRAWDOWN</span><span class="iv" style="color:var(--hot)" id="an-dd2">0.00%</span></div>
          <div class="irow"><span class="ik">MIN BALANCE</span><span class="iv" id="an-minbal">$0.00</span></div>
          <div class="irow"><span class="ik">RECOVERY FACTOR</span><span class="iv" style="color:var(--plasma)" id="an-recovery">—</span></div>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-matrix" class="tpane" style="flex-direction:row;padding:12px;gap:10px;overflow-y:auto">
    <div style="flex:1;display:flex;flex-direction:column;gap:10px">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
        <div class="glass" style="padding:13px">
          <div class="orb" style="font-size:.6rem;color:var(--text4);letter-spacing:.08em;margin-bottom:9px">DRAWDOWN</div>
          <div style="position:relative;height:130px"><canvas id="mx-dd"></canvas></div>
        </div>
        <div class="glass" style="padding:13px">
          <div class="orb" style="font-size:.6rem;color:var(--text4);letter-spacing:.08em;margin-bottom:9px">HOURLY PnL</div>
          <div style="position:relative;height:130px"><canvas id="mx-hourly"></canvas></div>
        </div>
        <div class="glass" style="padding:13px">
          <div class="orb" style="font-size:.6rem;color:var(--text4);letter-spacing:.08em;margin-bottom:9px">CONFIDENCE → WR</div>
          <div style="position:relative;height:130px"><canvas id="mx-prob"></canvas></div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;flex:1">
        <div class="glass" style="padding:13px">
          <div class="orb" style="font-size:.6rem;color:var(--text4);letter-spacing:.08em;margin-bottom:9px">RECENT TRADES</div>
          <div style="overflow-y:auto;max-height:200px">
            <table class="tbl"><thead><tr><th>TIME</th><th>OUT</th><th>PROB</th><th>PnL</th><th>BAL</th></tr></thead><tbody id="mx-tbody"></tbody></table>
          </div>
        </div>
        <div class="glass" style="padding:13px;display:flex;flex-direction:column">
          <div class="orb" style="font-size:.6rem;color:var(--text4);letter-spacing:.08em;margin-bottom:9px">AI SIGNAL FEED</div>
          <div id="feed-mx" style="flex:1;overflow-y:auto"></div>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-signals" class="tpane" style="flex-direction:row;padding:12px;gap:10px;overflow-y:auto">
    <div style="display:flex;flex-direction:column;gap:10px;min-width:280px;max-width:300px">
      <div class="glass corners corners-b" style="padding:16px">
        <div class="sh"><div class="sh-dot"></div>LATEST SIGNAL</div>
        <div style="display:flex;flex-direction:column;gap:7px">
          <div class="irow"><span class="ik">TIME</span><span class="iv" id="sig-time">—</span></div>
          <div class="irow"><span class="ik">DIRECTION</span><span class="orb" style="font-size:.7rem" id="sig-dir">—</span></div>
          <div class="irow"><span class="ik">BIAS</span><span class="iv" id="sig-bias">—</span></div>
          <div class="irow"><span class="ik">ENTRY MODE</span><span class="iv" style="color:var(--plasma)" id="sig-mode">—</span></div>
          <div class="irow"><span class="ik">LIMIT PRICE</span><span class="iv" id="sig-lmt">—</span></div>
          <div class="irow"><span class="ik">ACTION</span><span class="iv" id="sig-act">—</span></div>
        </div>
        <div style="margin-top:12px">
          <div class="tech" style="font-size:.55rem;color:var(--text);margin-bottom:5px;letter-spacing:.12em">AI CONFIDENCE</div>
          <div style="display:flex;align-items:center;gap:10px">
            <div style="flex:1;height:6px;background:var(--wire);border-radius:3px;overflow:hidden">
              <div id="pb-prob" style="height:100%;background:linear-gradient(90deg,var(--ion),var(--plasma));width:0%;transition:width .6s;box-shadow:0 0 10px var(--glow)"></div>
            </div>
            <div class="orb" style="font-size:.92rem;font-weight:700;color:var(--plasma);min-width:46px;text-align:right" id="sig-prob">0%</div>
          </div>
        </div>
        <div style="margin-top:10px">
          <div class="tech" style="font-size:.55rem;color:var(--text);margin-bottom:5px;letter-spacing:.12em">PULLBACK DEPTH</div>
          <div style="display:flex;align-items:center;gap:10px">
            <div style="flex:1;height:6px;background:var(--wire);border-radius:3px;overflow:hidden">
              <div id="pb-depth" style="height:100%;background:linear-gradient(90deg,var(--amber),var(--hot));width:0%;transition:width .6s"></div>
            </div>
            <div class="orb" style="font-size:.92rem;font-weight:700;color:var(--amber);min-width:46px;text-align:right" id="sig-pb">0%</div>
          </div>
        </div>
      </div>
      <div class="glass" style="padding:16px">
        <div class="sh"><div class="sh-dot"></div>CIRCUIT BREAKER</div>
        <div style="display:flex;flex-direction:column;gap:6px">
          <div class="irow"><span class="ik">STATUS</span><span class="orb" style="font-size:.68rem" id="cb-status">NOMINAL</span></div>
          <div class="irow"><span class="ik">PAUSED UNTIL</span><span class="iv" id="cb-until2">—</span></div>
          <div class="irow"><span class="ik">ROLLING WR (15)</span><span class="orb" style="font-size:.68rem" id="roll-wr">—</span></div>
          <div style="height:5px;background:var(--wire);border-radius:2px;margin-top:4px;overflow:hidden">
            <div id="roll-bar" style="height:100%;background:var(--acid);border-radius:2px;width:0%;transition:width .6s;box-shadow:0 0 6px rgba(16,185,129,.5)"></div>
          </div>
        </div>
      </div>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;gap:10px">
      <div class="glass corners corners-b" style="padding:16px;flex:1">
        <div class="sh"><div class="sh-dot"></div>FILTER DIAGNOSTICS <span style="color:var(--text);font-weight:400;font-size:.52rem;letter-spacing:.08em">— MIRRORS BACKTESTER COUNTERS</span></div>
        <div id="skip-cont" style="display:flex;flex-direction:column;gap:3px"></div>
      </div>
      <div class="glass" style="height:170px;display:flex;flex-direction:column">
        <div style="padding:5px 11px;border-bottom:1px solid var(--wire);font-size:.55rem;font-family:'Rajdhani';font-weight:700;text-transform:uppercase;letter-spacing:.15em;color:var(--text)">▶ SIGNAL FEED</div>
        <div id="feed-sig" style="flex:1;overflow-y:auto;padding:5px 11px"></div>
      </div>
    </div>
  </div>

  <div id="tab-backtest" class="tpane" style="flex-direction:column;overflow:hidden">
    <div style="padding:7px 14px;border-bottom:1px solid var(--wire);background:rgba(4,13,26,.9);display:flex;align-items:center;gap:8px;flex-shrink:0">
      <button class="sub-tab on" id="bnt-setup"  onclick="btTab('setup')">Setup</button>
      <button class="sub-tab"    id="bnt-run"    onclick="btTab('run')">Run Log</button>
      <button class="sub-tab"    id="bnt-report" onclick="btTab('report')">Report</button>
      <div style="flex:1"></div>
      <div id="bt-pill" style="display:none;font-size:.58rem;font-family:'Rajdhani';font-weight:700;text-transform:uppercase;letter-spacing:.1em;padding:3px 12px;border-radius:2px;border:1px solid var(--plasma);color:var(--plasma);animation:dp .9s ease-in-out infinite">▶ RUNNING</div>
    </div>
    <div id="bt-setup" style="flex:1;overflow-y:auto;padding:13px;display:flex;gap:12px">
      <div style="display:flex;flex-direction:column;gap:10px;min-width:290px;max-width:310px">
        <div class="glass" style="padding:15px">
          <div class="sh"><div class="sh-dot"></div>DATA &amp; MODEL</div>
          <div style="display:flex;flex-direction:column;gap:9px">
            <div><label>SCRIPT</label><input type="text" id="bt-script" value="ai_backtester_be2.py"></div>
            <div><label>FEATURE CSV</label><input type="text" id="bt-csv" value="ml_dataset_labeled_v15.csv"></div>
            <div><label>OHLC CSV</label><input type="text" id="bt-ohlc" placeholder="./historical_data/V75_M5.csv"></div>
            <div><label>MODEL FILE</label><input type="text" id="bt-model" placeholder="model_1R_20pb_v15.json"></div>
            <div><label>METADATA</label><input type="text" id="bt-meta" placeholder="model_1R_20pb_v15_metadata.json"></div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
              <div><label>START DATE</label><input type="text" id="bt-start" value="2025-05-01"></div>
              <div><label>END DATE</label><input type="text" id="bt-end" value="2026-06-01"></div>
            </div>
          </div>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:10px;flex:1">
        <div class="glass" style="padding:15px">
          <div class="sh"><div class="sh-dot"></div>SIGNAL FILTERS</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
            <div><label>DIRECTION</label><select id="bt-dir"><option value="both">BOTH</option><option value="bullish">BULLISH</option><option value="bearish">BEARISH</option></select></div>
            <div><label>RR RATIO</label><input type="text" id="bt-rr" value="2.0"></div>
            <div><label>BE TRIGGER</label><input type="text" id="bt-be" value="1.5"></div>
            <div><label>BE TRAIL %</label><input type="text" id="bt-trail" value="0.65"></div>
            <div><label>THRESH MIN</label><input type="text" id="bt-thresh" value="0.50"></div>
            <div><label>THRESH MAX</label><input type="text" id="bt-tmax" value="0.65"></div>
            <div><label>BLOCKED HOURS</label><input type="text" id="bt-blk" value="3,4,5,7,9,10,14,19,20,22"></div>
            <div><label>BALANCE</label><input type="text" id="bt-bal" value="100.0"></div>
            <div><label>RISK %</label><input type="text" id="bt-risk" value="1.0"></div>
            <div><label>SLIPPAGE TICKS</label><input type="text" id="bt-slip" value="0"></div>
          </div>
          <div style="margin-top:10px;display:flex;align-items:center;gap:8px">
            <input type="checkbox" id="bt-ar"><label for="bt-ar" style="display:inline;margin:0;cursor:pointer">ANTI-REPAINT</label>
          </div>
        </div>
        <div style="display:flex;gap:10px">
          <button class="btn btn-p" id="btn-bt" style="flex:1;padding:11px" onclick="runBacktest()">⟶ LAUNCH BACKTEST</button>
          <div id="bt-err" style="flex:1;font-size:.65rem;color:var(--hot);display:flex;align-items:center;font-family:'Share Tech Mono'"></div>
        </div>
      </div>
    </div>
    <div id="bt-run" style="flex:1;overflow-y:auto;padding:13px;display:none;flex-direction:column;gap:10px">
      <div class="glass" style="padding:10px">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
          <div class="tech" style="font-size:.6rem;color:var(--text)" id="bt-stxt">WAITING…</div>
          <div class="tech" style="font-size:.62rem;color:var(--plasma)" id="bt-pct">0%</div>
        </div>
        <div class="pbar"><div class="pbar-f" id="bt-pbar" style="width:0%"></div></div>
      </div>
      <div class="glass" style="flex:1;padding:10px;font-family:'Share Tech Mono';font-size:.65rem;color:var(--text3);overflow-y:auto" id="bt-log"></div>
    </div>
    <div id="bt-report" style="flex:1;overflow-y:auto;padding:13px;display:none;flex-direction:column;gap:10px">
      <div id="bt-no-data" style="flex:1;display:flex;align-items:center;justify-content:center;color:var(--text2);font-family:'Rajdhani';font-size:.8rem;letter-spacing:.15em">NO BACKTEST DATA YET — RUN A BACKTEST FIRST</div>
      <div id="bt-results" style="display:none;flex-direction:column;gap:10px">
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:9px" id="bt-heroes"></div>
        <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px" id="bt-tiles"></div>
        <div style="display:grid;grid-template-columns:2fr 1fr;gap:10px">
          <div class="glass" style="padding:13px"><div class="orb" style="font-size:.6rem;color:var(--text4);margin-bottom:9px">EQUITY CURVE</div><div style="position:relative;height:180px"><canvas id="br-equity"></canvas></div></div>
          <div class="glass" style="padding:13px"><div class="orb" style="font-size:.6rem;color:var(--text4);margin-bottom:9px">OUTCOME MATRIX</div>
            <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
              <div style="width:120px;height:120px;position:relative"><canvas id="br-donut"></canvas></div>
              <div id="br-donut-leg" style="width:100%"></div>
            </div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
          <div class="glass" style="padding:13px"><div class="orb" style="font-size:.6rem;color:var(--text4);margin-bottom:9px">DRAWDOWN</div><div style="position:relative;height:130px"><canvas id="br-dd"></canvas></div></div>
          <div class="glass" style="padding:13px"><div class="orb" style="font-size:.6rem;color:var(--text4);margin-bottom:9px">HOURLY PnL</div><div style="position:relative;height:130px"><canvas id="br-hourly"></canvas></div></div>
          <div class="glass" style="padding:13px"><div class="orb" style="font-size:.6rem;color:var(--text4);margin-bottom:9px">CONFIDENCE → WR</div><div style="position:relative;height:130px"><canvas id="br-prob"></canvas></div></div>
        </div>
        <div style="display:grid;grid-template-columns:1.3fr 1fr;gap:10px">
          <div class="glass" style="padding:13px"><div class="orb" style="font-size:.6rem;color:var(--text4);margin-bottom:9px">MONTHLY PnL &amp; WIN RATE</div><div style="position:relative;height:150px"><canvas id="br-monthly"></canvas></div></div>
          <div class="glass" style="padding:13px;overflow-y:auto;max-height:220px"><div class="orb" style="font-size:.6rem;color:var(--text4);margin-bottom:9px">RECENT TRADES</div><table class="tbl"><thead><tr><th>#</th><th>TIME</th><th>OUT</th><th>PROB</th><th>LOT</th><th>PnL</th></tr></thead><tbody id="br-tbody"></tbody></table></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ████ CONFIG ████████████████████████████████████████████████████████████ -->
  <div id="tab-config" class="tpane" style="align-items:flex-start;justify-content:center;padding:14px;overflow-y:auto">
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;max-width:1100px;width:100%">
      <div class="glass corners corners-b" style="padding:20px;display:flex;flex-direction:column;gap:13px">
        <div class="sh"><div class="sh-dot"></div>SIGNAL FILTERS</div>
        <div><div class="flbl">AI THRESHOLD MIN <span id="lbl-t">50%</span></div><input type="range" min="0.1" max="0.99" step="0.01" value="0.50" id="p-thresh" oninput="document.getElementById('lbl-t').textContent=Math.round(this.value*100)+'%'"></div>
        <div><div class="flbl">AI THRESHOLD MAX <span id="lbl-tmax">65%</span></div><input type="range" min="0.1" max="0.99" step="0.01" value="0.65" id="p-thresh-max" oninput="document.getElementById('lbl-tmax').textContent=Math.round(this.value*100)+'%'"></div>
        <div><label>DIRECTION</label><select id="p-dir"><option value="both">BOTH (auto per-bar bias)</option><option value="bullish">BULLISH</option><option value="bearish">BEARISH</option></select></div>
        <div><label>BLOCKED HOURS (comma list)</label><input type="text" id="p-blk" value="3,4,5,7,9,10,14,19,20,22" style="font-size:.72rem"></div>
        <div><div class="flbl">COOLDOWN (MIN) <span id="lbl-c">0</span></div><input type="range" min="0" max="240" step="5" value="0" id="p-cool" oninput="document.getElementById('lbl-c').textContent=this.value"></div>
      </div>
      <div class="glass corners corners-b" style="padding:20px;display:flex;flex-direction:column;gap:13px">
        <div class="sh"><div class="sh-dot"></div>RISK MANAGEMENT</div>
        <div><div class="flbl">RISK PER TRADE <span id="lbl-r">1.0%</span></div><input type="range" min="0.1" max="20" step="0.1" value="1.0" id="p-risk" oninput="document.getElementById('lbl-r').textContent=this.value+'%'"></div>
        <div><div class="flbl">RR RATIO <span id="lbl-rr">2.0</span></div><input type="range" min="0.5" max="5" step="0.5" value="2.0" id="p-rr" oninput="document.getElementById('lbl-rr').textContent=this.value"></div>
        <div><div class="flbl">BE TRIGGER <span id="lbl-be">1.5R</span></div><input type="range" min="0.5" max="3" step="0.25" value="1.5" id="p-be" oninput="document.getElementById('lbl-be').textContent=this.value+'R'"></div>
        <div><div class="flbl">TRAIL SL (% of TP) <span id="lbl-trail">65%</span></div><input type="range" min="0" max="1" step="0.05" value="0.65" id="p-trail" oninput="document.getElementById('lbl-trail').textContent=Math.round(this.value*100)+'%'"></div>
        <button class="btn btn-p" style="width:100%;padding:11px;margin-top:4px" onclick="pushParams()">⟶ &nbsp; PUSH HOT UPDATE</button>
        <div id="params-msg" style="font-size:.65rem;text-align:center;color:var(--acid);min-height:13px;font-family:'Share Tech Mono'"></div>
      </div>
      <!-- ACTION BUTTONS PANEL -->
      <div class="glass corners corners-b" style="padding:20px;display:flex;flex-direction:column;gap:10px">
        <div class="sh"><div class="sh-dot"></div>SYSTEM ACTIONS</div>
        <div style="display:flex;flex-direction:column;gap:8px">
          <button class="btn btn-ghost" style="width:100%;padding:11px;text-align:left;justify-content:flex-start;gap:10px" onclick="resetCircuitBreaker()">
            <span style="color:var(--amber)">⚡</span> RESET CIRCUIT BREAKER
          </button>
          <button class="btn btn-ghost" style="width:100%;padding:11px;text-align:left;justify-content:flex-start;gap:10px" onclick="cancelPendingOrder()">
            <span style="color:var(--hot)">✕</span> CANCEL PENDING ORDER
          </button>
          <button class="btn btn-ghost" style="width:100%;padding:11px;text-align:left;justify-content:flex-start;gap:10px" onclick="exportTradeLogs()">
            <span style="color:var(--plasma)">⬇</span> EXPORT TRADE LOGS (JSON)
          </button>
          <button class="btn btn-ghost" style="width:100%;padding:11px;text-align:left;justify-content:flex-start;gap:10px" onclick="fetchAndRenderAnalytics()">
            <span style="color:var(--acid)">↻</span> REFRESH LIVE ANALYTICS
          </button>
          <button class="btn btn-ghost" style="width:100%;padding:11px;text-align:left;justify-content:flex-start;gap:10px" onclick="showTab('analytics')">
            <span style="color:var(--violet)">↗</span> OPEN ANALYTICS DASHBOARD
          </button>
        </div>
        <div style="height:1px;background:var(--wire);margin:4px 0"></div>
        <div class="sh" style="margin-top:4px"><div class="sh-dot"></div>SYSTEM STATUS</div>
        <div style="display:flex;flex-direction:column;gap:5px">
          <div class="irow"><span class="ik">BOT STATUS</span><span class="iv" id="cfg-status">OFFLINE</span></div>
          <div class="irow"><span class="ik">CIRCUIT BREAKER</span><span class="iv" id="cfg-cb">NOMINAL</span></div>
          <div class="irow"><span class="ik">PENDING ORDER</span><span class="iv" id="cfg-pend">NONE</span></div>
          <div class="irow"><span class="ik">LAST SIGNAL</span><span class="iv" id="cfg-last">—</span></div>
        </div>
        <div id="cfg-msg" style="font-size:.65rem;text-align:center;color:var(--acid);min-height:13px;font-family:'Share Tech Mono';margin-top:4px"></div>
      </div>
    </div>
  </div>

  </div><!-- /body -->
</div><!-- /pg-main inner -->
</div><!-- /pg-main -->

</div><!-- /ui-root -->

<script>
// ════════════════════════════════════════════════════════════════════════════
// LAYER 1: STARFIELD + GRID
// ════════════════════════════════════════════════════════════════════════════
(function(){
  const cv=document.getElementById('starfield');
  const ctx=cv.getContext('2d');
  let W,H,stars=[],meteors=[];
  function resize(){W=cv.width=window.innerWidth;H=cv.height=window.innerHeight;}
  function mkStars(){
    stars=Array.from({length:320},()=>({
      x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.4+.2,
      a:Math.random(),s:Math.random()*.5+.04,pulse:Math.random()*6,
      hue:Math.random()>0.9?180+Math.random()*40:195
    }));
  }
  function spawnMeteor(){
    if(meteors.length<3&&Math.random()<0.003){
      meteors.push({x:Math.random()*W,y:0,vx:Math.random()*3+1,vy:Math.random()*4+2,life:1,len:Math.random()*80+40});
    }
  }
  function draw(){
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='#020817';ctx.fillRect(0,0,W,H);
    const t=Date.now()/1000;
    // Grid
    ctx.strokeStyle='rgba(6,182,212,.028)';ctx.lineWidth=.5;
    for(let gx=0;gx<W;gx+=80){ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,H);ctx.stroke();}
    for(let gy=0;gy<H;gy+=80){ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke();}
    // Stars
    stars.forEach(s=>{
      const alpha=.25+.45*Math.sin(t*s.s+s.pulse);
      ctx.beginPath();ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
      ctx.fillStyle=`hsla(${s.hue},80%,70%,${alpha*.6})`;ctx.fill();
    });
    // Meteors
    spawnMeteor();
    meteors=meteors.filter(m=>m.life>0);
    meteors.forEach(m=>{
      const grad=ctx.createLinearGradient(m.x,m.y,m.x-m.vx*m.len/m.vy,m.y-m.len);
      grad.addColorStop(0,`rgba(6,182,212,${m.life*.9})`);
      grad.addColorStop(1,'transparent');
      ctx.beginPath();ctx.moveTo(m.x,m.y);ctx.lineTo(m.x-m.vx*m.len/m.vy,m.y-m.len);
      ctx.strokeStyle=grad;ctx.lineWidth=1.5;ctx.stroke();
      m.x+=m.vx;m.y+=m.vy;m.life-=0.015;
    });
    requestAnimationFrame(draw);
  }
  window.addEventListener('resize',()=>{resize();mkStars();});
  resize();mkStars();draw();
})();

// ════════════════════════════════════════════════════════════════════════════
// LAYER 2: DATA RAIN (Matrix digits)
// ════════════════════════════════════════════════════════════════════════════
(function(){
  const cv=document.getElementById('data-rain');
  const ctx=cv.getContext('2d');
  let W,H,cols=[],drops=[];
  const CHARS='0123456789ABCDEF↑↓△▽◇◆█▓░01';
  function resize(){
    W=cv.width=window.innerWidth;H=cv.height=window.innerHeight;
    const nc=Math.floor(W/18);
    cols=Array(nc).fill(0);
    drops=Array(nc).fill(0).map(()=>-Math.random()*H/14);
  }
  function draw(){
    ctx.fillStyle='rgba(2,8,23,.08)';ctx.fillRect(0,0,W,H);
    ctx.font='11px "Share Tech Mono"';
    cols.forEach((_,i)=>{
      const ch=CHARS[Math.floor(Math.random()*CHARS.length)];
      const y=drops[i]*14;
      // Brighter head
      ctx.fillStyle=`rgba(6,182,212,${.85-drops[i]*14/H*.4})`;
      ctx.fillText(ch,i*18,y);
      // Fade trail
      ctx.fillStyle='rgba(6,182,212,.08)';
      ctx.fillText(CHARS[Math.floor(Math.random()*CHARS.length)],i*18,y-14);
      if(y>H&&Math.random()>.975)drops[i]=-Math.random()*5;
      drops[i]+=.4+Math.random()*.3;
    });
    requestAnimationFrame(draw);
  }
  window.addEventListener('resize',resize);
  resize();draw();
})();

// ════════════════════════════════════════════════════════════════════════════
// GLOBE ENGINE — Full interactive Three.js globe for the Globe Tab
// ════════════════════════════════════════════════════════════════════════════
let globeRenderer=null,globeScene=null,globeCamera=null;
let globeIsDragging=false,globeMouse={x:0,y:0};
let globeRotX=0,globeRotY=0,globeVelX=0,globeVelY=0;
let globeZoom=3.2;
let eqMat=null;

/* ═══════════════════════════════════════════════════════════════════════════
   NEXUS GLOBE ENGINE v2 — NEON FUSION SPHERE (drop-in update)
   ---------------------------------------------------------------------------
   HOW TO PLUG IN:
   In your NEXUS script, find the section:
       // GLOBE ENGINE — Full interactive Three.js globe for the Globe Tab
   KEEP these existing top-level declarations exactly as they are:
       let globeRenderer=null,globeScene=null,globeCamera=null;
       let globeIsDragging=false,globeMouse={x:0,y:0};
       let globeRotX=0,globeRotY=0,globeVelX=0,globeVelY=0;
       let globeZoom=3.2;
       let eqMat=null;
   Then DELETE the entire old `function initGlobe(){ ... }` (everything from
   `function initGlobe(){` down to its closing `animate(); }` brace) and paste
   this whole file's `function initGlobe(){...}` in its place.
   Nothing else changes — showTab('globe') still calls initGlobe(), and
   setGlobeRegime(regime, prob) still works and recolors the sphere.
   ═══════════════════════════════════════════════════════════════════════════ */

function initGlobe(){
  if(globeRenderer)return;
  const container=document.getElementById('globe-three');
  if(!container||typeof THREE==='undefined')return;

  /* ── 3D Simplex noise (compact Gustavson port) ── */
  const SimplexNoise=(function(){
    const grad3=[[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],[0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]];
    const p=[];for(let i=0;i<256;i++)p[i]=Math.floor(Math.random()*256);
    const perm=[];for(let i=0;i<512;i++)perm[i]=p[i&255];
    function dot(g,x,y,z){return g[0]*x+g[1]*y+g[2]*z;}
    return function(xin,yin,zin){
      const F3=1/3,G3=1/6;let n0,n1,n2,n3;
      const s=(xin+yin+zin)*F3;
      const i=Math.floor(xin+s),j=Math.floor(yin+s),k=Math.floor(zin+s);
      const t=(i+j+k)*G3;
      const x0=xin-(i-t),y0=yin-(j-t),z0=zin-(k-t);
      let i1,j1,k1,i2,j2,k2;
      if(x0>=y0){ if(y0>=z0){i1=1;j1=0;k1=0;i2=1;j2=1;k2=0;}
        else if(x0>=z0){i1=1;j1=0;k1=0;i2=1;j2=0;k2=1;}
        else{i1=0;j1=0;k1=1;i2=1;j2=0;k2=1;} }
      else{ if(y0<z0){i1=0;j1=0;k1=1;i2=0;j2=1;k2=1;}
        else if(x0<z0){i1=0;j1=1;k1=0;i2=0;j2=1;k2=1;}
        else{i1=0;j1=1;k1=0;i2=1;j2=1;k2=0;} }
      const x1=x0-i1+G3,y1=y0-j1+G3,z1=z0-k1+G3;
      const x2=x0-i2+2*G3,y2=y0-j2+2*G3,z2=z0-k2+2*G3;
      const x3=x0-1+3*G3,y3=y0-1+3*G3,z3=z0-1+3*G3;
      const ii=i&255,jj=j&255,kk=k&255;
      const gi0=perm[ii+perm[jj+perm[kk]]]%12;
      const gi1=perm[ii+i1+perm[jj+j1+perm[kk+k1]]]%12;
      const gi2=perm[ii+i2+perm[jj+j2+perm[kk+k2]]]%12;
      const gi3=perm[ii+1+perm[jj+1+perm[kk+1]]]%12;
      let t0=0.6-x0*x0-y0*y0-z0*z0;
      if(t0<0)n0=0;else{t0*=t0;n0=t0*t0*dot(grad3[gi0],x0,y0,z0);}
      let t1=0.6-x1*x1-y1*y1-z1*z1;
      if(t1<0)n1=0;else{t1*=t1;n1=t1*t1*dot(grad3[gi1],x1,y1,z1);}
      let t2=0.6-x2*x2-y2*y2-z2*z2;
      if(t2<0)n2=0;else{t2*=t2;n2=t2*t2*dot(grad3[gi2],x2,y2,z2);}
      let t3=0.6-x3*x3-y3*y3-z3*z3;
      if(t3<0)n3=0;else{t3*=t3;n3=t3*t3*dot(grad3[gi3],x3,y3,z3);}
      return 32*(n0+n1+n2+n3);
    };
  })();

  /* ── Renderer / scene / camera ── */
  const W=container.clientWidth||800,H=container.clientHeight||600;
  globeRenderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});
  globeRenderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  globeRenderer.setSize(W,H);
  globeRenderer.setClearColor(0x000000,0); // starfield shows through
  container.insertBefore(globeRenderer.domElement,container.firstChild);

  globeScene=new THREE.Scene();
  globeCamera=new THREE.PerspectiveCamera(55,W/H,.1,100);
  globeZoom=4.6;
  globeCamera.position.set(0,0,globeZoom);

  const world=new THREE.Group();      // rotated by drag
  globeScene.add(world);
  const shellGroup=new THREE.Group(); // outer wavy wireframe
  const coreGroup=new THREE.Group();  // inner vortex — own spin, out of sync
  world.add(shellGroup);world.add(coreGroup);

  /* ── round glow sprite texture ── */
  function glowTexture(){
    const cv=document.createElement('canvas');cv.width=cv.height=64;
    const cx=cv.getContext('2d');
    const g=cx.createRadialGradient(32,32,0,32,32,32);
    // soft-cored sprite: less white centre so vertexColors tint dominates
    g.addColorStop(0,'rgba(255,255,255,.9)');
    g.addColorStop(.2,'rgba(255,255,255,.5)');
    g.addColorStop(.55,'rgba(255,255,255,.18)');
    g.addColorStop(1,'rgba(255,255,255,0)');
    cx.fillStyle=g;cx.fillRect(0,0,64,64);
    return new THREE.CanvasTexture(cv);
  }
  const sprite=glowTexture();
  /* darker, fully-tinted sprite for core particles — no white centre, so
     additive stacking lands on the vertex color instead of white */
  function tintTexture(){
    const cv=document.createElement('canvas');cv.width=cv.height=64;
    const cx=cv.getContext('2d');
    const g=cx.createRadialGradient(32,32,0,32,32,32);
    g.addColorStop(0,'rgba(255,255,255,.4)');
    g.addColorStop(.35,'rgba(255,255,255,.2)');
    g.addColorStop(.7,'rgba(255,255,255,.06)');
    g.addColorStop(1,'rgba(255,255,255,0)');
    cx.fillStyle=g;cx.fillRect(0,0,64,64);
    return new THREE.CanvasTexture(cv);
  }
  const coreSprite=tintTexture();

  /* ══ 1) OUTER SHELL — simplex-deformed wireframe + glowing vertices ══ */
  const SHELL_R=1.9;
  const shellGeo=new THREE.SphereGeometry(SHELL_R,56,56);
  const posAttr=shellGeo.attributes.position;
  const vCount=posAttr.count;
  const baseDir=new Float32Array(vCount*3);
  const v=new THREE.Vector3();
  for(let i=0;i<vCount;i++){
    v.set(posAttr.getX(i),posAttr.getY(i),posAttr.getZ(i)).normalize();
    baseDir[i*3]=v.x;baseDir[i*3+1]=v.y;baseDir[i*3+2]=v.z;
  }
  const shellCol=new Float32Array(vCount*3);
  shellGeo.setAttribute('color',new THREE.BufferAttribute(shellCol,3));
  const tmpC=new THREE.Color();
  function paintShell(hexA,hexB){ // gradient along diagonal axis, lerped in HSL
    const cA=new THREE.Color(hexA),cB=new THREE.Color(hexB);
    const hA={},hB={};cA.getHSL(hA);cB.getHSL(hB);
    for(let i=0;i<vCount;i++){
      let mix=.5+.5*(baseDir[i*3]*.55+baseDir[i*3+1]*.75+baseDir[i*3+2]*.2);
      mix=Math.min(1,Math.max(0,mix));
      mix=mix*mix*(3-2*mix); // steepen — fewer washed-out midpoints
      tmpC.setHSL(hA.h+(hB.h-hA.h)*mix,1,.5); // full saturation through the blend
      shellCol[i*3]=tmpC.r;shellCol[i*3+1]=tmpC.g;shellCol[i*3+2]=tmpC.b;
    }
    shellGeo.attributes.color.needsUpdate=true;
  }
  paintShell(0x0090ff,0xff1493); // saturated neon blue → hot pink (default/NEUTRAL)

  const shellWire=new THREE.Mesh(shellGeo,new THREE.MeshBasicMaterial({
    wireframe:true,vertexColors:true,transparent:true,opacity:.14,
    blending:THREE.AdditiveBlending,depthWrite:false}));
  shellGroup.add(shellWire);
  const shellPts=new THREE.Points(shellGeo,new THREE.PointsMaterial({
    vertexColors:true,size:.04,map:sprite,transparent:true,opacity:.55,
    blending:THREE.AdditiveBlending,depthWrite:false,sizeAttenuation:true}));
  shellGroup.add(shellPts);
  eqMat=shellWire.material; // keep old global pointed at something sane

  /* ══ 2) INNER CORE — dense swirling particle vortex ══ */
  const CORE_N=9000,CORE_R=1.15;
  const corePos=new Float32Array(CORE_N*3);
  const coreCol=new Float32Array(CORE_N*3);
  const pR=new Float32Array(CORE_N),pA=new Float32Array(CORE_N),
        pI=new Float32Array(CORE_N),pPh=new Float32Array(CORE_N),
        pSp=new Float32Array(CORE_N);
  const cDeep=new THREE.Color(0x2a44ff),cPurp=new THREE.Color(0x9d3bff),
        cCyan=new THREE.Color(0x55d4ff),cCoral=new THREE.Color(0xff4d33);
  for(let i=0;i<CORE_N;i++){
    const isOuter=i>CORE_N*.82; // outer coral skin — full spherical layer, not a ring
    const r=isOuter?CORE_R*(.95+Math.random()*.1):CORE_R*Math.pow(Math.random(),.62);
    pR[i]=r;pA[i]=Math.random()*Math.PI*2;
    pI[i]=Math.asin(2*Math.random()-1); // uniform over the sphere — 3D cloud, no disc
    pPh[i]=Math.random()*Math.PI*2;
    pSp[i]=(.25+.9*(1-r/CORE_R))*(Math.random()*.5+.75);
    const t=r/CORE_R;
    if(isOuter){tmpC.setHex(0xff2810);}
    else if(t<.35){tmpC.copy(cCyan).lerp(cDeep,.4+Math.random()*.5).multiplyScalar(.7);}
    else{tmpC.copy(cDeep).lerp(cPurp,(t-.35)/.5+Math.random()*.15).multiplyScalar(.65);}
    coreCol[i*3]=tmpC.r;coreCol[i*3+1]=tmpC.g;coreCol[i*3+2]=tmpC.b;
  }
  // split into two draws so the coral outer layer can be bigger + brighter than the body
  const RING_START=Math.floor(CORE_N*.82);
  const coreGeo=new THREE.BufferGeometry();
  coreGeo.setAttribute('position',new THREE.BufferAttribute(corePos,3));
  coreGeo.setAttribute('color',new THREE.BufferAttribute(coreCol,3));
  coreGeo.setDrawRange(0,RING_START);
  const corePtsMat=new THREE.PointsMaterial({
    vertexColors:true,size:.04,map:coreSprite,transparent:true,opacity:1,
    blending:THREE.AdditiveBlending,depthWrite:false,sizeAttenuation:true});
  coreGroup.add(new THREE.Points(coreGeo,corePtsMat));
  const ringGeo=new THREE.BufferGeometry();
  ringGeo.setAttribute('position',new THREE.BufferAttribute(corePos,3));
  ringGeo.setAttribute('color',new THREE.BufferAttribute(coreCol,3));
  ringGeo.setDrawRange(RING_START,CORE_N-RING_START);
  const ringPtsMat=new THREE.PointsMaterial({
    vertexColors:true,size:.09,map:coreSprite,transparent:true,opacity:1,
    blending:THREE.AdditiveBlending,depthWrite:false,sizeAttenuation:true});
  coreGroup.add(new THREE.Points(ringGeo,ringPtsMat));

  /* central plasma glow */
  (function(){
    const cv=document.createElement('canvas');cv.width=cv.height=256;
    const cx=cv.getContext('2d');
    const g=cx.createRadialGradient(128,128,6,128,128,128);
    g.addColorStop(0,'rgba(90,130,255,.55)');
    g.addColorStop(.3,'rgba(50,60,220,.22)');
    g.addColorStop(.7,'rgba(110,40,210,.08)');
    g.addColorStop(1,'rgba(0,0,0,0)');
    cx.fillStyle=g;cx.fillRect(0,0,256,256);
    const spr=new THREE.Sprite(new THREE.SpriteMaterial({
      map:new THREE.CanvasTexture(cv),transparent:true,opacity:.4,
      blending:THREE.AdditiveBlending,depthWrite:false}));
    spr.scale.set(1.4,1.4,1);coreGroup.add(spr);
  })();

  /* ── interaction: drag orbit + cursor flow + click disturb + zoom ── */
  const mouseNDC={x:0,y:0},mouseSm={x:0,y:0};
  let disturb=0; // click ripple energy
  let downXY=null;
  container.addEventListener('mousedown',e=>{
    globeIsDragging=true;globeMouse={x:e.clientX,y:e.clientY};
    downXY={x:e.clientX,y:e.clientY};
  });
  window.addEventListener('mouseup',e=>{
    globeIsDragging=false;
    if(downXY&&Math.abs(e.clientX-downXY.x)+Math.abs(e.clientY-downXY.y)<6)disturb=1; // click = disturb
    downXY=null;
  });
  window.addEventListener('mousemove',e=>{
    const rect=globeRenderer.domElement.getBoundingClientRect();
    mouseNDC.x=((e.clientX-rect.left)/rect.width)*2-1;
    mouseNDC.y=-((e.clientY-rect.top)/rect.height)*2+1;
    if(!globeIsDragging)return;
    globeVelY+=(e.clientX-globeMouse.x)*.0022;
    globeVelX+=(e.clientY-globeMouse.y)*.0022;
    globeMouse={x:e.clientX,y:e.clientY};
  });
  container.addEventListener('wheel',e=>{
    globeZoom=Math.max(2.4,Math.min(9,globeZoom+e.deltaY*.004));
    e.preventDefault();
  },{passive:false});

  new ResizeObserver(()=>{
    const w=container.clientWidth,h=container.clientHeight;
    if(w>0&&h>0){globeRenderer.setSize(w,h);globeCamera.aspect=w/h;globeCamera.updateProjectionMatrix();}
  }).observe(container);

  /* ── regime control (same public API as before) ── */
  window.setGlobeRegime=function(regime,prob){
    const shellMap={BULLISH:[0x10b981,0x22d3ee],BEARISH:[0xf43f5e,0xff2ba6],NEUTRAL:[0x00b3ff,0xff2ba6]};
    const pair=shellMap[regime]||shellMap.NEUTRAL;
    paintShell(pair[0],pair[1]);
    ['glob-regime','gh-regime'].forEach(id=>{const el=document.getElementById(id);if(el)el.textContent=regime;});
    const conf=Math.round((prob||.5)*100);
    const cb=document.getElementById('cl-conf');if(cb)cb.style.width=conf+'%';
    const gb=document.getElementById('gh-conf-bar');if(gb)gb.style.width=conf+'%';
    const gct=document.getElementById('gh-conf');if(gct)gct.textContent=conf+'%';
    const colors={BULLISH:'var(--acid)',BEARISH:'var(--hot)',NEUTRAL:'var(--plasma)'};
    ['glob-regime','gh-regime'].forEach(id=>{const el=document.getElementById(id);if(el)el.style.color=colors[regime]||'var(--plasma)';});
  };

  /* ── animation loop ── */
  const rotDir=new THREE.Vector3(),mouseDir=new THREE.Vector3();
  globeRotX=.15;
  function animate(){
    requestAnimationFrame(animate);
    const t=performance.now()/1000;

    // inertia + auto-rotate
    globeVelX*=.93;globeVelY*=.93;
    globeRotX+=globeVelX;globeRotY+=globeVelY;
    if(!globeIsDragging)globeRotY+=.0018;
    globeRotX=Math.max(-1.4,Math.min(1.4,globeRotX));
    world.rotation.x=globeRotX;world.rotation.y=globeRotY;

    // smoothed cursor + local-space cursor direction
    mouseSm.x+=(mouseNDC.x-mouseSm.x)*.06;
    mouseSm.y+=(mouseNDC.y-mouseSm.y)*.06;
    mouseDir.set(mouseSm.x*1.4,mouseSm.y*1.1,.9).normalize();
    rotDir.copy(mouseDir);
    rotDir.applyAxisAngle(new THREE.Vector3(0,1,0),-globeRotY);
    rotDir.applyAxisAngle(new THREE.Vector3(1,0,0),-globeRotX);

    // camera: zoom easing + gentle drift toward cursor
    globeCamera.position.z+=(globeZoom-globeCamera.position.z)*.08;
    globeCamera.position.x+=(mouseSm.x*.25-globeCamera.position.x)*.04;
    globeCamera.position.y+=(mouseSm.y*.18-globeCamera.position.y)*.04;
    globeCamera.lookAt(0,0,0);

    // click-disturb decay
    disturb*=.965;

    // outer shell: wavy simplex blob + cursor bulge + disturb surge
    const freq=1.15,amp=.30+.28*disturb,ts=t*.35;
    for(let i=0;i<vCount;i++){
      const dx=baseDir[i*3],dy=baseDir[i*3+1],dz=baseDir[i*3+2];
      const n=SimplexNoise(dx*freq+ts,dy*freq+ts*.8,dz*freq-ts*.6);
      const n2=SimplexNoise(dx*3.1-ts*.5,dy*3.1+ts*.4,dz*3.1+ts*.7)*.35;
      const dp=dx*rotDir.x+dy*rotDir.y+dz*rotDir.z;
      const bump=dp>0?Math.pow(dp,5)*.28:0;
      const r=SHELL_R*(1+amp*(n+n2))+bump;
      posAttr.setXYZ(i,dx*r,dy*r,dz*r);
    }
    posAttr.needsUpdate=true;
    shellWire.material.opacity=.24+.10*Math.sin(t*.8)+.2*disturb;
    shellPts.material.size=.040+.012*Math.sin(t*1.3);

    // inner vortex: spiral swirl, out of sync with shell
    coreGroup.rotation.y=-t*.22;
    coreGroup.rotation.z=.28*Math.sin(t*.17);
    const flowX=mouseSm.x*.22,flowY=mouseSm.y*.18;
    const spd=1+2.2*disturb;
    for(let i=0;i<CORE_N;i++){
      const a=pA[i]+t*pSp[i]*spd;
      const r=pR[i]*(1+.06*Math.sin(t*1.4+pPh[i])+.15*disturb);
      const incl=pI[i]+.12*Math.sin(t*.5+pPh[i]);
      const cy=Math.sin(incl)*r+.05*Math.sin(t*2+pPh[i]);
      const cr=Math.cos(incl)*r;
      corePos[i*3]  =cr*Math.cos(a)+flowX*(1-r/CORE_R);
      corePos[i*3+1]=cy            +flowY*(1-r/CORE_R);
      corePos[i*3+2]=cr*Math.sin(a);
    }
    coreGeo.attributes.position.needsUpdate=true;
    ringGeo.attributes.position.needsUpdate=true;
    corePtsMat.opacity=.9+.1*Math.sin(t*1.7);
    ringPtsMat.opacity=.85+.15*Math.sin(t*1.3);

    globeRenderer.render(globeScene,globeCamera);
  }
  animate();
}


// ════════════════════════════════════════════════════════════════════════════
// DASHBOARD LOGIC
// ════════════════════════════════════════════════════════════════════════════
let ws,chart,cSeries,structSeries=[],drawnLines=[];
let eqChart=null,moChart=null,btPoll=null;
let anHourlyChart=null,anDdChart=null,anProbChart=null,anEqChart=null;
let mxHourlyChart=null,mxDdChart=null,mxProbChart=null;
let cOHLC={M5:[],M15:[],H4:[]},cStr={M5:null,M15:null,H4:null};
let cLines=null,curTF='M5',running=false;
const FEEDS=['feed-ov','feed-ch','feed-mx','feed-sig','feed-glob'];
const C={responsive:true,maintainAspectRatio:false,animation:{duration:280},plugins:{legend:{display:false}},
  scales:{x:{grid:{color:'rgba(6,182,212,.04)'},ticks:{color:'#1e3450',font:{size:8}}},
          y:{grid:{color:'rgba(6,182,212,.04)'},ticks:{color:'#1e3450',font:{size:8}}}}};
function fN(n){return n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':n.toFixed(2);}
function $(id){return document.getElementById(id);}
function mkChart(id,cfg){const el=$(id);if(!el)return null;return new Chart(el.getContext('2d'),cfg);}

// ─── Page control ─────────────────────────────────────────────────────────
function showPg(id){['pg-login','pg-gateway','pg-main'].forEach(p=>$(p).classList.remove('on'));$(id).classList.add('on');}
function showTab(t){
  ['overview','chart','globe','matrix','analytics','signals','backtest','config'].forEach(id=>{
    $('tab-'+id).classList.remove('on');$('nav-'+id).classList.remove('on');
  });
  $('tab-'+t).classList.add('on');$('nav-'+t).classList.add('on');
  if(t==='chart')setTimeout(()=>{initChart();chart&&chart.timeScale().fitContent();},60);
  if(t==='globe')setTimeout(()=>{initGlobe();},80);
  if(t==='analytics')setTimeout(()=>{fetchAndRenderAnalytics();},80);
}
function btTab(t){
  ['setup','run','report'].forEach(id=>{$('bt-'+id).style.display='none';$('bnt-'+id).classList.remove('on');});
  $('bt-'+t).style.display='flex';$('bnt-'+t).classList.add('on');
}

// ─── Auth ──────────────────────────────────────────────────────────────────
function webLogin(){
  $('login-err').textContent='';
  fetch('/api/web-login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({username:$('web-user').value,password:$('web-pass').value})})
  .then(r=>r.json()).then(d=>{
    if(d.status==='success'){localStorage.setItem('web_token','ok');loadSymbols();showPg('pg-gateway');checkStatus();}
    else $('login-err').textContent=d.msg||'AUTH FAILED';
  }).catch(()=>{$('login-err').textContent='NET ERROR';});
}
function webLogout(){if(running)fetch('/api/stop',{method:'POST'}).catch(()=>{});running=false;localStorage.removeItem('web_token');showPg('pg-login');}
window.onload=()=>{if(localStorage.getItem('web_token')==='ok'){showPg('pg-gateway');loadSymbols();checkStatus();}else showPg('pg-login');};
function loadSymbols(){fetch('/api/symbols').then(r=>r.json()).then(d=>{const s=$('symbol-select');s.innerHTML='';d.symbols.forEach(sym=>{const o=document.createElement('option');o.value=o.textContent=sym;s.appendChild(o);});}).catch(()=>{});}
function checkStatus(){fetch('/api/status').then(r=>r.json()).then(d=>{if(d.is_running){running=true;setRunUI(true);showPg('pg-main');setTimeout(()=>{initChart();initEq();initMo();connectWS();startTickPoll();},150);}}).catch(()=>{});}

// ─── Live price poll ───────────────────────────────────────────────────────
let tickPollInt=null;
function startTickPoll(){
  if(tickPollInt)clearInterval(tickPollInt);
  tickPollInt=setInterval(()=>{
    fetch('/api/tick').then(r=>r.json()).then(d=>{
      if(d.price&&$('hdr-price'))$('hdr-price').textContent=d.price.toFixed(5||2);
    }).catch(()=>{});
  },2000);
}

// ─── Engine ────────────────────────────────────────────────────────────────
function toggleEngine(){
  $('gw-err').textContent='';
  if(running){
    fetch('/api/stop',{method:'POST'}).then(r=>r.json()).then(d=>{
      running=false;setRunUI(false);showPg('pg-gateway');termLog(d.msg);
      if(tickPollInt){clearInterval(tickPollInt);tickPollInt=null;}
    }).catch(()=>{});return;
  }
  const b=$('btn-engage');b.textContent='LINKING…';b.disabled=true;
  fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    account:$('mt5-account').value,password:$('mt5-pass').value,server:$('mt5-server').value,
    symbol:$('symbol-select').value,direction:$('gw-dir').value,rr:$('gw-rr').value})})
  .then(r=>r.json()).then(d=>{
    b.disabled=false;
    if(d.status==='success'){running=true;setRunUI(true);showPg('pg-main');setTimeout(()=>{initChart();initEq();initMo();connectWS();startTickPoll();},200);}
    else{b.textContent='⟶ ENGAGE PIPELINE';$('gw-err').textContent=d.msg||'FAILED';}
  }).catch(e=>{b.disabled=false;b.textContent='⟶ ENGAGE PIPELINE';$('gw-err').textContent='NET ERROR';});
}
function setRunUI(on){
  const hd=$('hdr-sig'),gw=$('gw-sig');
  hd.className='sig';gw.className='sig';
  $('hdr-dot').className='dot '+(on?'dot-live':'dot-idle');
  $('hdr-stxt').textContent=on?'LIVE':'OFFLINE';
  hd.style.color=on?'var(--acid)':'var(--hot)';
  $('gw-stxt').textContent=on?'LIVE':'OFFLINE';
  gw.style.color=on?'var(--acid)':'var(--hot)';
  $('gw-title').textContent=on?'PIPELINE ACTIVE':'ESTABLISH CONNECTION';
  ['mt5-account','mt5-pass','symbol-select','gw-dir','gw-rr'].forEach(id=>{const e=$(id);if(e)e.disabled=on;});
  const b=$('btn-engage');b.textContent=on?'■ STOP PIPELINE':'⟶ ENGAGE PIPELINE';
  b.className='btn '+(on?'btn-red':'btn-p')+' ';
  if(!on){cLines=null;drawnLines.forEach(l=>{try{cSeries&&cSeries.removePriceLine(l);}catch(e2){}});drawnLines=[];}
}

// ─── WebSocket ─────────────────────────────────────────────────────────────
function connectWS(){
  if(ws&&ws.readyState===WebSocket.OPEN)return;
  ws=new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage=e=>{const d=JSON.parse(e.data);
    if(d.type==='log')termLog(d.msg);
    else if(d.type==='multi_tf_update'){cOHLC=d.ohlc;cStr=d.structure;renderChart();}
    else if(d.type==='trade_lines'){cLines=d;drawTrades();}
    else if(d.type==='full_state')onFS(d);
    else if(d.type==='pipeline_stage'){try{GlobeFX.onStage(d);}catch(e2){}}};
  ws.onerror=()=>termLog('<span class="err">[WS] ERROR</span>');
  ws.onclose=()=>{if(running)setTimeout(connectWS,3000);};
}

function onFS(d){
  // Header
  $('hdr-bal').textContent='$'+d.balance.toFixed(2);
  const td=d.today_pnl||0;const te=$('hdr-today');te.textContent=(td>=0?'+$':'-$')+Math.abs(td).toFixed(2);te.style.color=td>=0?'var(--acid)':'var(--hot)';
  // PnL heroes
  const ph=(id,v)=>{const el=$(id);if(!el)return;el.textContent=(v>=0?'+$':'-$')+fN(Math.abs(v));el.style.color=v>=0?'currentColor':'var(--hot)';};
  ph('ov-today',d.today_pnl||0);ph('ov-week',d.week_pnl||0);ph('ov-month',d.month_pnl||0);ph('ov-pnl',d.pnl||0);
  // Primary tiles — animate value changes
  function setVal(id,txt){const el=$(id);if(!el)return;if(el.textContent!==txt){el.style.transition='color .1s';el.style.color='#fff';setTimeout(()=>{el.style.color='';},300);}el.textContent=txt;}
  setVal('m-wr',(d.winrate||0).toFixed(1)+'%');
  setVal('m-pf',(d.profit_factor||0).toFixed(2));
  setVal('m-sh',(d.sharpe||0).toFixed(2));
  setVal('m-dd',(d.max_drawdown||0).toFixed(2)+'%');
  setVal('m-exp','$'+(d.expectancy||0).toFixed(2));
  setVal('m-minbal','$'+(d.min_balance||0).toFixed(2));
  // Secondary tiles
  setVal('m-total',String(d.trades||0));
  setVal('m-wins',String(d.wins||0));
  const losses=d.losses||Math.max(0,(d.trades||0)-(d.wins||0));
  setVal('m-losses',String(losses));
  $('m-wl').textContent=(d.wins||0)+'W / '+losses+'L';
  if(d.gross_p!==undefined)setVal('m-gp','$'+fN(d.gross_p||0));
  if(d.gross_l!==undefined)setVal('m-gl','$'+fN(d.gross_l||0));
  // New tertiary tiles
  if(d.avg_win!==undefined){setVal('m-avgwin','$'+fN(d.avg_win||0));setVal('an-avgwin','$'+fN(d.avg_win||0));}
  if(d.avg_loss!==undefined){setVal('m-avgloss','$'+fN(d.avg_loss||0));setVal('an-avgloss','$'+fN(d.avg_loss||0));}
  if(d.session_stats){
    const ss=d.session_stats;
    setVal('m-tdtrades',String(ss.trades_today||0));setVal('an-tdtrades',String(ss.trades_today||0));
    setVal('m-streak',String(ss.streak||0));setVal('an-streak',String(ss.streak||0));
    const st=ss.streak_type||'—';
    if($('m-strktype'))$('m-strktype').textContent=st;
    if($('an-strktype')){$('an-strktype').textContent=st;$('an-strktype').style.color=st==='Win'?'var(--acid)':st==='Loss'?'var(--hot)':'var(--text3)';}
    setVal('m-peak','$'+fN(ss.peak_equity||0));
    if($('an-peak'))$('an-peak').textContent='$'+fN(ss.peak_equity||0);
    const avg_rr=ss.avg_rr||0;setVal('m-avgrr',avg_rr.toFixed(2));
    if($('an-rr'))$('an-rr').textContent=avg_rr.toFixed(2)+'x';
    // Streak dots
    const sd=$('an-streak-dots');
    if(sd){sd.innerHTML='';for(let i=0;i<Math.min(ss.streak||0,10);i++){const dot=document.createElement('div');dot.style.cssText=`width:8px;height:8px;border-radius:50%;background:${st==='Win'?'var(--acid)':'var(--hot)'};box-shadow:0 0 4px currentColor`;sd.appendChild(dot);}}
  }
  // Analytics panel quick stats
  const anphl=(id,v)=>{const el=$(id);if(!el)return;el.textContent=(v>=0?'+$':'-$')+fN(Math.abs(v));el.style.color=v>=0?'var(--acid)':'var(--hot)';};
  anphl('an-today',d.today_pnl||0);anphl('an-per-today',d.today_pnl||0);anphl('an-per-week',d.week_pnl||0);anphl('an-per-month',d.month_pnl||0);anphl('an-per-90',d.pnl||0);
  if($('an-pf')){$('an-pf').textContent=(d.profit_factor||0).toFixed(2);const pfw=Math.min(100,(d.profit_factor||0)/5*100);if($('an-pf-bar'))$('an-pf-bar').style.width=pfw+'%';}
  if($('an-sh')){$('an-sh').textContent=(d.sharpe||0).toFixed(2);const shw=Math.min(100,(d.sharpe||0)/3*100);if($('an-sh-bar'))$('an-sh-bar').style.width=Math.max(0,shw)+'%';}
  if($('an-wr')){$('an-wr').textContent=(d.winrate||0).toFixed(1)+'%';if($('an-wr-bar'))$('an-wr-bar').style.width=(d.winrate||0)+'%';}
  if($('an-exp'))$('an-exp').textContent='$'+(d.expectancy||0).toFixed(2);
  if($('an-dd2'))$('an-dd2').textContent=(d.max_drawdown||0).toFixed(2)+'%';
  if($('an-maxdd'))$('an-maxdd').textContent=(d.max_drawdown||0).toFixed(2)+'%';
  if($('an-minbal'))$('an-minbal').textContent='$'+fN(d.min_balance||0);
  const rf=d.pnl&&d.metrics&&d.max_drawdown>0?((d.pnl||0)/(d.max_drawdown||1)).toFixed(2):'—';if($('an-recovery'))$('an-recovery').textContent=rf;
  // Config status panel
  if($('cfg-status'))$('cfg-status').textContent=running?'LIVE':'OFFLINE';
  if($('cfg-status'))$('cfg-status').style.color=running?'var(--acid)':'var(--hot)';
  if($('cfg-cb')){const cb2=d.circuit_breaker;$('cfg-cb').textContent=cb2?'⚡ PAUSED':'NOMINAL';$('cfg-cb').style.color=cb2?'var(--amber)':'var(--acid)';}
  if($('cfg-last')&&d.last_signal)$('cfg-last').textContent=d.last_signal.time||'—';
  bot_state_pending=d.pending_ticket||null;
  if($('cfg-pend'))$('cfg-pend').textContent=(bot_state_pending?`#${bot_state_pending}`:'NONE');
  if(d.outcome_counts)updateLiveDonut(d.outcome_counts);
  if(d.prob_wr_series&&d.prob_wr_series.length)renderAnProb(d.prob_wr_series);
  // Wire matrix mini-charts from live data
  if(d.hourly_pnl&&d.hourly_pnl.length){renderMxHourly(d.hourly_pnl);renderAnHourly(d.hourly_pnl);}
  if(d.drawdown_series&&d.drawdown_series.length){renderMxDd(d.drawdown_series);renderAnDd(d.drawdown_series);}
  // Analytics equity
  if(d.equity_history&&d.equity_history.length>1)updateAnEq(d.equity_history);
  // Analytics trade table
  if(d.trade_log)renderAnTradeTable(d.trade_log);
  // Charts
  if(d.equity_history&&d.equity_history.length>1){
    updateEq(d.equity_history);
    const s=d.equity_history[0].balance,e2=d.equity_history[d.equity_history.length-1].balance;
    const chg=((e2-s)/s*100).toFixed(2);const ep=$('ov-eq-pct');ep.textContent=(chg>=0?'+':'')+chg+'%';ep.style.color=chg>=0?'var(--acid)':'var(--hot)';
  }
  if(d.monthly_breakdown&&d.monthly_breakdown.length>0)updateMo(d.monthly_breakdown);
  // Circuit breaker
  const cb=d.circuit_breaker;
  $('cb-tag').style.display=cb?'flex':'none';if(cb&&$('cb-until'))$('cb-until').textContent=d.circuit_until||'';
  $('cb-status').textContent=cb?'⚡ PAUSED':'NOMINAL';$('cb-status').style.color=cb?'var(--amber)':'var(--acid)';
  $('cb-until2').textContent=d.circuit_until||'—';
  const rwr=d.recent_wr||0;$('roll-wr').textContent=rwr.toFixed(1)+'%';
  $('roll-wr').style.color=rwr<35?'var(--hot)':rwr<50?'var(--amber)':'var(--acid)';
  $('roll-bar').style.width=Math.min(100,rwr)+'%';$('roll-bar').style.background=rwr<35?'var(--hot)':rwr<50?'var(--amber)':'var(--acid)';
  // Globe regime
  if(typeof setGlobeRegime==='function'&&d.last_signal){
    const bias=(d.last_signal.bias||'').toLowerCase();
    setGlobeRegime(bias==='bullish'?'BULLISH':bias==='bearish'?'BEARISH':'NEUTRAL',d.last_signal.prob||.5);
    if($('glob-bias'))$('glob-bias').textContent=d.last_signal.bias||'—';
    if($('glob-pb'))$('glob-pb').textContent=((d.last_signal.pullback_depth||0)*100).toFixed(1)+'%';
    if($('glob-mom'))$('glob-mom').textContent=d.last_signal.direction||'—';
    if($('glob-amp'))$('glob-amp').textContent=d.winrate?(d.winrate>50?'EXPANSION':'CONTRACTION'):'—';
    if($('cl-mtf'))$('cl-mtf').style.width=(d.winrate||0)+'%';
    if($('cl-str'))$('cl-str').style.width=((d.profit_factor||0)*20)+'%';
    if($('cl-pb'))$('cl-pb').style.width=((d.last_signal.pullback_depth||0)*100)+'%';
  }
  if($('glob-cb'))$('glob-cb').textContent=cb?'⚡ PAUSED':'NOMINAL';
  if($('glob-cb'))$('glob-cb').style.color=cb?'var(--amber)':'var(--acid)';
  // Signal
  if(d.last_signal){const ls=d.last_signal;
    $('sig-time').textContent=ls.time||'—';
    const de=$('sig-dir');de.textContent=ls.direction||'—';de.style.color=ls.direction==='BEAR'?'var(--hot)':ls.direction==='BULL'?'var(--acid)':'var(--text3)';
    $('sig-bias').textContent=ls.bias||'—';$('sig-mode').textContent=ls.entry_mode||'—';
    $('sig-lmt').textContent=ls.limit_price?ls.limit_price.toFixed(5):'—';
    const ae=$('sig-act');ae.textContent=ls.action||'—';ae.style.color=ls.action&&ls.action.includes('PLACED')?'var(--acid)':ls.action&&ls.action.includes('SKIP')?'var(--amber)':'var(--text3)';
    const pr=(ls.prob||0)*100;$('sig-prob').textContent=pr.toFixed(1)+'%';$('pb-prob').style.width=Math.min(100,pr)+'%';
    const pb=Math.min(100,(ls.pullback_depth||0)*100);$('sig-pb').textContent=pb.toFixed(1)+'%';$('pb-depth').style.width=pb+'%';}
  // Skip counters
  if(d.skip_counts){const con=$('skip-cont');con.innerHTML='';
    const labels={session_block:'SESSION BLOCK',circuit_breaker:'CIRCUIT BREAKER',trade_open:'TRADE OPEN',cooldown:'COOLDOWN',limit_missed:'LIMIT MISSED',invalid_structure:'INVALID STRUCTURE',sl_too_tight:'SL TOO TIGHT',sl_too_wide:'SL TOO WIDE',threshold_low:'BELOW MIN THRESH',threshold_high:'ABOVE MAX THRESH',direction_filter:'DIRECTION FILTER'};
    const colors={limit_missed:'var(--amber)',session_block:'var(--ion)',trade_open:'var(--violet)',circuit_breaker:'var(--hot)',threshold_low:'var(--plasma)'};
    const total=Object.values(d.skip_counts).reduce((a,b)=>a+b,0)||1;
    Object.entries(d.skip_counts).sort((a,b)=>b[1]-a[1]).forEach(([k,v])=>{if(!v)return;const c=colors[k]||'var(--plasma)';const pct=Math.min(100,v/total*100);const row=document.createElement('div');row.className='srow';row.innerHTML=`<span style="font-size:.56rem;color:var(--text2);min-width:155px;font-family:'Rajdhani';letter-spacing:.07em">${labels[k]||k}</span><div class="sbar-w"><div class="sbar" style="width:${pct}%;background:${c}"></div></div><span class="tech" style="font-size:.65rem;color:${c};min-width:28px;text-align:right">${v}</span>`;con.appendChild(row);});}
  // Matrix trade log
  if(d.trade_log){const tb=$('mx-tbody');if(tb){tb.innerHTML='';[...d.trade_log].reverse().slice(0,20).forEach(t=>{const pnl=t.pnl||0;const tr=document.createElement('tr');tr.innerHTML=`<td style="color:var(--text2)">${(t.time||'').slice(0,14)}</td><td><span class="badge b${(t.outcome||'L')[0]}">${t.outcome||'—'}</span></td><td style="color:var(--plasma)">${((t.ai_prob||0)*100).toFixed(1)}%</td><td style="color:${pnl>=0?'var(--acid)':'var(--hot)'}">${pnl>=0?'+':''}$${fN(Math.abs(pnl))}</td><td style="color:var(--text3)">$${fN(t.balance||0)}</td>`;tb.appendChild(tr);});}}
}

// ─── Analytics helpers ─────────────────────────────────────────────────────
let bot_state_pending=null;

function renderMxHourly(data){
  const blk=new Set([3,4,5,7,9,10,14,19,20,22]);
  const cfg2={type:'bar',data:{labels:data.map(r=>'H'+r.hour),datasets:[{data:data.map(r=>r.pnl),backgroundColor:data.map(r=>blk.has(r.hour)?'rgba(244,63,94,.4)':'rgba(6,182,212,.55)'),borderRadius:2,borderSkipped:false}]},options:{...C,scales:{x:{...C.scales.x},y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>'$'+fN(v)}}}}};
  if(mxHourlyChart){mxHourlyChart.data=cfg2.data;mxHourlyChart.update('none');}else{const e=$('mx-hourly');if(e)mxHourlyChart=new Chart(e.getContext('2d'),cfg2);}
}
function renderMxDd(data){
  const stride=Math.max(1,Math.floor(data.length/80));const sl=data.filter((_,i)=>i%stride===0);
  const cfg2={type:'line',data:{labels:sl.map(r=>r.time),datasets:[{data:sl.map(r=>r.dd),borderColor:'#f43f5e',backgroundColor:'rgba(244,63,94,.07)',borderWidth:1.5,fill:true,tension:.3,pointRadius:0}]},options:{...C,scales:{x:{...C.scales.x,ticks:{...C.scales.x.ticks,maxTicksLimit:4}},y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>v+'%'}}}}};
  if(mxDdChart){mxDdChart.data=cfg2.data;mxDdChart.update('none');}else{const e=$('mx-dd');if(e)mxDdChart=new Chart(e.getContext('2d'),cfg2);}
}
function renderAnHourly(data){
  const blk=new Set([3,4,5,7,9,10,14,19,20,22]);
  const best=data.reduce((a,b)=>b.pnl>a.pnl?b:a,{hour:-1,pnl:-Infinity});
  if($('an-best-hour')&&best.hour>=0)$('an-best-hour').textContent='BEST: H'+best.hour+' (+$'+fN(best.pnl)+')';
  const cfg2={type:'bar',data:{labels:data.map(r=>'H'+r.hour),datasets:[{data:data.map(r=>r.pnl),backgroundColor:data.map(r=>blk.has(r.hour)?'rgba(244,63,94,.35)':'rgba(6,182,212,.55)'),borderRadius:2,borderSkipped:false}]},options:{...C,scales:{x:{...C.scales.x},y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>'$'+fN(v)}}}}};
  if(anHourlyChart){anHourlyChart.data=cfg2.data;anHourlyChart.update('none');}else{const e=$('an-hourly');if(e)anHourlyChart=new Chart(e.getContext('2d'),cfg2);}
}
function renderAnDd(data){
  const stride=Math.max(1,Math.floor(data.length/120));const sl=data.filter((_,i)=>i%stride===0);
  const cfg2={type:'line',data:{labels:sl.map(r=>r.time),datasets:[{data:sl.map(r=>r.dd),borderColor:'#f43f5e',backgroundColor:'rgba(244,63,94,.07)',borderWidth:1.5,fill:true,tension:.3,pointRadius:0}]},options:{...C,scales:{x:{...C.scales.x,ticks:{...C.scales.x.ticks,maxTicksLimit:5}},y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>v+'%'}}}}};
  if(anDdChart){anDdChart.data=cfg2.data;anDdChart.update('none');}else{const e=$('an-dd');if(e)anDdChart=new Chart(e.getContext('2d'),cfg2);}
}
function renderAnProb(data){
  const p2=(data||[]).filter(x=>x.count>0);
  const cfg2={type:'bar',data:{labels:p2.map(x=>x.label||x.pb_bin||''),datasets:[{data:p2.map(x=>x.wr),backgroundColor:'rgba(6,182,212,.6)',borderRadius:2,borderSkipped:false}]},options:{...C,scales:{x:{...C.scales.x},y:{...C.scales.y,min:0,max:100,ticks:{...C.scales.y.ticks,callback:v=>v+'%'}}}}};
  if(anProbChart){anProbChart.data=cfg2.data;anProbChart.update('none');}else{const e=$('an-prob');if(e)anProbChart=new Chart(e.getContext('2d'),cfg2);}
  // Also wire matrix prob chart
  if(mxProbChart){mxProbChart.data=cfg2.data;mxProbChart.update('none');}else{const e=$('mx-prob');if(e)mxProbChart=new Chart(e.getContext('2d'),cfg2);}
}
function updateAnEq(h){
  const cfg2={type:'line',data:{labels:h.map(e=>e.time),datasets:[{data:h.map(e=>e.balance),borderColor:'#06b6d4',backgroundColor:'rgba(6,182,212,.07)',borderWidth:1.5,fill:true,tension:.42,pointRadius:0}]},options:{...C,scales:{x:{...C.scales.x,ticks:{...C.scales.x.ticks,maxTicksLimit:7}},y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>'$'+fN(v)}}}}};
  if(anEqChart){anEqChart.data=cfg2.data;anEqChart.update('none');}else{const e=$('an-equity');if(e)anEqChart=new Chart(e.getContext('2d'),cfg2);}
}
function renderAnTradeTable(trades){
  const tb=$('an-tbody');if(!tb)return;
  tb.innerHTML='';
  if($('an-tradecount'))$('an-tradecount').textContent='['+trades.length+' records]';
  [...trades].reverse().slice(0,30).forEach(t=>{
    const pnl=t.pnl||0;const tr=document.createElement('tr');
    tr.innerHTML=`<td style="color:var(--text2)">${(t.time||'').slice(0,14)}</td><td><span class="badge b${(t.outcome||'L')[0]}">${t.outcome||'—'}</span></td><td style="color:${pnl>=0?'var(--acid)':'var(--hot)'}">${pnl>=0?'+':''}$${fN(Math.abs(pnl))}</td><td style="color:var(--text3)">$${fN(t.balance||0)}</td>`;
    tb.appendChild(tr);
  });
}
function fetchAndRenderAnalytics(){
  fetch('/api/live-analytics').then(r=>r.json()).then(d=>{
    if(d.status!=='ok')return;
    if(d.hourly_pnl&&d.hourly_pnl.length){renderAnHourly(d.hourly_pnl);renderMxHourly(d.hourly_pnl);}
    if(d.drawdown_series&&d.drawdown_series.length){renderAnDd(d.drawdown_series);renderMxDd(d.drawdown_series);}
    if(d.trade_log)renderAnTradeTable(d.trade_log);
    if(d.session_stats){
      const ss=d.session_stats;
      if($('an-avgwin'))$('an-avgwin').textContent='$'+fN(ss.avg_win||0);
      if($('an-avgloss'))$('an-avgloss').textContent='$'+fN(ss.avg_loss||0);
      if($('an-rr'))$('an-rr').textContent=(ss.avg_rr||0).toFixed(2)+'x';
      if($('an-tdtrades'))$('an-tdtrades').textContent=String(ss.trades_today||0);
      if($('an-streak'))$('an-streak').textContent=String(ss.streak||0);
      if($('an-strktype')){$('an-strktype').textContent=ss.streak_type||'—';$('an-strktype').style.color=ss.streak_type==='Win'?'var(--acid)':ss.streak_type==='Loss'?'var(--hot)':'var(--text3)';}
      if($('an-peak'))$('an-peak').textContent='$'+fN(ss.peak_equity||0);
    }
  }).catch(()=>{});
}
function resetCircuitBreaker(){
  fetch('/api/reset-circuit-breaker',{method:'POST'}).then(r=>r.json()).then(d=>{
    const m=$('cfg-msg');if(m){m.textContent=d.msg||'Done';m.style.color=d.status==='ok'?'var(--acid)':'var(--hot)';setTimeout(()=>m.textContent='',3000);}
  }).catch(()=>{});
}
function cancelPendingOrder(){
  fetch('/api/cancel-pending',{method:'POST'}).then(r=>r.json()).then(d=>{
    const m=$('cfg-msg');if(m){m.textContent=d.msg||'Done';m.style.color=d.status==='ok'?'var(--acid)':'var(--hot)';setTimeout(()=>m.textContent='',3000);}
  }).catch(()=>{});
}
function exportTradeLogs(){
  fetch('/api/export-logs',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.status!=='ok')return;
    const blob=new Blob([JSON.stringify(d.trades,null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;
    a.download='nexus_trade_log_'+new Date().toISOString().slice(0,10)+'.json';a.click();URL.revokeObjectURL(url);
  }).catch(()=>{});
}

// ─── Chart.js ─────────────────────────────────────────────────────────────
function initEq(){if(eqChart)return;const ctx=$('c-equity')?.getContext('2d');if(!ctx)return;eqChart=new Chart(ctx,{type:'line',data:{labels:[],datasets:[{data:[],borderColor:'#06b6d4',backgroundColor:'rgba(6,182,212,.07)',borderWidth:1.5,fill:true,tension:.42,pointRadius:0}]},options:{...C,scales:{x:{...C.scales.x,ticks:{...C.scales.x.ticks,maxTicksLimit:7}},y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>'$'+fN(v)}}}}});}
function updateEq(h){if(!eqChart)initEq();if(!eqChart)return;eqChart.data.labels=h.map(e=>e.time);eqChart.data.datasets[0].data=h.map(e=>e.balance);eqChart.update('none');}
function initMo(){if(moChart)return;const ctx=$('c-monthly')?.getContext('2d');if(!ctx)return;moChart=new Chart(ctx,{type:'bar',data:{labels:[],datasets:[{data:[],backgroundColor:[],borderRadius:2,borderSkipped:false,yAxisID:'y'},{data:[],type:'line',borderColor:'#3b82f6',borderWidth:1.5,pointRadius:2,tension:.3,yAxisID:'y2',fill:false,pointBackgroundColor:'#3b82f6'}]},options:{...C,scales:{y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>'$'+fN(v)}},y2:{position:'right',grid:{display:false},ticks:{color:'#1e3450',font:{size:8},callback:v=>v+'%'}},x:{...C.scales.x}}}});}
function updateMo(data){if(!moChart)initMo();if(!moChart)return;moChart.data.labels=data.map(d=>d.month);moChart.data.datasets[0].data=data.map(d=>d.pnl);moChart.data.datasets[0].backgroundColor=data.map(d=>d.pnl>=0?'rgba(16,185,129,.6)':'rgba(244,63,94,.55)');moChart.data.datasets[1].data=data.map(d=>d.wr);moChart.update('none');}
function mkDonut(id,labels,vals,colors){const ctx=$(id)?.getContext('2d');if(!ctx)return;return new Chart(ctx,{type:'doughnut',data:{labels,datasets:[{data:vals,backgroundColor:colors,borderColor:colors.map(c=>c),borderWidth:1,hoverOffset:5}]},options:{responsive:true,maintainAspectRatio:true,cutout:'70%',plugins:{legend:{display:false}},animation:{duration:400}}});}
function buildLeg(legId,labels,vals,colors){const leg=$(legId);if(!leg)return;leg.innerHTML='';const tot=vals.reduce((a,b)=>a+b,0)||1;labels.forEach((l,i)=>{const pct=(vals[i]/tot*100).toFixed(1);const d=document.createElement('div');d.style.cssText='display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-family:Share Tech Mono;font-size:.6rem;transition:color .2s;cursor:default';d.innerHTML=`<span style="display:flex;align-items:center;gap:5px;color:var(--text2)"><span style="width:7px;height:7px;border-radius:50%;background:${colors[i]};flex-shrink:0;box-shadow:0 0 5px ${colors[i]}"></span>${l}</span><span style="color:${colors[i]}">${vals[i]} (${pct}%)</span>`;leg.appendChild(d);});}

// Init charts on load
let liveDonut=null;
const OV_OC_LABELS=['Win','Loss','BE'];
const OV_OC_COLORS=['rgba(16,185,129,.85)','rgba(244,63,94,.85)','rgba(245,158,11,.85)'];
function updateLiveDonut(oc){
  const vals=OV_OC_LABELS.map(l=>oc[l]||0);
  if(!liveDonut){liveDonut=mkDonut('c-donut',OV_OC_LABELS,vals,OV_OC_COLORS);}
  else{liveDonut.data.datasets[0].data=vals;liveDonut.update('none');}
  buildLeg('donut-leg',OV_OC_LABELS,vals,OV_OC_COLORS);
}
setTimeout(()=>{initEq();initMo();updateLiveDonut({Win:0,Loss:0,BE:0});},150);

// ─── TradingView chart ─────────────────────────────────────────────────────
function initChart(){
  if(chart)return;if(typeof LightweightCharts==='undefined'){setTimeout(initChart,400);return;}
  const el=$('tvchart');if(!el)return;if(el.clientHeight===0)el.style.minHeight='300px';
  chart=LightweightCharts.createChart(el,{
    layout:{textColor:'#1e3450',background:{type:'solid',color:'transparent'}},
    grid:{vertLines:{color:'rgba(6,182,212,.04)'},horzLines:{color:'rgba(6,182,212,.04)'}},
    crosshair:{mode:0},timeScale:{borderColor:'rgba(6,182,212,.15)',timeVisible:true,secondsVisible:false},
    rightPriceScale:{borderColor:'rgba(6,182,212,.15)'},handleScroll:true,handleScale:true});
  cSeries=chart.addCandlestickSeries({upColor:'#10b981',downColor:'#f43f5e',borderVisible:false,wickUpColor:'#10b981',wickDownColor:'#f43f5e'});
  new ResizeObserver(es=>{if(!es[0])return;const r=es[0].contentRect;if(r.width>0&&r.height>0)chart.applyOptions({width:r.width,height:r.height});}).observe(el);
  if(cOHLC[curTF]&&cOHLC[curTF].length>0)renderChart();
}
function renderChart(){
  if(!chart||!cSeries)return;
  try{const d=cOHLC[curTF];if(d&&d.length>0){cSeries.setData(d);chart.timeScale().fitContent();$('cdebug').textContent=d.length+' BARS · '+curTF;}}catch(e){}
  try{if(cStr[curTF])applyStr(cStr[curTF]);}catch(e){}
  if(cLines)drawTrades();
}
function applyStr(d){
  if(!chart||!cSeries)return;try{cSeries.setMarkers([]);}catch(e){}
  structSeries.forEach(s=>{try{chart.removeSeries(s);}catch(e){}});structSeries=[];
  if(d.markers&&d.markers.length){const seen=new Set(),clean=[];[...d.markers].sort((a,b)=>a.time-b.time).forEach(m=>{if(!seen.has(m.time)){clean.push(m);seen.add(m.time);}});try{cSeries.setMarkers(clean);}catch(e){}}
  (d.segments||[]).forEach(seg=>{const s=chart.addLineSeries({color:seg.color,lineWidth:2,lineStyle:seg.style,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});s.setData([...seg.data].sort((a,b)=>a.time-b.time));structSeries.push(s);});
  (d.zones||[]).forEach(z=>{const t=chart.addLineSeries({color:z.color,lineWidth:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});const b=chart.addLineSeries({color:z.color,lineWidth:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});t.setData([...z.top].sort((a,b)=>a.time-b.time));b.setData([...z.bottom].sort((a,b)=>a.time-b.time));structSeries.push(t,b);});
}
function drawTrades(){drawnLines.forEach(l=>{try{cSeries&&cSeries.removePriceLine(l);}catch(e){}});drawnLines=[];if(!cSeries||!cLines)return;drawnLines.push(cSeries.createPriceLine({price:cLines.entry,color:'#06b6d4',lineWidth:2,title:'ENTRY'}),cSeries.createPriceLine({price:cLines.sl,color:'#f43f5e',lineWidth:1,lineStyle:2,title:'SL'}),cSeries.createPriceLine({price:cLines.tp,color:'#10b981',lineWidth:1,lineStyle:2,title:'TP'}));}
function setTF(tf,btn){curTF=tf;document.querySelectorAll('.ctab').forEach(b=>b.classList.remove('on'));btn.classList.add('on');renderChart();}

// ─── Terminal ──────────────────────────────────────────────────────────────
function termLog(msg){FEEDS.forEach(id=>{const el=$(id);if(!el)return;const dv=document.createElement('div');dv.className='ll';dv.innerHTML='<span class="ts">['+new Date().toLocaleTimeString('en-GB',{hour12:false})+']</span> '+msg;el.appendChild(dv);el.scrollTop=el.scrollHeight;while(el.children.length>400)el.removeChild(el.firstChild);});}

// ─── Params push ───────────────────────────────────────────────────────────
function pushParams(){if(!ws||ws.readyState!==WebSocket.OPEN){alert('NOT CONNECTED');return;}ws.send(JSON.stringify({action:'update_settings',threshold:$('p-thresh').value,threshold_max:$('p-thresh-max').value,risk:$('p-risk').value,rr:$('p-rr').value,be_trigger:$('p-be').value,be_trail_pct:$('p-trail').value,cooldown:$('p-cool').value,blocked_hours:$('p-blk').value,direction:$('p-dir').value}));const m=$('params-msg');m.textContent='✓ TRANSMITTED';setTimeout(()=>m.textContent='',2500);}

// ─── Pipeline replay ────────────────────────────────────────────────────────
function replayPipeline(){
  initGlobe&&initGlobe();
  fetch('/api/replay-pipeline',{method:'POST'}).then(r=>r.json()).then(d=>{
    termLog('<span class="ok">[PIPELINE] '+(d.msg||'triggered')+'</span>');
  }).catch(()=>termLog('<span class="err">[PIPELINE] request failed (is the server running?)</span>'));
}

// ─── Backtest ──────────────────────────────────────────────────────────────
function runBacktest(){
  const model=$('bt-model').value.trim();if(!model){$('bt-err').textContent='MODEL FILE REQUIRED';return;}
  $('bt-err').textContent='';
  const p={script:$('bt-script').value,in_csv:$('bt-csv').value,ohlc_csv:$('bt-ohlc').value,model,metadata:$('bt-meta').value,
    direction:$('bt-dir').value,rr:$('bt-rr').value,be_trigger:$('bt-be').value,be_trail_pct:$('bt-trail').value,
    threshold:$('bt-thresh').value,threshold_max:$('bt-tmax').value,blocked_hours:$('bt-blk').value,
    balance:$('bt-bal').value,risk_pct:$('bt-risk').value,slippage_ticks:$('bt-slip').value,
    start_date:$('bt-start').value,end_date:$('bt-end').value,anti_repaint:$('bt-ar').checked};
  fetch('/api/run-backtest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)})
  .then(r=>r.json()).then(d=>{
    if(d.status==='ok'){btTab('run');$('bt-log').innerHTML='';$('bt-pill').style.display='flex';$('btn-bt').disabled=true;$('btn-bt').textContent='■ RUNNING…';startBtPoll();}
    else $('bt-err').textContent=d.msg||'FAILED';
  }).catch(e=>$('bt-err').textContent='NET ERROR: '+e);
}
function startBtPoll(){
  if(btPoll)clearInterval(btPoll);
  btPoll=setInterval(()=>{
    fetch('/api/backtest-status').then(r=>r.json()).then(s=>{
      if($('bt-pbar'))$('bt-pbar').style.width=s.progress+'%';
      if($('bt-pct'))$('bt-pct').textContent=s.progress+'%';
      if($('bt-stxt'))$('bt-stxt').textContent=s.status.toUpperCase();
      const log=$('bt-log');if(log&&s.log){log.innerHTML='';s.log.forEach(l=>{const d=document.createElement('div');d.textContent=l;log.appendChild(d);});log.scrollTop=log.scrollHeight;}
      if(!s.running){clearInterval(btPoll);btPoll=null;$('bt-pill').style.display='none';const b=$('btn-bt');b.disabled=false;b.textContent='⟶ LAUNCH BACKTEST';
        if(s.status==='done'){fetch('/api/backtest-data').then(r=>r.json()).then(d=>{if(d.status==='ok')renderBtReport(d.data);});}}
    });
  },1500);
}
function renderBtReport(data){
  btTab('report');$('bt-no-data').style.display='none';const res=$('bt-results');res.style.display='flex';
  const S=data.stats;
  const OCC=['rgba(6,182,212,.85)','rgba(16,185,129,.85)','rgba(244,63,94,.85)','rgba(245,158,11,.85)'];
  const OCL=['Trail','Win','Loss','BE'];const OCV=OCL.map(l=>(data.outcomes||{})[l]||0);
  const h=$('bt-heroes');h.innerHTML='';
  [{lbl:'Net Profit',v:'+$'+fN(S.net),c:'var(--acid)',bg:'rgba(16,185,129,.1)',bc:'rgba(16,185,129,.28)'},
   {lbl:'Final Balance',v:'$'+fN(S.end_bal),c:'var(--plasma)',bg:'rgba(6,182,212,.1)',bc:'rgba(6,182,212,.28)'},
   {lbl:'Profit Factor',v:S.pf.toFixed(3),c:'var(--ion)',bg:'rgba(59,130,246,.1)',bc:'rgba(59,130,246,.28)'},
   {lbl:'Max Drawdown',v:S.max_dd.toFixed(2)+'%',c:'var(--hot)',bg:'rgba(244,63,94,.1)',bc:'rgba(244,63,94,.25)'},
  ].forEach(x=>{const e=document.createElement('div');e.className='phero';e.style.cssText=`background:linear-gradient(135deg,${x.bg},transparent);border-color:${x.bc};color:${x.c}`;e.innerHTML=`<div class="phl">${x.lbl}</div><div class="phv">${x.v}</div>`;h.appendChild(e);});
  const ti=$('bt-tiles');ti.innerHTML='';
  [{lbl:'WIN RATE',v:S.wr.toFixed(2)+'%',c:'var(--plasma)'},{lbl:'SHARPE',v:S.sharpe.toFixed(2),c:'var(--ion)'},
   {lbl:'EXPECTANCY',v:'$'+S.exp.toFixed(2),c:'var(--amber)'},{lbl:'TRADES',v:S.total,c:'var(--violet)'},
   {lbl:'TRAIL EXITS',v:S.trail,c:'var(--acid)'},{lbl:'RETURN',v:'+'+fN(S.ret_pct)+'%',c:'var(--plasma)'},
  ].forEach(x=>{const e=document.createElement('div');e.className='mtile';e.style.setProperty('--tc',x.c);e.innerHTML=`<div class="lbl">${x.lbl}</div><div class="val" style="color:${x.c}">${x.v}</div>`;ti.appendChild(e);});
  mkChart('br-equity',{type:'line',data:{labels:data.equity.map((_,i)=>i%50===0?data.equity[i].t:''),datasets:[{data:data.equity.map(e=>e.b),borderColor:'#06b6d4',backgroundColor:'rgba(6,182,212,.07)',borderWidth:1.5,fill:true,tension:.42,pointRadius:0}]},options:{...C,scales:{x:{...C.scales.x,ticks:{...C.scales.x.ticks,maxTicksLimit:8,callback:(v,i)=>(data.equity[i]&&i%50===0)?data.equity[i].t:''}},y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>'$'+fN(v)}}}}});
  mkDonut('br-donut',OCL,OCV,OCC);buildLeg('br-donut-leg',OCL,OCV,OCC);
  if(data.drawdown)mkChart('br-dd',{type:'line',data:{labels:data.drawdown.map((_,i)=>i%50===0?data.drawdown[i].t:''),datasets:[{data:data.drawdown.map(d=>d.dd),borderColor:'#f43f5e',backgroundColor:'rgba(244,63,94,.07)',borderWidth:1.5,fill:true,tension:.3,pointRadius:0}]},options:{...C,scales:{x:{...C.scales.x,ticks:{...C.scales.x.ticks,maxTicksLimit:5,callback:(v,i)=>(data.drawdown[i]&&i%50===0)?data.drawdown[i].t:''}},y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>v+'%'}}}}});
  if(data.hourly){const h2=data.hourly,blk=new Set([3,4,5,7,9,10,14,19,20,22]);mkChart('br-hourly',{type:'bar',data:{labels:h2.map(r=>'H'+r.hour),datasets:[{data:h2.map(r=>r.pnl),backgroundColor:h2.map(r=>blk.has(r.hour)?'rgba(244,63,94,.35)':'rgba(6,182,212,.55)'),borderRadius:2,borderSkipped:false}]},options:{...C,scales:{x:{...C.scales.x},y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>'$'+fN(v)}}}}});}
  if(data.prob){const p2=(data.prob||[]).filter(x=>x.count>0);mkChart('br-prob',{type:'bar',data:{labels:p2.map(x=>x.label||x.pb_bin||''),datasets:[{data:p2.map(x=>x.wr),backgroundColor:'rgba(6,182,212,.6)',borderRadius:2,borderSkipped:false}]},options:{...C,scales:{x:{...C.scales.x},y:{...C.scales.y,min:80,max:100,ticks:{...C.scales.y.ticks,callback:v=>v+'%'}}}}});}
  if(data.monthly){const m2=data.monthly;mkChart('br-monthly',{type:'bar',data:{labels:m2.map(x=>x.month),datasets:[{data:m2.map(x=>x.pnl),backgroundColor:m2.map(x=>x.pnl>=0?'rgba(16,185,129,.6)':'rgba(244,63,94,.55)'),borderRadius:2,borderSkipped:false,yAxisID:'y'},{data:m2.map(x=>x.wr),type:'line',borderColor:'#3b82f6',borderWidth:1.5,pointRadius:2,tension:.3,yAxisID:'y2',fill:false,pointBackgroundColor:'#3b82f6'}]},options:{...C,scales:{y:{...C.scales.y,ticks:{...C.scales.y.ticks,callback:v=>'$'+fN(v)}},y2:{position:'right',grid:{display:false},ticks:{color:'#1e3450',font:{size:8},callback:v=>v+'%'}},x:{...C.scales.x}}}});}
  const tb=$('br-tbody');if(tb){tb.innerHTML='';const trades=[...(data.trades||[])].reverse().slice(0,30);trades.forEach((t,i)=>{const pnl=t.pnl||0;const tr=document.createElement('tr');tr.innerHTML=`<td style="color:var(--text)">${trades.length-i}</td><td style="color:var(--text2)">${(t.entry_str||'').slice(0,14)}</td><td><span class="badge b${(t.outcome||'L')[0]}">${t.outcome||'?'}</span></td><td style="color:var(--plasma)">${((t.ai_prob||0)*100).toFixed(1)}%</td><td style="color:var(--text3)">${(t.lot_size||0).toFixed(2)}</td><td style="color:${pnl>=0?'var(--acid)':'var(--hot)'}">${pnl>=0?'+':''}$${fN(Math.abs(pnl))}</td>`;tb.appendChild(tr);});}
}

// ════════════════════════════════════════════════════════════════════════════
// GLOBE FX ENGINE — 3D animations driven by REAL pipeline stage events.
// Every effect corresponds to an actual step of the model's decision cycle:
//   uplink   → MT5 bars streaming in     (3 particle streams spiral into globe)
//   parse    → SMC structure scan        (latitude scan-ring sweeps the sphere)
//   merge    → multi-TF fusion           (M5/M15/H4 chart planes — drawn from
//                                         the REAL last 24 candles — emerge,
//                                         orbit, spin together and are absorbed
//                                         into the core: the model feeding)
//   sanitize → anti-repaint shift        (integrity shield shell shimmer)
//   infer    → XGBoost forward pass      (neural lightning arcs charge the core)
//   verdict  → decision                  (hologram ejects from the globe with
//                                         probability + BUY / SELL / SKIP)
// ════════════════════════════════════════════════════════════════════════════
const GlobeFX=(function(){
const STAGES=['uplink','parse','merge','sanitize','infer','verdict'];
const STAGE_LABEL={uplink:'UPLINK',parse:'STRUCT SCAN',merge:'TF FUSION',sanitize:'SHIELD',infer:'INFERENCE',verdict:'VERDICT'};
const ease=x=>x<.5?2*x*x:1-Math.pow(-2*x+2,2)/2;
let queue=[],busy=false;
const fxObjs=[];
let fxPrev=performance.now();
(function fxLoop(){
  requestAnimationFrame(fxLoop);
  const now=performance.now(),dt=Math.min(.1,(now-fxPrev)/1000);fxPrev=now;
  for(let i=fxObjs.length-1;i>=0;i--){
    const o=fxObjs[i];o.t+=dt;
    let alive=false;
    try{alive=o.update(o.t,dt);}catch(e){alive=false;}
    if(!alive){try{o.dispose&&o.dispose();}catch(e){}fxObjs.splice(i,1);}
  }
})();
function addFx(update,dispose){fxObjs.push({t:0,update,dispose});}
function rmObj(obj){
  if(!globeScene||!obj)return;
  globeScene.remove(obj);
  const kill=c=>{if(c.geometry&&c.geometry.dispose)c.geometry.dispose();
    if(c.material){if(c.material.map&&c.material.map.dispose)c.material.map.dispose();
      if(c.material.dispose)c.material.dispose();}};
  if(obj.traverse)obj.traverse(kill);else kill(obj);
}

// ── HUD stage chips ──
function chips(stage){
  const idx=STAGES.indexOf(stage);
  STAGES.forEach((st,i)=>{
    const el=$('pst-'+st);if(!el)return;
    el.classList.remove('active','done');
    if(i<idx)el.classList.add('done');
    else if(i===idx)el.classList.add('active');
  });
  const ht=$('hdr-tick');if(ht)ht.textContent=STAGE_LABEL[stage]||stage.toUpperCase();
}

// ── Canvas texture helpers ──
function texCanvas(w,h,draw){
  const cv=document.createElement('canvas');cv.width=w;cv.height=h;
  draw(cv.getContext('2d'),w,h);
  const t=new THREE.CanvasTexture(cv);t.needsUpdate=true;return t;
}
function candleTex(tf){
  return texCanvas(256,160,(ctx,w,h)=>{
    ctx.fillStyle='rgba(3,12,26,.88)';ctx.fillRect(0,0,w,h);
    ctx.strokeStyle='rgba(6,182,212,.85)';ctx.lineWidth=2;ctx.strokeRect(1,1,w-2,h-2);
    ctx.fillStyle='rgba(6,182,212,.9)';[[0,0],[w-22,0],[0,h-6],[w-22,h-6]].forEach(q=>ctx.fillRect(q[0],q[1],22,6));
    ctx.fillStyle='#22d3ee';ctx.font='bold 20px Orbitron, monospace';
    ctx.shadowColor='#22d3ee';ctx.shadowBlur=10;ctx.fillText(tf,12,30);ctx.shadowBlur=0;
    const data=(typeof cOHLC!=='undefined'&&cOHLC[tf]&&cOHLC[tf].length>25)?cOHLC[tf].slice(-24):null;
    let candles;
    if(data){
      const lo=Math.min.apply(null,data.map(c=>c.low)),hi=Math.max.apply(null,data.map(c=>c.high));
      const sy=v=>h-14-((v-lo)/((hi-lo)||1))*(h-56);
      candles=data.map((c,i)=>({x:12+i*(w-26)/24,o:sy(c.open),c:sy(c.close),h:sy(c.high),l:sy(c.low)}));
    }else{
      let pv=h*.55;
      candles=Array.from({length:24},(_,i)=>{
        const o=pv;pv+=(Math.random()-.5)*22;const c2=pv;
        return {x:12+i*(w-26)/24,o:o,c:c2,h:Math.min(o,c2)-Math.random()*9,l:Math.max(o,c2)+Math.random()*9};
      });
    }
    candles.forEach(k=>{
      const up=k.c<=k.o;ctx.strokeStyle=ctx.fillStyle=up?'#10b981':'#f43f5e';
      ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(k.x+3.5,k.h);ctx.lineTo(k.x+3.5,k.l);ctx.stroke();
      ctx.fillRect(k.x,Math.min(k.o,k.c),7,Math.max(2,Math.abs(k.c-k.o)));
    });
  });
}
function holoTex(lines){
  return texCanvas(512,256,(ctx,w,h)=>{
    ctx.fillStyle='rgba(3,12,28,.8)';ctx.fillRect(0,0,w,h);
    ctx.strokeStyle='rgba(6,182,212,.9)';ctx.lineWidth=3;ctx.strokeRect(2,2,w-4,h-4);
    ctx.fillStyle='rgba(6,182,212,.9)';[[0,0],[w-28,0],[0,h-8],[w-28,h-8]].forEach(q=>ctx.fillRect(q[0],q[1],28,8));
    for(let y2=10;y2<h;y2+=6){ctx.fillStyle='rgba(6,182,212,.04)';ctx.fillRect(4,y2,w-8,2);}
    let y=6;
    lines.forEach(L=>{
      y+=L.size+16;
      ctx.font='bold '+L.size+'px Orbitron, monospace';
      ctx.fillStyle=L.color;ctx.shadowColor=L.color;ctx.shadowBlur=18;
      const tw=ctx.measureText(L.txt).width;
      ctx.fillText(L.txt,(w-tw)/2,y);ctx.shadowBlur=0;
    });
  });
}

// ── Orb-aware pulse: glow flash + particle detonation + surface ripple ──
function orbC(){return (window._orb&&window._orb.position)?window._orb.position.clone():new THREE.Vector3(0,0,0);}
function rippleOrb(){
  if(window._orbWave){
    window._orbWave.t=0;
    window._orbWave.dir.set(Math.random()-.5,Math.random()-.5,Math.random()-.5).normalize();
  }
}
function burst(hex,count,speed){
  if(!globeScene)return;
  const oc=orbC(),N=count||160;
  const pos=new Float32Array(N*3),vel=new Float32Array(N*3);
  for(let i=0;i<N;i++){
    let vx=Math.random()-.5,vy=Math.random()-.5,vz=Math.random()-.5;
    const L=Math.sqrt(vx*vx+vy*vy+vz*vz)||1,sp2=(speed||.9)*(.4+Math.random());
    vel[i*3]=vx/L*sp2;vel[i*3+1]=vy/L*sp2;vel[i*3+2]=vz/L*sp2;
    pos[i*3]=oc.x;pos[i*3+1]=oc.y;pos[i*3+2]=oc.z;
  }
  const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  const mat=new THREE.PointsMaterial({color:hex,size:.02,transparent:true,opacity:1,blending:THREE.AdditiveBlending,depthWrite:false});
  const pts=new THREE.Points(geo,mat);globeScene.add(pts);
  addFx((t,dt)=>{
    const T=1.1;
    for(let i=0;i<N;i++){
      pos[i*3]+=vel[i*3]*dt;pos[i*3+1]+=vel[i*3+1]*dt;pos[i*3+2]+=vel[i*3+2]*dt;
      vel[i*3]*=.985;vel[i*3+1]*=.985;vel[i*3+2]*=.985;
    }
    geo.attributes.position.needsUpdate=true;
    mat.opacity=Math.max(0,1-t/T);
    return t<T;
  },()=>rmObj(pts));
}
function pulseCore(hex,mag){
  if(!globeScene)return;
  const oc=orbC();
  const mat=new THREE.MeshBasicMaterial({color:hex,transparent:true,opacity:.5});
  const sp=new THREE.Mesh(new THREE.SphereGeometry(.22,18,18),mat);
  sp.position.copy(oc);globeScene.add(sp);
  addFx((t)=>{
    const T=.9;
    sp.scale.setScalar(1+(mag||1.5)*ease(Math.min(1,t/T)));
    mat.opacity=Math.max(0,.5*(1-t/T));
    return t<T;
  },()=>rmObj(sp));
  burst(hex,140,(mag||1.5)*.55);
  rippleOrb();
}

// ── STAGE FX ──
function fxUplink(detail){
  if(!globeScene)return 900;
  const cols=[0x22d3ee,0x3b82f6,0x8b5cf6];
  cols.forEach((c,k)=>{
    const N=26;
    const pos=new Float32Array(N*3);
    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
    const mat=new THREE.PointsMaterial({color:c,size:.04,transparent:true,opacity:.95,blending:THREE.AdditiveBlending,depthWrite:false});
    const pts=new THREE.Points(geo,mat);globeScene.add(pts);
    const phi0=Math.PI*2*k/3;
    addFx((t)=>{
      const T=1.7;
      const oc=orbC();
      for(let i=0;i<N;i++){
        const sN=Math.max(0,Math.min(1,t/T-i*.02));
        const r=3.3-2.95*sN;
        const a=phi0+sN*6+i*.14;
        pos[i*3]=oc.x*sN+r*Math.cos(a);
        pos[i*3+1]=oc.y*sN+(1-sN)*(k-1)*.85+Math.sin(a*2)*.12*(1-sN);
        pos[i*3+2]=oc.z*sN+r*Math.sin(a);
      }
      geo.attributes.position.needsUpdate=true;
      mat.opacity=t<T?.95:Math.max(0,.95-(t-T)*3.5);
      return t<T+.3;
    },()=>rmObj(pts));
  });
  pulseCore(0x22d3ee,1.3);
  return 1900;
}

function fxParse(){
  if(!globeScene)return 900;
  const mat=new THREE.MeshBasicMaterial({color:0x22d3ee,transparent:true,opacity:.85});
  const ring=new THREE.Mesh(new THREE.TorusGeometry(1.01,.008,8,120),mat);
  ring.rotation.x=Math.PI/2;globeScene.add(ring);
  addFx((t)=>{
    const T=1.8,sN=Math.min(1,t/T);
    const lat=Math.PI*(sN-.5);
    ring.position.y=Math.sin(lat);
    const r=Math.max(.03,Math.cos(lat));
    ring.scale.set(r,r,1);
    mat.opacity=(t<T?.85:Math.max(0,.85-(t-T)*4))*(1-Math.abs(2*sN-1)*.2);
    return t<T+.25;
  },()=>rmObj(ring));
  return 1950;
}

function fxMerge(){
  if(!globeScene)return 1200;
  const tfs=['M5','M15','H4'];
  const group=new THREE.Group();group.position.copy(orbC());globeScene.add(group);
  const planes=tfs.map(tf=>{
    const m=new THREE.Mesh(
      new THREE.PlaneGeometry(.92,.58),
      new THREE.MeshBasicMaterial({map:candleTex(tf),transparent:true,opacity:0,side:THREE.DoubleSide}));
    group.add(m);return m;
  });
  let flashed=false;
  addFx((t)=>{
    const T1=1.0,T2=1.15,T3=1.05,T=T1+T2+T3;
    planes.forEach((m,k)=>{
      const base=Math.PI*2*k/3;
      let r,ang,sc,op,y;
      if(t<T1){const sN=ease(t/T1);r=1.25*sN;ang=base+sN*1.2;sc=sN;op=sN;y=.24*Math.sin(base);}
      else if(t<T1+T2){const sN=(t-T1)/T2;r=1.25;ang=base+1.2+sN*3.2;sc=1;op=1;y=.24*Math.sin(base+sN*7);}
      else{const sN=ease((t-T1-T2)/T3);r=1.25*(1-sN);ang=base+4.4+sN*10;sc=Math.max(.02,1-sN*.95);op=1-sN;y=.24*Math.sin(base)*(1-sN);}
      m.position.set(r*Math.cos(ang),y,r*Math.sin(ang));
      m.scale.setScalar(Math.max(.001,sc));
      m.material.opacity=op;
      m.lookAt(group.position);
      m.rotateY(Math.PI);
    });
    if(t>=T1+T2+T3*.8&&!flashed){flashed=true;pulseCore(0x22d3ee,2.4);}
    return t<T1+T2+T3;
  },()=>rmObj(group));
  return 3450;
}

function fxSanitize(){
  if(!globeScene)return 800;
  const mat=new THREE.MeshBasicMaterial({color:0x10b981,wireframe:true,transparent:true,opacity:0});
  const shell=new THREE.Mesh(new THREE.IcosahedronGeometry(.62,1),mat);
  shell.position.copy(orbC());
  globeScene.add(shell);
  addFx((t)=>{
    const T=1.4;
    shell.rotation.y+=.02;shell.rotation.x+=.008;
    if(t<.35)mat.opacity=t/.35*.5;
    else if(t<T)mat.opacity=.5-((t-.35)/(T-.35))*.2;
    else mat.opacity=Math.max(0,.3-(t-T)*2);
    return t<T+.2;
  },()=>rmObj(shell));
  return 1550;
}

function fxInfer(){
  if(!globeScene)return 1000;
  const arcs=[];
  for(let i=0;i<7;i++){
    const oc=orbC();
    const dir=new THREE.Vector3().setFromSphericalCoords(1,Math.random()*Math.PI,Math.random()*Math.PI*2);
    const pts=[];const N=9;
    for(let j=0;j<=N;j++){
      const sN=j/N;
      const pt=dir.clone().lerp(oc,sN);
      if(j>0&&j<N){pt.x+=(Math.random()-.5)*.15;pt.y+=(Math.random()-.5)*.15;pt.z+=(Math.random()-.5)*.15;}
      pts.push(pt);
    }
    const geo=new THREE.BufferGeometry().setFromPoints(pts);
    const mat=new THREE.LineBasicMaterial({color:0x8b5cf6,transparent:true,opacity:0});
    const ln=new THREE.Line(geo,mat);globeScene.add(ln);
    arcs.push({ln:ln,mat:mat,delay:i*.15});
  }
  addFx((t)=>{
    let alive=false;
    arcs.forEach(a=>{
      const lt=t-a.delay;
      if(lt<0){alive=true;return;}
      const T=.5;
      a.mat.opacity=lt<T?Math.sin(Math.PI*Math.min(1,lt/T)):0;
      if(lt<T)alive=true;
    });
    return alive;
  },()=>arcs.forEach(a=>rmObj(a.ln)));
  pulseCore(0x8b5cf6,1.9);
  return 1700;
}

function fxVerdict(detail){
  detail=detail||{};
  const act=String(detail.action||'SKIP').toUpperCase();
  const probPct=((detail.prob||0)*100).toFixed(1)+'%';
  const isBuy=act==='BUY',isSell=act==='SELL';
  const col=isBuy?'#10b981':isSell?'#f43f5e':'#f59e0b';
  const hex=isBuy?0x10b981:isSell?0xf43f5e:0xf59e0b;
  // DOM banner — visible even if the 3D scene is not initialised
  const vb=$('verdict-banner');
  if(vb){
    let inner='<span style="color:'+col+'">'+act+'</span> <span style="color:#c8e0f0">'+probPct+'</span>';
    if(detail.reason)inner+='<div style="font-size:.45em;color:#5a7898;letter-spacing:.2em">'+String(detail.reason).toUpperCase()+'</div>';
    else if(detail.entry)inner+='<div style="font-size:.45em;color:#5a7898;letter-spacing:.2em">ENTRY '+Number(detail.entry).toFixed(5)+' · LOT '+(detail.lot||0)+'</div>';
    vb.innerHTML=inner;
    vb.style.borderColor=col;vb.style.boxShadow='0 0 30px '+col+'55';
    vb.classList.add('show');
    setTimeout(()=>vb.classList.remove('show'),3200);
  }
  if(!globeScene)return 1500;
  const lines=[{txt:act,color:col,size:74},{txt:'P = '+probPct,color:'#c8e0f0',size:40}];
  if(detail.reason)lines.push({txt:String(detail.reason).toUpperCase().slice(0,22),color:'#8ab0cc',size:22});
  else if(detail.entry)lines.push({txt:'@ '+Number(detail.entry).toFixed(5),color:'#8ab0cc',size:22});
  const oc=orbC();
  const spr=new THREE.Sprite(new THREE.SpriteMaterial({map:holoTex(lines),transparent:true,opacity:0}));
  spr.scale.set(.001,.001,1);globeScene.add(spr);
  const rmat=new THREE.MeshBasicMaterial({color:hex,transparent:true,opacity:.8,side:THREE.DoubleSide});
  const ring=new THREE.Mesh(new THREE.RingGeometry(.60,.64,90),rmat);
  ring.rotation.x=Math.PI/2;ring.position.copy(oc);globeScene.add(ring);
  addFx((t)=>{
    const T1=.7,Th=2.3,T=T1+Th+.6;
    if(t<T1){const sN=ease(t/T1);spr.position.set(oc.x,oc.y+.2+1.15*sN,oc.z);spr.scale.set(1.5*sN,.75*sN,1);spr.material.opacity=sN;}
    else if(t<T1+Th){spr.position.y=oc.y+1.35+.05*Math.sin((t-T1)*3);spr.material.opacity=1;}
    else{const sN=(t-T1-Th)/.6;spr.material.opacity=Math.max(0,1-sN);spr.position.y=oc.y+1.35+.5*sN;}
    const rs=1+Math.min(2.4,t*1.7);ring.scale.set(rs,rs,1);
    rmat.opacity=Math.max(0,.8-t*.45);
    return t<T;
  },()=>{rmObj(spr);rmObj(ring);});
  pulseCore(hex,2.6);
  return 3700;
}

const FX={uplink:fxUplink,parse:fxParse,merge:fxMerge,sanitize:fxSanitize,infer:fxInfer,verdict:fxVerdict};

function onStage(d){
  chips(d.stage);
  if(d.stage==='verdict'){
    const det=d.detail||{};
    const c2=det.action==='BUY'?'#10b981':det.action==='SELL'?'#f43f5e':'#f59e0b';
    termLog('<span style="color:'+c2+'">[VERDICT] '+(det.action||'SKIP')+
      (det.reason?' · '+det.reason:'')+' · P='+(((det.prob||0)*100).toFixed(1))+'%</span>');
  }
  queue.push(d);
  if(queue.length>8)queue.splice(0,queue.length-8);
  if(!busy)next();
}
function next(){
  if(!queue.length){busy=false;return;}
  busy=true;
  const d=queue.shift();
  let dur=600;
  try{const fn=FX[d.stage];if(fn)dur=fn(d.detail||{})||600;}catch(e){dur=600;}
  setTimeout(next,dur);
}
return {onStage:onStage};
})();

</script>
</body>
</html>"""

if __name__ == "__main__":
    print("=" * 62)
    print("  NEXUS AI TERMINAL v7.0 — GLOBE FX EDITION")
    print("  http://localhost:8000")
    print("  NEW: Live pipeline-stage 3D animations on the Globe tab")
    print("       (uplink / struct scan / TF fusion / shield / inference / verdict)")
    print("  NEW: Live Outcome Matrix donut + Confidence->WR chart (from MT5 deals)")
    print("  NEW: Pending-order status wired to Config panel")
    print("=" * 62)
    uvicorn.run(app, host="0.0.0.0", port=8000)