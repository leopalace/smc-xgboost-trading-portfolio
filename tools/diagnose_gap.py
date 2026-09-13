#!/usr/bin/env python3
"""
diagnose_gap.py — Compare labeler labels vs backtester outcomes.
Identifies exactly where the backtester diverges from the labeler's verdicts.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import gc
import sys

sys.path.insert(0, ".")
from preprocessing import create_category_mappings, preprocess_for_model

TARGET = "label_1R_50pb_win"
THRESHOLD = 0.65

# ═══════════════════════════════════════════════════════════════
# 1. Load OOS data + model, get predictions
# ═══════════════════════════════════════════════════════════════
print("[*] Loading OOS data...")
chunks = []
for chunk in pd.read_csv("ml_labeled_v17_OOS.csv", low_memory=False, chunksize=50000):
    valid = chunk[chunk[TARGET] != -2]
    if len(valid) > 0:
        chunks.append(valid)
oos = pd.concat(chunks, ignore_index=True)
del chunks; gc.collect()

print(f"OOS valid setups: {len(oos)}")

# Load production model
with open("model_1R_50pb_v17r_metadata.json") as f:
    meta = json.load(f)

features = meta["features"]
cat_maps = meta.get("category_mappings", {})

bst = xgb.Booster()
bst.load_model("model_1R_50pb_v17r.json")

# Preprocess + predict
X, proc = preprocess_for_model(oos.copy(), features, cat_maps, None)
proba = bst.predict(xgb.DMatrix(X))

proc["ai_prob"] = proba
proc["label"] = proc[TARGET].astype(int)

# ═══════════════════════════════════════════════════════════════
# 2. What does the model select at 0.65 threshold?
# ═══════════════════════════════════════════════════════════════
selected = proc[proc["ai_prob"] >= THRESHOLD].copy()

print(f"\n[*] Model selected {len(selected)} trades at >= {THRESHOLD*100:.0f}% threshold")
print(f"    Labeler says: {(selected['label']==1).sum()} wins, {(selected['label']==0).sum()} losses")
label_wr = (selected['label']==1).mean() * 100
print(f"    Label-based WR: {label_wr:.1f}%")

# ═══════════════════════════════════════════════════════════════
# 3. Now load backtester results to compare
# ═══════════════════════════════════════════════════════════════
try:
    bt = pd.read_csv("backtest_decisions_be.csv")
    print(f"\n[*] Backtester results: {len(bt)} trades")
    print(f"    Outcomes: {bt['outcome'].value_counts().to_dict()}")
    bt_wr = bt['outcome'].isin(['Win','Trail']).mean() * 100
    print(f"    Backtester WR: {bt_wr:.1f}%")

    # Check spread impact
    print(f"\n[*] Spread analysis:")
    print(f"    Avg spread: {bt['spread_pts'].mean():.2f} pts")
    print(f"    Max spread: {bt['spread_pts'].max():.2f} pts")
    print(f"    Min spread: {bt['spread_pts'].min():.2f} pts")

    # Compare entry vs SL distance vs spread
    bt['sl_dist'] = abs(bt['entry_price'] - bt['sl_price'])
    bt['tp_dist'] = abs(bt['tp_price'] - bt['entry_price'])
    bt['spread_pct_of_risk'] = bt['spread_pts'] / bt['sl_dist'] * 100

    print(f"\n[*] Spread as % of risk (SL distance):")
    print(f"    Mean: {bt['spread_pct_of_risk'].mean():.1f}%")
    print(f"    Median: {bt['spread_pct_of_risk'].median():.1f}%")
    print(f"    Max: {bt['spread_pct_of_risk'].max():.1f}%")

    # How many wins flipped?
    print(f"\n[*] Gap analysis:")
    print(f"    Label-based WR:     {label_wr:.1f}% on {len(selected)} trades")
    print(f"    Backtester WR:      {bt_wr:.1f}% on {len(bt)} trades")
    print(f"    Gap:                {label_wr - bt_wr:.1f}%")
    print(f"    Trades lost to filters: {len(selected) - len(bt)}")

    # BE analysis
    be_trades = bt[bt['outcome'] == 'BE']
    print(f"\n[*] BE analysis: {len(be_trades)} breakevens")
    if len(be_trades) > 0:
        print(f"    These contribute 0 PnL but reduce trade count for WR calculation")
        print(f"    WR excluding BEs: {bt[bt['outcome'].isin(['Win','Trail','Loss'])]['outcome'].isin(['Win','Trail']).mean()*100:.1f}%")

except FileNotFoundError:
    print("\n[!] backtest_decisions_be.csv not found — run backtester first")

# ═══════════════════════════════════════════════════════════════
# 4. Quantify: what does spread actually cost in R terms?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SPREAD IMPACT ANALYSIS")
print(f"{'='*60}")

# For each selected trade, compute how spread changes the R:R
# Labeler: entry=close, SL=structural, TP=close + RR*(close-SL)
# Backtester BUY: entry=close+spread, same SL. Risk is BIGGER, TP is FURTHER.
# Backtester SELL: entry=close, but SL/TP detection adds spread to highs/lows.

# Load a sample of OHLC for spread values
ohlc_chunks = []
for chunk in pd.read_csv("historical_data/Volatility 75 (1s) Index_M15.csv", low_memory=False, chunksize=50000, nrows=100000):
    if 'spread' in [c.lower().replace('<','').replace('>','').strip() for c in chunk.columns]:
        chunk.columns = [c.lower().replace('<','').replace('>','').strip() for c in chunk.columns]
        ohlc_chunks.append(chunk[['spread']].head(10000))
        break

if ohlc_chunks:
    spreads = pd.concat(ohlc_chunks)['spread']
    avg_spread_ticks = spreads.mean()
    avg_spread_pts = avg_spread_ticks * 0.01
    print(f"\nAverage OHLC spread: {avg_spread_ticks:.0f} ticks = {avg_spread_pts:.2f} pts")

    # For a typical V75 1s trade with ~100pt SL distance
    typical_sl = 100  # example
    spread_cost_pct = avg_spread_pts / typical_sl * 100
    print(f"Spread as % of typical {typical_sl}pt SL: {spread_cost_pct:.1f}%")

    # For 1R trade: effective RR after spread
    # BUY: effective_risk = SL_dist + spread (entry is worse)
    # effective_reward = TP_dist - spread (TP relative to BID is worse)
    # Actually no — in the backtester, TP = entry + risk*RR where risk already includes spread
    # So TP distance from entry = risk*RR = risk. But risk > labeler's risk by spread.
    # And the SL detection doesn't add spread for buys (line 665: low <= sl_price)
    # So for BUYS: effective RR from labeler's perspective:
    #   Labeler risk = close - SL
    #   Backtester risk = (close+spread) - SL = labeler_risk + spread
    #   Backtester reward = backtester_risk * RR = (labeler_risk + spread) * 1.0
    #   From labeler's POV: backtester needs to reach close + 2*(labeler_risk + spread)
    #   vs labeler only needs close + 2*labeler_risk
    print(f"\nFor a BUY with {typical_sl}pt labeler SL distance and {avg_spread_pts:.1f}pt spread:")
    print(f"  Labeler risk:    {typical_sl} pts")
    print(f"  Backtester risk: {typical_sl + avg_spread_pts:.1f} pts (+{avg_spread_pts/typical_sl*100:.1f}%)")
    print(f"  Labeler TP dist:    {typical_sl} pts (1R)")
    print(f"  Backtester TP dist: {typical_sl + avg_spread_pts:.1f} pts (1R from entry)")
    effective_rr = typical_sl / (typical_sl + avg_spread_pts)
    print(f"  Effective RR (labeler terms): {effective_rr:.3f}R instead of 1.0R")
else:
    print("Could not load OHLC for spread analysis")

print(f"\n{'='*60}")
print("CONCLUSION")
print(f"{'='*60}")
print("""
The gap between label WR and backtester WR comes from:
1. SPREAD: Buys enter at ASK (close+spread), making risk larger and TP further.
   Sells have spread added to hit detection, making SL closer and TP further.
   The labeler ignores spread entirely.

2. BE FLIPS: Trades labeled as wins by the labeler become BEs in the backtester
   because the spread shifts the BE trigger and entry point. After BE trigger,
   the trail SL is at entry (not close), so price retracement to entry causes
   a BE exit that the labeler wouldn't see.

3. FILTER REJECTIONS: sl_too_tight/sl_too_wide remove trades that the label
   test counts as wins, biasing the sample.

FIX: Add spread to the LABELER so labels reflect realistic execution costs.
""")
