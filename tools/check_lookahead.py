#!/usr/bin/env python3
"""
check_lookahead.py — Check if labeler has lookahead bias from M5/M15 mismatch.

The state table has M5 timestamps but M15 OHLC. Multiple M5 rows share the same
M15 bar's high/low/open/close. When the labeler does `window_highs = highs[i+1:end_idx]`,
row i+1 might be in the SAME M15 bar as row i — giving the labeler access to the
entry bar's high/low as "future" data, even though that extreme may have occurred
BEFORE the close (entry price).

The backtester correctly uses: future_df = ohlc_df[timestamp > trade_time],
which starts from the NEXT M15 bar.
"""
import pandas as pd
import numpy as np
import gc

TARGET = "label_1R_50pb_win"

print("[*] Loading labeled data (valid rows only)...")
chunks = []
for chunk in pd.read_csv("ml_dataset_labeled_v17.csv", low_memory=False, chunksize=50000):
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce', utc=True)
    valid = chunk[chunk[TARGET].isin([0, 1])]
    if len(valid) > 0:
        chunks.append(valid)
df = pd.concat(chunks, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
del chunks; gc.collect()

print(f"Valid trades: {len(df)}")

# Check M5 vs M15 relationship
# M15 bars are at :00, :15, :30, :45
# M5 rows within a bar: :00/:05/:10, :15/:20/:25, :30/:35/:40, :45/:50/:55
df['m15_bar'] = df['timestamp'].dt.floor('15min')

# For each trade, check: does the NEXT row share the same M15 bar?
df['next_m15_bar'] = df['m15_bar'].shift(-1)
df['same_bar_next'] = df['m15_bar'] == df['next_m15_bar']

# How many trigger rows have at least 1 more M5 row in same M15 bar?
same_bar_count = df['same_bar_next'].sum()
print(f"\nTrigger rows with next M5 row in SAME M15 bar: {same_bar_count}/{len(df)} ({same_bar_count/len(df)*100:.1f}%)")
print("(These rows give the labeler access to the entry bar's high/low as 'future' data)")

# Check: do the high/low values repeat across M5 rows in same M15 bar?
df['next_high'] = df['high'].shift(-1)
df['next_low'] = df['low'].shift(-1)
same_ohlc = df['same_bar_next'] & (df['high'] == df['next_high']) & (df['low'] == df['next_low'])
print(f"Of those, rows where high/low is identical to next row: {same_ohlc.sum()}/{same_bar_count}")

# KEY QUESTION: For wins, could the entry bar's high (for buys) or low (for sells)
# have reached TP?
print(f"\n--- LOOKAHEAD IMPACT ON WINS ---")

# Determine direction
bias_col = "m15_external_bias" if "m15_external_bias" in df.columns else "external_bias"
df['is_bull'] = df[bias_col].astype(str).str.lower() == 'bullish'

# Get SL levels
df['sl_level'] = np.where(
    df['is_bull'],
    pd.to_numeric(df.get('m15_last_internal_low', 0), errors='coerce').fillna(0),
    pd.to_numeric(df.get('m15_last_internal_high', 0), errors='coerce').fillna(0)
)

# Compute risk, TP
df['risk'] = np.abs(df['close'] - df['sl_level'])
df['tp_level'] = np.where(
    df['is_bull'],
    df['close'] + df['risk'],  # 1R TP for buys
    df['close'] - df['risk']   # 1R TP for sells
)

# For wins: could the ENTRY BAR's extreme reach TP?
wins = df[df[TARGET] == 1].copy()

# Bull wins: did the entry bar's high >= TP?
bull_wins = wins[wins['is_bull']]
bull_entry_bar_hits = bull_wins['high'] >= bull_wins['tp_level']
print(f"\nBull wins where entry bar high >= TP: {bull_entry_bar_hits.sum()}/{len(bull_wins)} ({bull_entry_bar_hits.mean()*100:.1f}%)")

# Bear wins: did the entry bar's low <= TP?
bear_wins = wins[~wins['is_bull']]
bear_entry_bar_hits = bear_wins['low'] <= bear_wins['tp_level']
print(f"Bear wins where entry bar low <= TP: {bear_entry_bar_hits.sum()}/{len(bear_wins)} ({bear_entry_bar_hits.mean()*100:.1f}%)")

total_lookahead = bull_entry_bar_hits.sum() + bear_entry_bar_hits.sum()
total_wins = len(wins)
print(f"\nTotal wins potentially from entry-bar lookahead: {total_lookahead}/{total_wins} ({total_lookahead/total_wins*100:.1f}%)")
print(f"These 'wins' may have had their TP hit BEFORE the close (entry), not after.")

# Also check: entry bar's extreme reaching BE trigger
be_level = np.where(
    df['is_bull'],
    df['close'] + 0.5 * df['risk'],
    df['close'] - 0.5 * df['risk']
)
df['be_level'] = be_level

bull_be_bar = bull_wins['high'] >= bull_wins['close'] + 0.5 * bull_wins['risk']
bear_be_bar = bear_wins['low'] <= bear_wins['close'] - 0.5 * bear_wins['risk']
print(f"\nBull wins where entry bar reaches BE trigger: {bull_be_bar.sum()}/{len(bull_wins)}")
print(f"Bear wins where entry bar reaches BE trigger: {bear_be_bar.sum()}/{len(bear_wins)}")

# Check how many M5 rows per M15 bar in trigger rows
m15_groups = df.groupby('m15_bar').size()
print(f"\nM5 rows per M15 bar (for trigger rows):")
print(f"  Mean: {m15_groups.mean():.1f}")
print(f"  Median: {m15_groups.median():.0f}")
print(f"  Max: {m15_groups.max()}")

# Check minute distribution of triggers
df['minute_in_bar'] = df['timestamp'].dt.minute % 15
print(f"\nTrigger minute distribution within M15 bar:")
print(df['minute_in_bar'].value_counts().sort_index().to_string())
