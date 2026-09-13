#!/usr/bin/env python3
"""
check_lookahead2.py — Check M5 rows per M15 bar in the FULL dataset.
The labeler uses highs[i+1:end_idx] on the full 525K array.
If i+1 is still in the same M15 bar, the labeler sees the entry bar's
high/low as "future" data — this is a lookahead bias.
"""
import pandas as pd
import numpy as np
import gc

print("[*] Loading full labeled CSV (first 10000 rows for speed)...")
df = pd.read_csv("ml_dataset_labeled_v17.csv", low_memory=False, nrows=10000)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Rows loaded: {len(df)}")
print(f"Timestamp range: {df.timestamp.min()} to {df.timestamp.max()}")

# Check M15 bar membership
df['m15_bar'] = df['timestamp'].dt.floor('15min')
df['minute_in_bar'] = (df['timestamp'] - df['m15_bar']).dt.total_seconds() / 60

print(f"\nMinute distribution within M15 bar:")
print(df['minute_in_bar'].value_counts().sort_index().head(10).to_string())

# M5 rows per M15 bar
bar_counts = df.groupby('m15_bar').size()
print(f"\nM5 rows per M15 bar:")
print(f"  Mean: {bar_counts.mean():.1f}")
print(f"  Median: {bar_counts.median():.0f}")
print(f"  Min: {bar_counts.min()}")
print(f"  Max: {bar_counts.max()}")
print(f"  Distribution: {bar_counts.value_counts().sort_index().to_dict()}")

# For a trigger row (valid setup), what's i+1?
TARGET = "label_1R_50pb_win"
triggers = df[df[TARGET].isin([0, 1])].copy()
print(f"\nTrigger rows in first 10K: {len(triggers)}")

for idx in triggers.index[:5]:
    row = df.loc[idx]
    next_row = df.loc[idx + 1] if idx + 1 < len(df) else None

    print(f"\n  Trigger at idx={idx}:")
    print(f"    timestamp: {row['timestamp']}")
    print(f"    minute_in_bar: {row['minute_in_bar']}")
    print(f"    high: {row['high']:.2f}, low: {row['low']:.2f}, close: {row['close']:.2f}")

    if next_row is not None:
        same_bar = row['m15_bar'] == next_row['m15_bar']
        print(f"    NEXT row (i+1) timestamp: {next_row['timestamp']}")
        print(f"    NEXT minute_in_bar: {next_row['minute_in_bar']}")
        print(f"    SAME M15 bar? {'YES ← LOOKAHEAD!' if same_bar else 'No'}")
        if same_bar:
            print(f"    NEXT high: {next_row['high']:.2f} (same as entry bar? {row['high'] == next_row['high']})")
            print(f"    NEXT low: {next_row['low']:.2f} (same as entry bar? {row['low'] == next_row['low']})")

# Quantify: for all triggers, how many have i+1 in same M15 bar?
count_same = 0
count_total = 0
for idx in triggers.index:
    if idx + 1 >= len(df):
        continue
    count_total += 1
    if df.loc[idx, 'm15_bar'] == df.loc[idx + 1, 'm15_bar']:
        count_same += 1

print(f"\n=== RESULT ===")
print(f"Triggers with i+1 in SAME M15 bar: {count_same}/{count_total} ({count_same/count_total*100:.1f}%)")
print(f"These triggers give the labeler access to the entry bar's high/low as 'future' data.")
print(f"The backtester correctly starts from the NEXT M15 bar.")
