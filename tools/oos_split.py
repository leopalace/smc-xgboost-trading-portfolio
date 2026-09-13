#!/usr/bin/env python3
"""
oos_split.py — Split labeled CSV into train (70%) and OOS (30%) by time.
Then retrain model on train split, backtest on OOS split.
"""
import pandas as pd
import numpy as np
import sys

# Only load needed columns first to check split
print("[*] Loading labeled CSV...")
df = pd.read_csv("ml_dataset_labeled_v17.csv", low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Total rows: {len(df)}")

# 70/30 split
split_idx = int(len(df) * 0.70)
split_time = df.loc[split_idx, 'timestamp']
print(f"Split at row {split_idx}, timestamp: {split_time}")

train_df = df.iloc[:split_idx].copy()
oos_df = df.iloc[split_idx:].copy()

# Stats on valid setups
target = 'label_1R_50pb_win'
train_valid = train_df[train_df[target] != -2]
oos_valid = oos_df[oos_df[target] != -2]

print(f"\n--- TRAIN (first 70%) ---")
print(f"Rows: {len(train_df)}")
print(f"Valid setups: {len(train_valid)}")
print(f"Wins: {(train_valid[target]==1).sum()}, Losses: {(train_valid[target]==0).sum()}")
tr_total = len(train_valid)
tr_wr = (train_valid[target]==1).sum() / tr_total * 100 if tr_total > 0 else 0
print(f"Base WR: {tr_wr:.2f}%")
print(f"Date range: {train_df.timestamp.min()} to {train_df.timestamp.max()}")

print(f"\n--- OOS (last 30%) ---")
print(f"Rows: {len(oos_df)}")
print(f"Valid setups: {len(oos_valid)}")
print(f"Wins: {(oos_valid[target]==1).sum()}, Losses: {(oos_valid[target]==0).sum()}")
oos_total = len(oos_valid)
oos_wr = (oos_valid[target]==1).sum() / oos_total * 100 if oos_total > 0 else 0
print(f"Base WR: {oos_wr:.2f}%")
print(f"Date range: {oos_df.timestamp.min()} to {oos_df.timestamp.max()}")

# Save splits
train_df.to_csv("ml_labeled_v17_TRAIN.csv", index=False)
oos_df.to_csv("ml_labeled_v17_OOS.csv", index=False)
print(f"\n[SUCCESS] Saved ml_labeled_v17_TRAIN.csv ({len(train_df)} rows)")
print(f"[SUCCESS] Saved ml_labeled_v17_OOS.csv ({len(oos_df)} rows)")
