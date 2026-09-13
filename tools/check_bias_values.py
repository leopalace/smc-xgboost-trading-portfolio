#!/usr/bin/env python3
"""Quick check: what does m15_external_bias look like after preprocessing?"""
import pandas as pd
import json
import gc
import sys
sys.path.insert(0, ".")
from preprocessing import preprocess_for_model

TARGET = "label_1R_50pb_win"

chunks = []
for chunk in pd.read_csv("ml_labeled_v17_OOS.csv", low_memory=False, chunksize=50000):
    valid = chunk[chunk[TARGET] != -2]
    if len(valid) > 0:
        chunks.append(valid.head(50))
        break
df = pd.concat(chunks, ignore_index=True)

with open("model_1R_50pb_v17r_metadata.json") as f:
    meta = json.load(f)

print("BEFORE preprocessing:")
print(f"  m15_external_bias dtype: {df['m15_external_bias'].dtype}")
print(f"  Values: {df['m15_external_bias'].unique()[:10]}")
print(f"  Sample: {df['m15_external_bias'].head(10).tolist()}")

X, proc = preprocess_for_model(df, meta["features"], meta.get("category_mappings", {}), None)

print("\nAFTER preprocessing:")
col = "m15_external_bias"
if col in proc.columns:
    print(f"  dtype: {proc[col].dtype}")
    print(f"  Values: {proc[col].unique()}")
    vals = proc[col].head(10).tolist()
    print(f"  Sample: {vals}")
    print(f"  str(vals[0]): '{str(vals[0])}'")

    # Test the backtester's direction check
    for v in proc[col].unique():
        sv = str(v).strip().lower()
        is_bearish = sv in ('bearish', 'bear', '-1', '0', 'sell', 'short')
        print(f"  Value={v} → str='{sv}' → is_bearish={is_bearish}")
