#!/usr/bin/env python3
"""
get_top_features.py — Extract top N features from trained model,
then print the --drop-features command for all other features.
"""
import json
import numpy as np
import xgboost as xgb

TOP_N = 25

# Load metadata to get feature list
with open("model_1R_50pb_v17_metadata.json") as f:
    meta = json.load(f)

features = meta["features"]

# Load model to get importances
model = xgb.XGBClassifier()
model.load_model("model_1R_50pb_v17.json")

importances = model.feature_importances_

# Rank
ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

print(f"=== TOP {TOP_N} FEATURES (from full v17 model) ===\n")
top_features = []
for i, (feat, imp) in enumerate(ranked[:TOP_N]):
    print(f"  {i+1:>2}. {imp:.4f}  {feat}")
    top_features.append(feat)

# Features to drop = everything NOT in top N
drop_features = [feat for feat, _ in ranked[TOP_N:]]

print(f"\n=== FEATURES TO DROP ({len(drop_features)}) ===\n")
for feat in drop_features:
    print(f"  - {feat}")

# Print the command-line argument
print(f"\n=== COPY-PASTE FOR --drop-features ===\n")
print("--drop-features " + " ".join(drop_features))
