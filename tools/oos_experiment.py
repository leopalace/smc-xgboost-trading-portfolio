#!/usr/bin/env python3
"""
oos_experiment.py — Memory-lean OOS experiment.
Loads pre-split CSVs, filters to valid rows immediately.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import gc
import sys

sys.path.insert(0, ".")
from preprocessing import create_category_mappings, preprocess_for_model

TARGET = "label_1R_50pb_win"

def get_feature_columns(df_in):
    drop_cols = set()
    for col in df_in.columns:
        cl = col.lower()
        if cl.startswith('label_') or cl.startswith('weight_'):
            drop_cols.add(col)
        elif ('time' in cl or 'date' in cl) and not cl.endswith('_since'):
            drop_cols.add(col)
        elif cl.endswith('_idx') or cl in ['index', 'open', 'high', 'low', 'close']:
            drop_cols.add(col)
        elif cl.endswith('_low') or cl.endswith('_high') or cl.endswith('equilibrium'):
            if not cl.startswith('close_wick'):
                drop_cols.add(col)
        elif 'notes' in cl or '_mitigated_' in cl:
            drop_cols.add(col)
    return [f for f in df_in.columns if f not in drop_cols]

# ═══════════════════════════════════════════════════════════════
# STEP 1: Load ONLY valid rows from pre-split files
# ═══════════════════════════════════════════════════════════════
print("[*] Loading train split (chunked, valid rows only)...")
train_chunks = []
for chunk in pd.read_csv("ml_labeled_v17_TRAIN.csv", low_memory=False, chunksize=50000):
    valid = chunk[chunk[TARGET] != -2]
    if len(valid) > 0:
        train_chunks.append(valid)
train_df = pd.concat(train_chunks, ignore_index=True)
del train_chunks; gc.collect()

print(f"Train: {len(train_df)} valid setups, WR={( train_df[TARGET]==1).mean()*100:.1f}%")

print("[*] Loading OOS split (chunked, valid rows only)...")
oos_chunks = []
for chunk in pd.read_csv("ml_labeled_v17_OOS.csv", low_memory=False, chunksize=50000):
    valid = chunk[chunk[TARGET] != -2]
    if len(valid) > 0:
        oos_chunks.append(valid)
oos_df = pd.concat(oos_chunks, ignore_index=True)
del oos_chunks; gc.collect()

print(f"OOS:   {len(oos_df)} valid setups, base WR={( oos_df[TARGET]==1).mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Feature ranking
# ═══════════════════════════════════════════════════════════════
all_features = get_feature_columns(train_df)
cat_maps = create_category_mappings(train_df, all_features)

X_full, train_proc = preprocess_for_model(train_df.copy(), all_features, cat_maps, None)
y_full = train_proc[TARGET].astype(int).reset_index(drop=True)
X_full = X_full.reset_index(drop=True)

quick = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="logloss")
quick.fit(X_full, y_full, verbose=False)

ranked = sorted(zip(all_features, quick.feature_importances_), key=lambda x: x[1], reverse=True)
del quick, X_full, train_proc; gc.collect()

print(f"\nTop 15 features:")
for i, (f, imp) in enumerate(ranked[:15]):
    print(f"  {i+1:>2}. {imp:.4f} {f}")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Test configs
# ═══════════════════════════════════════════════════════════════
from sklearn.model_selection import TimeSeriesSplit

configs = [
    {"name": "5feat_max_reg",    "top_n": 5,  "max_depth": 2, "reg_alpha": 2.0, "reg_lambda": 10.0, "lr": 0.02, "n_est": 100},
    {"name": "10feat_heavy_reg", "top_n": 10, "max_depth": 3, "reg_alpha": 1.0, "reg_lambda": 5.0, "lr": 0.03, "n_est": 150},
    {"name": "15feat_heavy_reg", "top_n": 15, "max_depth": 3, "reg_alpha": 1.0, "reg_lambda": 5.0, "lr": 0.03, "n_est": 150},
    {"name": "10feat_mod_reg",   "top_n": 10, "max_depth": 4, "reg_alpha": 0.5, "reg_lambda": 2.0, "lr": 0.05, "n_est": 200},
    {"name": "20feat_heavy_reg", "top_n": 20, "max_depth": 3, "reg_alpha": 1.0, "reg_lambda": 5.0, "lr": 0.03, "n_est": 150},
]

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Config: {cfg['name']} | {cfg['top_n']} features | depth={cfg['max_depth']}")
    print(f"{'='*60}")

    top_feats = [f[0] for f in ranked[:cfg["top_n"]]]

    X_train, tr_p = preprocess_for_model(train_df.copy(), top_feats, cat_maps, None)
    y_train = tr_p[TARGET].astype(int).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    w_col = TARGET.replace("label_", "weight_").replace("_win", "")
    w_train = tr_p[w_col].astype(float).reset_index(drop=True) if w_col in tr_p.columns else pd.Series(np.ones(len(y_train)))

    # Walk-forward CV
    tscv = TimeSeriesSplit(n_splits=5, gap=288)
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
        Xtr, Xte = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        ytr, yte = y_train.iloc[tr_idx], y_train.iloc[te_idx]
        wtr = w_train.iloc[tr_idx].values.copy()
        pos_c = (ytr == 1).sum()
        if pos_c == 0: continue
        wtr[ytr == 1] *= (ytr == 0).sum() / pos_c

        m = xgb.XGBClassifier(
            n_estimators=cfg["n_est"], learning_rate=cfg["lr"],
            max_depth=cfg["max_depth"], reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"], subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, eval_metric="logloss",
            early_stopping_rounds=15, random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xte, yte)], verbose=False)

        proba = m.predict_proba(Xte)[:, 1]
        best_t, best_p, best_n = 0.5, 0, 0
        for t in np.arange(0.40, 0.85, 0.05):
            sigs = (proba >= t).astype(int)
            n = sigs.sum()
            if n >= 15:
                p = ((sigs == 1) & (yte == 1)).sum() / n
                if p > best_p: best_p, best_t, best_n = p, t, n
        fold_metrics.append({"prec": best_p, "thresh": best_t, "trades": best_n})
        print(f"  Fold {fold+1}: {best_p*100:.1f}% on {best_n} trades @{best_t:.2f}")

    if not fold_metrics: continue
    avg_p = np.mean([f["prec"] for f in fold_metrics])
    print(f"  CV Avg: {avg_p*100:.1f}%")

    # Train on ALL train data, evaluate on OOS
    w_f = w_train.values.copy()
    w_f[y_train == 1] *= (y_train == 0).sum() / (y_train == 1).sum()
    fm = xgb.XGBClassifier(
        n_estimators=cfg["n_est"], learning_rate=cfg["lr"],
        max_depth=cfg["max_depth"], reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"], subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, eval_metric="logloss", random_state=42, n_jobs=-1)
    fm.fit(X_train, y_train, sample_weight=w_f, verbose=False)

    X_oos, oos_p = preprocess_for_model(oos_df.copy(), top_feats, cat_maps, None)
    y_oos = oos_p[TARGET].astype(int).reset_index(drop=True)
    X_oos = X_oos.reset_index(drop=True)
    oos_proba = fm.predict_proba(X_oos)[:, 1]

    oos_base = (y_oos == 1).mean() * 100
    print(f"\n  OOS (base WR: {oos_base:.1f}%):")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        preds = (oos_proba >= t).astype(int)
        n = preds.sum()
        if n > 0:
            wr = ((preds == 1) & (y_oos == 1)).sum() / n * 100
            flag = "✓ EDGE" if wr > oos_base else "✗"
            print(f"    @{t:.2f}: {wr:.1f}% on {n} trades {flag}")

    del X_train, X_oos, fm; gc.collect()

print("\n[DONE]")
