#!/usr/bin/env python3
"""
oos_confirmed.py — OOS pipeline with M5 BOS/CHOCH and engulfing confirmation filters.
Tests 3 filter variants + unfiltered baseline on the 1R target.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import gc
import sys
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, ".")
from preprocessing import create_category_mappings, preprocess_for_model

TARGET = "label_1R_50pb_win"
BREAKEVEN = 50.0

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


def apply_filter(df, filter_name):
    """Return boolean mask for the confirmation filter."""
    bias = df['m15_external_bias'].astype(str).str.lower()
    is_bull = bias == 'bullish'
    is_bear = bias == 'bearish'

    if filter_name == "none":
        return pd.Series(True, index=df.index)

    elif filter_name == "m5_bos_choch_5":
        bull_c = is_bull & ((df['m5_bos_bull_bars_since'] <= 5) | (df['m5_choch_bull_bars_since'] <= 5))
        bear_c = is_bear & ((df['m5_bos_bear_bars_since'] <= 5) | (df['m5_choch_bear_bars_since'] <= 5))
        return bull_c | bear_c

    elif filter_name == "engulfing_50":
        bull_eng = is_bull & (df['close'] > df['open'])
        bear_eng = is_bear & (df['close'] < df['open'])
        eng = bull_eng | bear_eng
        body_pct = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        return eng & (body_pct >= 0.50)

    elif filter_name == "m5_bos5_engulf":
        bull_c = is_bull & ((df['m5_bos_bull_bars_since'] <= 5) | (df['m5_choch_bull_bars_since'] <= 5))
        bear_c = is_bear & ((df['m5_bos_bear_bars_since'] <= 5) | (df['m5_choch_bear_bars_since'] <= 5))
        m5_conf = bull_c | bear_c
        bull_eng = is_bull & (df['close'] > df['open'])
        bear_eng = is_bear & (df['close'] < df['open'])
        eng = bull_eng | bear_eng
        body_pct = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        return m5_conf & eng & (body_pct >= 0.30)

    elif filter_name == "m5_bos_choch_10":
        bull_c = is_bull & ((df['m5_bos_bull_bars_since'] <= 10) | (df['m5_choch_bull_bars_since'] <= 10))
        bear_c = is_bear & ((df['m5_bos_bear_bars_since'] <= 10) | (df['m5_choch_bear_bars_since'] <= 10))
        return bull_c | bear_c

    return pd.Series(True, index=df.index)


def run_oos_test(train_df, oos_df, filter_name, all_features, cat_maps):
    """Run CV + OOS test for one filter variant."""
    filt_train = apply_filter(train_df, filter_name)
    filt_oos = apply_filter(oos_df, filter_name)

    tv = train_df[filt_train & train_df[TARGET].isin([0, 1])].copy().reset_index(drop=True)
    ov = oos_df[filt_oos & oos_df[TARGET].isin([0, 1])].copy().reset_index(drop=True)

    train_wr = (tv[TARGET] == 1).mean() * 100
    oos_base_wr = (ov[TARGET] == 1).mean() * 100

    print(f"\n  Filter: {filter_name}")
    print(f"  Train: {len(tv)} setups, WR={train_wr:.1f}%")
    print(f"  OOS:   {len(ov)} setups, base WR={oos_base_wr:.1f}%")

    if len(tv) < 50:
        print(f"  SKIP — too few training samples")
        return

    # Use features available in filtered subset
    usable_features = [f for f in all_features if f in tv.columns]

    # Feature ranking
    X_all, proc = preprocess_for_model(tv.copy(), usable_features, cat_maps, None)
    y_all = proc[TARGET].astype(int).reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)

    quick = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="logloss")
    quick.fit(X_all, y_all, verbose=False)
    ranked = sorted(zip(usable_features, quick.feature_importances_), key=lambda x: x[1], reverse=True)
    del quick, X_all, proc; gc.collect()

    # Adaptive CV
    n = len(tv)
    if n < 200:
        n_splits, gap = 2, 10
    elif n < 500:
        n_splits, gap = 3, 20
    elif n < 1500:
        n_splits, gap = 3, 50
    else:
        n_splits, gap = 5, 100

    configs = [
        {"name": "10feat_heavy", "top_n": 10, "max_depth": 3, "reg_alpha": 1.0, "reg_lambda": 5.0, "lr": 0.03, "n_est": 150},
        {"name": "5feat_max",    "top_n": 5,  "max_depth": 2, "reg_alpha": 2.0, "reg_lambda": 10.0, "lr": 0.02, "n_est": 100},
    ]

    for cfg in configs:
        top_n = min(cfg["top_n"], len(ranked))
        top_feats = [f[0] for f in ranked[:top_n]]

        print(f"\n    --- {cfg['name']} ({top_n} feats, splits={n_splits} gap={gap}) ---")

        X_train, tr_p = preprocess_for_model(tv.copy(), top_feats, cat_maps, None)
        y_train = tr_p[TARGET].astype(int).reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        w_col = TARGET.replace("label_", "weight_").replace("_win", "")
        w_train = tr_p[w_col].astype(float).reset_index(drop=True) if w_col in tr_p.columns else pd.Series(np.ones(len(y_train)))

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
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
            for t in np.arange(0.30, 0.85, 0.05):
                sigs = (proba >= t).astype(int)
                sig_n = sigs.sum()
                if sig_n >= 5:
                    p = ((sigs == 1) & (yte == 1)).sum() / sig_n
                    if p > best_p: best_p, best_t, best_n = p, t, sig_n
            fold_metrics.append({"prec": best_p, "thresh": best_t, "trades": best_n})
            print(f"      Fold {fold+1}: {best_p*100:.1f}% on {best_n} trades @{best_t:.2f}")

        if not fold_metrics: continue
        avg_p = np.mean([f["prec"] for f in fold_metrics])
        print(f"      CV Avg: {avg_p*100:.1f}%")

        # Full train → OOS
        w_f = w_train.values.copy()
        pos_total = (y_train == 1).sum()
        if pos_total > 0:
            w_f[y_train == 1] *= (y_train == 0).sum() / pos_total
        fm = xgb.XGBClassifier(
            n_estimators=cfg["n_est"], learning_rate=cfg["lr"],
            max_depth=cfg["max_depth"], reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"], subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, eval_metric="logloss", random_state=42, n_jobs=-1)
        fm.fit(X_train, y_train, sample_weight=w_f, verbose=False)

        X_oos, oos_p = preprocess_for_model(ov.copy(), top_feats, cat_maps, None)
        y_oos = oos_p[TARGET].astype(int).reset_index(drop=True)
        X_oos = X_oos.reset_index(drop=True)
        oos_proba = fm.predict_proba(X_oos)[:, 1]

        print(f"\n      OOS (base {oos_base_wr:.1f}%, need >{BREAKEVEN:.1f}%):")
        for t in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            preds = (oos_proba >= t).astype(int)
            sig_n = preds.sum()
            if sig_n > 0:
                wr = ((preds == 1) & (y_oos == 1)).sum() / sig_n * 100
                flag = " ✓ PROFITABLE" if wr > BREAKEVEN else ""
                print(f"        @{t:.2f}: {wr:.1f}% on {sig_n} trades{flag}")

        del X_train, X_oos, fm; gc.collect()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print("[*] Loading fixed CSV...")
chunks = []
for chunk in pd.read_csv("ml_dataset_labeled_v17_fixed.csv", low_memory=False, chunksize=50000):
    valid = chunk[chunk[TARGET].isin([0, 1])]
    if len(valid) > 0:
        chunks.append(valid)
df = pd.concat(chunks, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
df = df.sort_values('timestamp').reset_index(drop=True)
del chunks; gc.collect()

print(f"Valid rows: {len(df)}")
print(f"Date range: {df.timestamp.min()} to {df.timestamp.max()}")
print(f"Base WR: {(df[TARGET]==1).mean()*100:.1f}%")

# 70/30 time split
split_idx = int(len(df) * 0.70)
split_time = df.loc[split_idx, 'timestamp']
train_df = df.iloc[:split_idx].copy()
oos_df = df.iloc[split_idx:].copy()
del df; gc.collect()

print(f"Split at: {split_time}")
print(f"Train: {len(train_df)}, OOS: {len(oos_df)}")

# Get features and category maps from full train set
all_features = get_feature_columns(train_df)
cat_maps = create_category_mappings(train_df, all_features)

# Test each filter variant
filters = [
    "none",              # baseline (no confirmation)
    "m5_bos_choch_5",    # M5 BOS or CHOCH within 5 bars
    "m5_bos_choch_10",   # M5 BOS or CHOCH within 10 bars
    "engulfing_50",      # Directional candle, body >= 50%
    "m5_bos5_engulf",    # Both: M5 BOS/CHOCH(5) + engulfing(30%)
]

for f in filters:
    print(f"\n{'#'*70}")
    print(f"# FILTER: {f}")
    print(f"{'#'*70}")
    run_oos_test(train_df, oos_df, f, all_features, cat_maps)

print("\n[DONE]")
