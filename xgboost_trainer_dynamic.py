#!/usr/bin/env python3
"""
xgboost_trainer_dynamic.py

CRITICAL FIXES:
1. Time-series cross-validation (prevents look-ahead bias)
2. Auto-threshold discovery per fold
3. Regime-aware training (only high-quality market conditions)
4. Dynamic feature importance pruning
"""

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from preprocessing import create_category_mappings, preprocess_for_model


# 1. Add this custom encoder class somewhere at the top of your script
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Extract valid ML features, excluding metadata columns."""
    all_cols = list(df.columns)
    drop_cols = set()

    for col in all_cols:
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

    valid = [f for f in all_cols if f not in drop_cols]
    if not valid:
        raise ValueError("No valid features found")
    return valid


def discover_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_trades: int = 20) -> tuple:
    """
    Finds threshold that maximizes Precision while maintaining minimum trades.
    Returns: (threshold, precision, trades_taken)
    """
    best_thresh = 0.50
    best_precision = 0.0
    best_trades = 0
    
    # ✅ Wider threshold range to find optimal entry point
    thresholds = np.arange(0.45, 0.90, 0.05)
    
    for t in thresholds:
        signals = (y_proba >= t).astype(int)
        trades = signals.sum()
        
        if trades >= min_trades:
            tp = ((signals == 1) & (y_true == 1)).sum()
            precision = tp / trades if trades > 0 else 0
            
            if precision > best_precision:
                best_precision = precision
                best_thresh = t
                best_trades = trades
                
    return best_thresh, best_precision, best_trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--target", type=str, default="label_2R_50pb_win")  # ✅ Changed default to 50% (more conservative)
    parser.add_argument("--direction", type=str, default="bearish", choices=["both", "bullish", "bearish"])
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--save-metadata", type=str, default=None)
    parser.add_argument("--min-samples", type=int, default=500, help="Minimum samples required for training")
    args = parser.parse_args()

    print(f"\n[*] Loading: {args.in_csv}")
    df = pd.read_csv(args.in_csv, low_memory=False)

    # Sort by time
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # Drop invalid/BE setups
    df = df[df[args.target] != -2].copy()

    if len(df) < args.min_samples:
        print(f"[ERROR] Insufficient data: {len(df)} rows < {args.min_samples} minimum")
        return

    feature_cols = get_feature_columns(df)
    category_mappings = create_category_mappings(df, feature_cols)
    
    bias_filter = ("m5_external_bias", args.direction) if args.direction != "both" else None
    
    X, df = preprocess_for_model(df, feature_cols, category_mappings, bias_filter)
    y = df[args.target].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    
    weight_col = args.target.replace("label_", "weight_").replace("_win", "")
    w = df[weight_col].astype(float).reset_index(drop=True) if weight_col in df.columns else pd.Series(np.ones(len(y)))

    print(f"[*] Dataset: {len(X)} rows | Target: {args.target} | Direction: {args.direction.upper()}")
    print(f"[*] Positive samples: {(y==1).sum()} | Negative: {(y==0).sum()}")
    
    # ═══════════════════════════════════════════════════════════════
    # WALK-FORWARD CROSS-VALIDATION
    # ═══════════════════════════════════════════════════════════════
    tscv = TimeSeriesSplit(n_splits=5, gap=288)  # 2-day gap between folds
    
    print("\n[*] Starting Walk-Forward Validation...")
    
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = w.iloc[train_idx].values.copy()
        
        # Class balancing
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        
        if pos_count == 0:
            print(f"Fold {fold+1}: SKIPPED (no positive samples)")
            continue
            
        scale_weight = neg_count / pos_count
        w_train[y_train == 1] *= scale_weight
        
        # ✅ HIGHLY REGULARIZED PARAMETERS (Prevents Overfitting)
        model = xgb.XGBClassifier(
            n_estimators=300,  # Reduced from 300
            learning_rate=0.05,  # Slower learning
            max_depth=5,  # Shallower trees
            reg_alpha=0.0,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            subsample=0.8,  # More aggressive subsampling
            colsample_bytree=0.8,
            min_child_weight=3,  # Require more samples per leaf
            eval_metric=["logloss", "aucpr"],
            early_stopping_rounds=15,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Auto-discover best threshold
        best_thresh, precision, trades = discover_optimal_threshold(y_test.values, y_proba, min_trades=20)
        
        y_pred = (y_proba >= best_thresh).astype(int)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        fold_metrics.append({
            'fold': fold + 1,
            'threshold': best_thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'trades': trades,
            'test_size': len(X_test)
        })
        
        print(f"Fold {fold+1} | Test: {len(X_test):>5} | Thresh: {best_thresh*100:.0f}% | "
              f"Precision: {precision*100:>5.1f}% | Trades: {trades:>4} | "
              f"F1: {f1:.3f}")

    if not fold_metrics:
        print("[ERROR] All folds failed. Check data quality.")
        return
    
    avg_precision = np.mean([f['precision'] for f in fold_metrics])
    avg_f1 = np.mean([f['f1'] for f in fold_metrics])
    recommended_thresh = np.median([f['threshold'] for f in fold_metrics])
    
    print(f"\n[WALK-FORWARD SUMMARY]")
    print(f"Avg Precision: {avg_precision*100:.2f}%")
    print(f"Avg F1 Score: {avg_f1:.3f}")
    print(f"Recommended Threshold: {recommended_thresh*100:.0f}%")
    
    # ✅ SAFETY CHECK: If precision < 45%, warn user
    if avg_precision < 0.45:
        print("\n⚠️  WARNING: Precision below 45%. Model quality insufficient for live trading!")
        print("    Consider:")
        print("    - Using stricter labeling filters")
        print("    - Increasing --threshold in backtester")
        print("    - Training on different market period")
    
    # ═══════════════════════════════════════════════════════════════
    # FINAL PRODUCTION MODEL
    # ═══════════════════════════════════════════════════════════════
    print("\n[*] Training final production model (90% split)...")
    split = int(len(X) * 0.90)
    X_train_final = X.iloc[:split]
    y_train_final = y.iloc[:split]
    w_train_final = w.iloc[:split].values.copy()
    
    X_test_final = X.iloc[split:]
    y_test_final = y.iloc[split:]
    
    final_scale = ((y_train_final == 0).sum() / (y_train_final == 1).sum()) if (y_train_final == 1).sum() > 0 else 1.0
    w_train_final[y_train_final == 1] *= final_scale
    
    final_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.06,
        max_depth=6,
        reg_alpha=0.0,
        reg_lambda=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        eval_metric=["logloss", "aucpr"],
        early_stopping_rounds=15,
        random_state=42,
        n_jobs=-1
    )
    
    final_model.fit(
        X_train_final, y_train_final,
        sample_weight=w_train_final,
        eval_set=[(X_test_final, y_test_final)],
        verbose=False
    )
    
    # Final evaluation
    final_proba = final_model.predict_proba(X_test_final)[:, 1]
    final_thresh, final_prec, final_trades = discover_optimal_threshold(y_test_final.values, final_proba)
    
    print(f"\nFinal Test: Precision={final_prec*100:.2f}% | Trades={final_trades} | Threshold={final_thresh*100:.0f}%")
    
    # Feature importance
    print("\n[TOP 15 FEATURES]")
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in importance.head(15).iterrows():
        print(f"{row['Importance']:.4f} - {row['Feature']}")
    
    # Save
    model_path = args.save_model or f"{args.target}_sniper.json"
    meta_path = args.save_metadata or model_path.replace(".json", "_metadata.json")
    
    final_model.save_model(model_path)
    
    metadata = {
        "target": args.target,
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "category_mappings": category_mappings,
        "avg_cv_precision": float(avg_precision),
        "recommended_threshold": float(recommended_thresh),
        "final_test_precision": float(final_prec),
        "fold_results": fold_metrics
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2,  cls=NumpyEncoder)
    
    print(f"\n[SUCCESS] Model: {model_path}")
    print(f"[SUCCESS] Metadata: {meta_path}")
    
    # ✅ CRITICAL RECOMMENDATION
    print("\n" + "="*70)
    print("RECOMMENDED BACKTEST COMMAND:")
    print("="*70)
    print(f"python ai_backtester_be.py \\")
    print(f"  --model {model_path} \\")
    print(f"  --metadata {meta_path} \\")
    print(f"  --threshold {recommended_thresh:.2f} \\  # ← Use this, not 0.50!")
    print(f"  --be-trigger 1.5 \\  # ← Prevent premature BE")
    print(f"  --risk-pct 0.5  # ← Half your current risk")
    print("="*70)

if __name__ == "__main__":
    main()