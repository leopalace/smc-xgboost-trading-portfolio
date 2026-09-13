#!/usr/bin/env python3
"""
preprocessing.py
Unified Preprocessing Module for Training, Backtesting, and Live Trading

Ensures 100% prediction parity across all environments by using
identical categorical encoding and filtering logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def preprocess_for_model(
    df: pd.DataFrame,
    features: List[str],
    category_mappings: Dict[str, Dict[str, int]],
    bias_filter: Optional[Tuple[str, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unified preprocessing for training, backtesting, and live environments.

    Args:
        df: Raw feature DataFrame with all columns
        features: List of feature column names to use for prediction
        category_mappings: Dict mapping {column_name: {category_value: integer_code}}
                          e.g., {'m5_internal_bias': {'bearish': 0, 'bullish': 1, 'neutral': 2}}
        bias_filter: Optional tuple of ('column_name', 'value') to filter rows by direction
                    e.g., ('m5_external_bias', 'bearish') to keep only bearish setups.

                    Pass None only when the dataset has already been filtered upstream
                    OR when you are certain all category values in every direction are
                    present in category_mappings (e.g. a model trained on all directions).

                    WARNING: Passing None on a mixed-direction dataset whose model was
                    trained on a single direction will cause the other direction's bias
                    label to be unmapped → corrupted features → near-zero model scores.
                    Use preprocess_both_directions() instead for the 'both' case.

    Returns:
        X: Processed feature matrix (DataFrame) ready for XGBoost DMatrix
        valid_df: The filtered/processed DataFrame (for extracting metadata like SL prices)

    Raises:
        ValueError: If a categorical feature value is missing from mappings,
                    or if the bias column is not found, or if filtering leaves zero rows.
    """
    df = df.copy()

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: BIAS FILTERING (if specified)
    # ═══════════════════════════════════════════════════════════════
    if bias_filter:
        filter_col, filter_value = bias_filter
        if filter_col not in df.columns:
            raise ValueError(
                f"Bias filter column '{filter_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns[:20])}..."
            )

        initial_rows = len(df)
        df = df[df[filter_col] == filter_value].copy()
        filtered_rows = len(df)

        if filtered_rows == 0:
            raise ValueError(
                f"Bias filter removed all rows! No rows match {filter_col}='{filter_value}'. "
                f"Check your bias column values or remove the filter."
            )

        print(f"[Preprocessing] Bias filter: {initial_rows} → {filtered_rows} rows "
              f"({filter_col}='{filter_value}')")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: HANDLE MISSING FEATURE COLUMNS
    # ═══════════════════════════════════════════════════════════════
    for col in features:
        if col not in df.columns:
            print(f"[WARNING] Feature '{col}' not in DataFrame. Filling with 0.")
            df[col] = 0

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: CATEGORICAL ENCODING (STRICT - NO FALLBACK)
    # ═══════════════════════════════════════════════════════════════
    for col in features:
        if col in category_mappings:
            df[col] = df[col].fillna('unknown').astype(str)

            # After bias filtering, the only values that should appear are those
            # the model was trained on. Any unmapped value is a real data problem.
            df[col] = df[col].map(category_mappings[col])

            unmapped_mask = df[col].isna()
            if unmapped_mask.any():
                unmapped_values = df.loc[unmapped_mask, col].unique()
                raise ValueError(
                    f"Feature '{col}' contains values not in training category_mappings!\n"
                    f"Unmapped values: {unmapped_values[:10]}\n"
                    f"Expected mappings: {category_mappings[col]}\n"
                    f"This indicates:\n"
                    f"  - Training data was filtered differently (check --direction flag)\n"
                    f"  - New categories appeared in live/backtest data\n"
                    f"  - Metadata file is corrupted or from wrong model version\n"
                    f"  - You passed bias_filter=None on a mixed-direction dataset —\n"
                    f"    use preprocess_both_directions() for direction='both' instead.\n"
                    f"FIX: Retrain model or use preprocess_both_directions()."
                )

            df[col] = df[col].fillna(-1).astype(int)

        elif not pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            raise ValueError(
                f"Feature '{col}' is categorical but missing from category_mappings!\n"
                f"Unique values found: {unique_vals[:20]}\n"
                f"This will cause WRONG PREDICTIONS due to arbitrary encoding.\n"
                f"FIX: Retrain your model or check for typos in feature names."
            )

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: FINAL NUMERIC CONVERSION & CLEANUP
    # ═══════════════════════════════════════════════════════════════
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    return X, df


def preprocess_both_directions(
    df: pd.DataFrame,
    features: List[str],
    category_mappings: Dict[str, Dict[str, int]],
    bias_col: str,
    bullish_label: str = 'bullish',
    bearish_label: str = 'bearish',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses a mixed-direction dataset by running each direction separately
    through preprocess_for_model, then concatenating the results.

    This is the correct approach when direction='both' and the model was trained
    on a single direction at a time (which is the typical case). Calling
    preprocess_for_model directly with bias_filter=None on mixed data will corrupt
    the direction whose label wasn't seen during training.

    Args:
        df: Raw feature DataFrame containing rows for both directions
        features: List of feature column names
        category_mappings: Categorical encodings from model metadata
        bias_col: Name of the column that holds the direction label
                  (e.g. 'm5_external_bias')
        bullish_label: Exact string value used for bullish rows (default: 'bullish')
        bearish_label: Exact string value used for bearish rows (default: 'bearish')

    Returns:
        X: Combined processed feature matrix sorted by original row order
        valid_df: Combined filtered DataFrame sorted by timestamp

    Raises:
        ValueError: If BOTH directions fail preprocessing (at least one must succeed).
    """
    parts_X: List[pd.DataFrame] = []
    parts_df: List[pd.DataFrame] = []
    errors: Dict[str, str] = {}

    for label in (bullish_label, bearish_label):
        bias_filter = (bias_col, label)
        try:
            X_part, df_part = preprocess_for_model(
                df, features, category_mappings, bias_filter)
        except ValueError as e:
            # Zero rows or unmapped category for this direction — skip with warning
            errors[label] = str(e)
            print(f"[Preprocessing] WARNING — skipping direction '{label}': {e}")
            continue

        # Stash the original string label in a reserved column AFTER encoding.
        # preprocess_for_model encodes bias_col in-place (e.g. 'bearish' -> 0),
        # so by the time df_part is returned the original string is gone.
        # The backtester loop needs the raw string to determine trade direction.
        df_part = df_part.copy()
        df_part['_bias_raw'] = label

        parts_X.append(X_part)
        parts_df.append(df_part)

    if not parts_X:
        raise ValueError(
            f"preprocess_both_directions: both directions failed preprocessing.\n"
            + "\n".join(f"  {k}: {v}" for k, v in errors.items())
        )

    if len(parts_X) == 1:
        # Only one direction had valid rows — return it directly
        return parts_X[0], parts_df[0]

    # Concatenate and restore original timestamp order so the trade loop
    # sees a chronologically sorted sequence of mixed-direction setups.
    combined_df = (pd.concat(parts_df, ignore_index=True)
                     .sort_values('timestamp')
                     .reset_index(drop=True))
    combined_X  = (pd.concat(parts_X, ignore_index=True)
                     .loc[combined_df.index]   # realign X rows to match df order
                     .reset_index(drop=True))

    # After concat+sort the index alignment above won't work correctly because
    # combined_df already has a fresh 0..N index from reset_index. We need to
    # re-derive X in the correct order from the sorted combined_df.
    combined_X = combined_df[features].copy()
    combined_X = combined_X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    print(f"[Preprocessing] Both directions combined: {len(combined_df)} rows total "
          f"({sum(len(p) for p in parts_df[0:1])} {bullish_label} + "
          f"{sum(len(p) for p in parts_df[1:2])} {bearish_label})")

    return combined_X, combined_df


def create_category_mappings(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Creates categorical mappings from a FULL dataset (before any filtering).

    This function should be called in training BEFORE filtering by direction,
    so that all possible category values are captured in the mappings.

    Args:
        df: Full DataFrame (unfiltered)
        feature_cols: List of feature columns

    Returns:
        category_mappings: Dict of {col: {value: code}}
    """
    category_mappings = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            cat_col = df[col].fillna("unknown").astype(str).astype("category")
            category_mappings[col] = {
                str(cat): int(code)
                for code, cat in enumerate(cat_col.cat.categories)
            }
            print(f"[Mapping Created] {col}: {category_mappings[col]}")

    return category_mappings