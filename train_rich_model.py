"""Train LightGBM model with rich features (144 = 36 base × 4 combinations).

Uses predted's C-compatible feature computation as the single source of truth.
"""

import itertools
import json
import time

import lightgbm as lgb
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from predted.features import compute_features, NUM_FEATURES


def build_rich_features(fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
    """Build 144 rich pairwise features: diff + sum + min + max."""
    return np.concatenate([
        np.abs(fi - fj),          # diff (36)
        fi + fj,                  # sum  (36)
        np.minimum(fi, fj),       # min  (36)
        np.maximum(fi, fj),       # max  (36)
    ])


def main():
    structures_file = "data/structures.txt"
    ted_matrix_file = "data/ted_matrix.txt"

    with open(structures_file) as f:
        structures = [line.strip() for line in f if line.strip()]
    ted_matrix = np.loadtxt(ted_matrix_file, delimiter=" ")

    N = len(structures)
    print(f"Loaded {N} structures, TED matrix shape: {ted_matrix.shape}")

    # Compute per-structure features
    print("Computing per-structure features (36 base features)...")
    feature_matrix = np.array(
        [compute_features(s) for s in tqdm(structures, desc="Features")],
        dtype=np.float64,
    )

    # Build pairwise rich features (144)
    pairs = list(itertools.combinations(range(N), 2))
    y = np.array([ted_matrix[i, j] for i, j in pairs], dtype=np.float64)
    print(f"{len(pairs):,} pairs, building rich features (144)...")

    X = np.array(
        [build_rich_features(feature_matrix[i], feature_matrix[j])
         for i, j in tqdm(pairs, desc="Rich features")],
        dtype=np.float32,
    )
    print(f"Feature matrix shape: {X.shape}")

    # Remove constant features
    variances = np.var(X, axis=0)
    non_const = np.where(variances > 0)[0]
    X = X[:, non_const]
    print(f"After removing constants: {X.shape[1]} features")

    # Split: 64% train, 16% valid, 20% test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_tv, y_tv, test_size=0.20, random_state=42
    )
    print(f"Train: {len(y_train):,}  Valid: {len(y_valid):,}  Test: {len(y_test):,}")

    # Train
    best_params = {"max_depth": -1, "num_leaves": 250, "n_estimators": 100}
    print(f"Training LightGBM with params: {best_params}")

    reg = lgb.LGBMRegressor(**best_params)
    reg.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=10),
        ],
    )

    # Evaluate on test set
    test_pred = np.maximum(reg.predict(X_test), 0)
    test_pred_int = np.round(test_pred).astype(int)

    r2_float = r2_score(y_test, test_pred)
    mae_float = mean_absolute_error(y_test, test_pred)
    r2_int = r2_score(y_test, test_pred_int)
    mae_int = mean_absolute_error(y_test, test_pred_int)
    r_val, _ = pearsonr(y_test, test_pred)
    mape = float(np.mean(np.abs((y_test - test_pred) / (y_test + 1e-6))) * 100)

    print(f"\nTest Results (float):  R²={r2_float:.4f}  MAE={mae_float:.3f}  MAPE={mape:.2f}%  r={r_val:.4f}")
    print(f"Test Results (int):   R²={r2_int:.4f}  MAE={mae_int:.3f}")

    # Save model
    reg.booster_.save_model("model.txt")
    print("\nModel saved to model.txt")

    # Save summary
    summary = {
        "model": "Rich 144f",
        "base_features": NUM_FEATURES,
        "rich_features": NUM_FEATURES * 4,
        "features_after_const_removal": int(X.shape[1]),
        "non_const_indices": non_const.tolist(),
        "params": best_params,
        "test_metrics": {
            "R2_float": float(r2_float),
            "MAE_float": float(mae_float),
            "R2_int": float(r2_int),
            "MAE_int": float(mae_int),
            "MAPE_percent": float(mape),
            "pearson_r": float(r_val),
        },
    }
    with open("weights/results_summary_rich.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to weights/results_summary_rich.json")


if __name__ == "__main__":
    main()
