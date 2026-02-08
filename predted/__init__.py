"""predted — Predict pairwise Tree Edit Distances for RNA secondary structures.

Uses structural features + a LightGBM model to approximate TED without
running the expensive exact algorithm.

Usage::

    import predted

    # Single pair
    ted = predted.predict("((..))", "(())..")

    # Pairwise distance matrix
    matrix = predted.predict_matrix(["((..))", "(())..", "...((..))"])
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .features import NUM_FEATURES, compute_features

__version__ = "0.1.0"
__all__ = ["predict", "predict_float", "predict_matrix", "compute_features"]

_MODEL_PATH = Path(__file__).resolve().parent / "model.txt"

# Lazy-loaded booster singleton
_booster = None

# Number of rich features: diff + sum + min + max
NUM_RICH_FEATURES: int = NUM_FEATURES * 4  # 144


def _get_booster():
    """Load the LightGBM model on first use."""
    global _booster
    if _booster is None:
        import lightgbm as lgb

        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {_MODEL_PATH}\n"
                "Ensure model.txt is in the predTED repository root."
            )
        _booster = lgb.Booster(model_file=str(_MODEL_PATH))
    return _booster


def _build_rich_features(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Build 144 rich pairwise features: diff + sum + min + max."""
    return np.concatenate([
        np.abs(f1 - f2),          # diff (36)
        f1 + f2,                  # sum  (36)
        np.minimum(f1, f2),       # min  (36)
        np.maximum(f1, f2),       # max  (36)
    ])


def predict(struct1: str, struct2: str) -> int:
    """Predict the Tree Edit Distance between two dot-bracket structures.

    Returns the predicted TED as a non-negative integer.
    """
    val = predict_float(struct1, struct2)
    return max(0, round(val))


def predict_float(struct1: str, struct2: str) -> float:
    """Predict the Tree Edit Distance (raw float, before rounding)."""
    f1 = compute_features(struct1)
    f2 = compute_features(struct2)
    rich = _build_rich_features(f1, f2).reshape(1, -1)
    booster = _get_booster()
    pred = booster.predict(rich)[0]
    return max(0.0, float(pred))


def predict_matrix(
    structures: List[str],
    *,
    dtype: type = int,
) -> np.ndarray:
    """Compute the pairwise predicted-TED matrix for a list of structures.

    Parameters
    ----------
    structures
        List of dot-bracket notation strings.
    dtype
        ``int`` (default) for rounded integers, ``float`` for raw predictions.

    Returns
    -------
    np.ndarray
        Symmetric N*N distance matrix with zeros on the diagonal.
    """
    n = len(structures)
    if n == 0:
        return np.zeros((0, 0), dtype=dtype)

    # Pre-compute per-structure features
    all_features = np.array([compute_features(s) for s in structures])

    # Build rich pairwise features for upper triangle
    pairs: List[np.ndarray] = []
    indices: List[tuple] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(_build_rich_features(all_features[i], all_features[j]))
            indices.append((i, j))

    if not pairs:
        return np.zeros((n, n), dtype=dtype)

    pair_array = np.array(pairs, dtype=np.float32)
    booster = _get_booster()
    predictions = booster.predict(pair_array)
    predictions = np.clip(predictions, 0, None)

    use_float = (dtype is float or dtype is np.float64 or dtype is np.float32)

    if use_float:
        matrix = np.zeros((n, n), dtype=np.float64)
    else:
        predictions = np.round(predictions).astype(int)
        matrix = np.zeros((n, n), dtype=int)

    for k, (i, j) in enumerate(indices):
        matrix[i, j] = predictions[k]
        matrix[j, i] = predictions[k]

    return matrix
