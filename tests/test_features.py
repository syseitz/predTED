"""Tests for predted.features — the 36 structural feature computation."""

import numpy as np
import pytest

from predted.features import (
    NUM_FEATURES,
    USING_C_BACKEND,
    compute_features,
    _compute_features_python,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_num_features(self):
        assert NUM_FEATURES == 36

    def test_using_c_backend_is_bool(self):
        assert isinstance(USING_C_BACKEND, bool)


# ---------------------------------------------------------------------------
# Output shape & type
# ---------------------------------------------------------------------------

class TestOutputShape:
    @pytest.mark.parametrize("structure", [
        "((..))",
        "....",
        "((((...))))",
        ".((..((.))..)).",
    ])
    def test_returns_36_features(self, structure: str):
        features = compute_features(structure)
        assert features.shape == (NUM_FEATURES,)

    def test_returns_float64(self):
        features = compute_features("((..))")
        assert features.dtype == np.float64

    def test_no_nan_or_inf(self):
        features = compute_features("(((..((....))...)))")
        assert np.all(np.isfinite(features))


# ---------------------------------------------------------------------------
# Known structures
# ---------------------------------------------------------------------------

class TestKnownStructures:
    def test_length_feature(self):
        """Feature index 4 is the structure length."""
        s = "((....))"
        features = compute_features(s)
        assert features[4] == len(s)

    def test_all_unpaired(self):
        """A fully unpaired structure should have zero paired bases."""
        features = compute_features("........")
        # Feature 22 = paired base count
        assert features[22] == 0
        # Feature 23 = unpaired base count
        assert features[23] == 8

    def test_simple_hairpin(self):
        """A simple hairpin '((...))' should have 1 hairpin loop."""
        features = compute_features("((...))")
        # Feature 19 = hairpin loop count
        assert features[19] >= 1

    def test_stacked_pairs(self):
        """'((((...))))' has consecutive stacked base pairs."""
        features = compute_features("((((...))))")
        # Feature 20 = stacked pairs count
        assert features[20] > 0

    def test_identical_structures_same_features(self):
        s = "..(((..((...))..)))."
        f1 = compute_features(s)
        f2 = compute_features(s)
        np.testing.assert_array_equal(f1, f2)


# ---------------------------------------------------------------------------
# C backend vs. Python fallback consistency
# ---------------------------------------------------------------------------

class TestCPythonConsistency:
    """Ensure C extension and pure-Python produce identical features."""

    STRUCTURES = [
        "((..))",
        "....",
        "((((...))))",
        ".((..((.))..)).",
        "((((....))))..(((...)))...",
        "((.((..)).))",
    ]

    @pytest.mark.parametrize("structure", STRUCTURES)
    def test_c_matches_python(self, structure: str):
        python_features = _compute_features_python(structure)
        combined_features = compute_features(structure)
        np.testing.assert_allclose(
            combined_features, python_features, rtol=1e-10,
            err_msg=f"Feature mismatch for '{structure}'",
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_dot(self):
        features = compute_features(".")
        assert features.shape == (NUM_FEATURES,)
        assert features[4] == 1  # length

    def test_short_hairpin(self):
        features = compute_features("(..)")
        assert features.shape == (NUM_FEATURES,)

    def test_long_structure(self):
        """Stress test with a large structure."""
        s = "(" * 500 + "." * 100 + ")" * 500
        features = compute_features(s)
        assert features.shape == (NUM_FEATURES,)
        assert features[4] == 1100
