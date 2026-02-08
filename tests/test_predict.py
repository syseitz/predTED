"""Tests for predted prediction API — predict, predict_float, predict_matrix."""

import numpy as np
import pytest

import predted


# ---------------------------------------------------------------------------
# predict() — single pair, rounded integer
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_int(self):
        result = predted.predict("((..))", "(())..")
        assert isinstance(result, int)

    def test_non_negative(self):
        result = predted.predict("((..))", "(())..")
        assert result >= 0

    def test_identical_structures_lower_than_different(self):
        same = predted.predict("(((((......)))))", "(((((......)))))")
        diff = predted.predict("(((((......)))))", "................")
        assert same < diff

    def test_very_different_structures(self):
        s1 = "((((((.......))))))........."
        s2 = "......(((((((.......)))))))"
        result = predted.predict(s1, s2)
        assert result > 0

    def test_known_pair(self):
        """Regression test: this pair should give ~20."""
        result = predted.predict("((..))", "(())..")
        assert 10 <= result <= 30


# ---------------------------------------------------------------------------
# predict_float() — single pair, raw float
# ---------------------------------------------------------------------------

class TestPredictFloat:
    def test_returns_float(self):
        result = predted.predict_float("((..))", "(())..")
        assert isinstance(result, float)

    def test_non_negative(self):
        result = predted.predict_float("((..))", "(())..")
        assert result >= 0.0

    def test_identical_structures_lower_than_different(self):
        same = predted.predict_float("(((((......)))))", "(((((......)))))")
        diff = predted.predict_float("(((((......)))))", "................")
        assert same < diff

    def test_consistent_with_predict(self):
        raw = predted.predict_float("((..))", "(())..")
        rounded = predted.predict("((..))", "(())..")
        assert rounded == max(0, round(raw))


# ---------------------------------------------------------------------------
# predict_matrix() — pairwise distance matrix
# ---------------------------------------------------------------------------

class TestPredictMatrix:
    STRUCTURES = ["((..))", "(())..", "...((..))", "((((...))))"]

    def test_shape(self):
        matrix = predted.predict_matrix(self.STRUCTURES)
        n = len(self.STRUCTURES)
        assert matrix.shape == (n, n)

    def test_symmetric(self):
        matrix = predted.predict_matrix(self.STRUCTURES)
        np.testing.assert_array_equal(matrix, matrix.T)

    def test_zero_diagonal(self):
        matrix = predted.predict_matrix(self.STRUCTURES)
        np.testing.assert_array_equal(np.diag(matrix), 0)

    def test_non_negative(self):
        matrix = predted.predict_matrix(self.STRUCTURES)
        assert np.all(matrix >= 0)

    def test_dtype_int(self):
        matrix = predted.predict_matrix(self.STRUCTURES, dtype=int)
        assert matrix.dtype == int

    def test_dtype_float(self):
        matrix = predted.predict_matrix(self.STRUCTURES, dtype=float)
        assert matrix.dtype == np.float64

    def test_empty_list(self):
        matrix = predted.predict_matrix([])
        assert matrix.shape == (0, 0)

    def test_single_structure(self):
        matrix = predted.predict_matrix(["((..))"])
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 0

    def test_consistent_with_predict(self):
        """Matrix entries should match individual predict() calls."""
        structs = ["((..))", "(())..", "...((..))"]
        matrix = predted.predict_matrix(structs, dtype=int)
        for i in range(len(structs)):
            for j in range(i + 1, len(structs)):
                single = predted.predict(structs[i], structs[j])
                assert matrix[i, j] == single, (
                    f"Mismatch at [{i},{j}]: matrix={matrix[i,j]}, "
                    f"single={single}"
                )
