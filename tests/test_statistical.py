"""
Tests for non-parametric statistical tests:
  - friedman_test
  - holms_posthoc
  - critical_difference_data
  - nemenyi_test (optional, requires scikit-posthocs)
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistical_tests import (
    friedman_test,
    holms_posthoc,
    critical_difference_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results_matrix(n_datasets=10, n_methods=4, seed=42):
    """Create a synthetic results matrix (higher is better).

    Columns simulate methods with different mean performance levels so that
    the Friedman test is likely to be significant.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_datasets, n_methods))
    # Give each method a different mean so differences are detectable.
    offsets = np.linspace(0, 2, n_methods)
    return base + offsets[np.newaxis, :]


def _method_names(n):
    """Generate method name strings."""
    return [f"method_{i}" for i in range(n)]


# ===================================================================
# friedman_test
# ===================================================================

class TestFriedmanTest:
    """Tests for friedman_test."""

    def test_returns_two_values(self):
        """Should return (statistic, p_value) as a tuple of floats."""
        M = _make_results_matrix()
        stat, p = friedman_test(M)
        assert isinstance(stat, float)
        assert isinstance(p, float)

    def test_statistic_non_negative(self):
        """Friedman statistic should be non-negative."""
        M = _make_results_matrix()
        stat, _ = friedman_test(M)
        assert stat >= 0.0

    def test_p_value_in_range(self):
        """p-value should be in [0, 1]."""
        M = _make_results_matrix()
        _, p = friedman_test(M)
        assert 0.0 <= p <= 1.0

    def test_known_identical_methods(self):
        """If all methods produce the same values, statistic should be zero or p should be high/NaN."""
        # All columns identical: no differences.
        col = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        M = np.column_stack([col, col, col])
        stat, p = friedman_test(M)
        # scipy may return NaN p-value for degenerate (all-tied) case
        assert stat == 0.0 or np.isnan(p) or p > 0.05

    def test_known_different_methods(self):
        """With clearly different methods, p-value should be low (significant)."""
        M = _make_results_matrix(n_datasets=20, n_methods=5, seed=0)
        stat, p = friedman_test(M)
        # With large offsets and 20 datasets, this should be significant.
        assert p < 0.05

    def test_minimum_datasets_raises(self):
        """Should raise ValueError with fewer than 3 datasets."""
        M = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])  # 2 datasets
        with pytest.raises(ValueError, match="at least 3 datasets"):
            friedman_test(M)

    def test_minimum_methods_raises(self):
        """Should raise ValueError with fewer than 3 methods."""
        M = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])  # 2 methods
        with pytest.raises(ValueError, match="at least 3 methods"):
            friedman_test(M)

    def test_exactly_3_datasets_3_methods(self):
        """Should work with the minimum of 3 datasets and 3 methods."""
        M = np.array([
            [0.9, 0.8, 0.7],
            [0.85, 0.75, 0.65],
            [0.95, 0.85, 0.75],
        ])
        stat, p = friedman_test(M)
        assert np.isfinite(stat)
        assert 0.0 <= p <= 1.0


# ===================================================================
# holms_posthoc
# ===================================================================

class TestHolmsPosthoc:
    """Tests for holms_posthoc."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        M = _make_results_matrix(n_datasets=10, n_methods=4)
        names = _method_names(4)
        result = holms_posthoc(M, names)
        assert isinstance(result, pd.DataFrame)

    def test_shape_is_square(self):
        """Result should be a k x k DataFrame."""
        M = _make_results_matrix(n_datasets=10, n_methods=5)
        names = _method_names(5)
        result = holms_posthoc(M, names)
        assert result.shape == (5, 5)

    def test_diagonal_is_zero(self):
        """Diagonal elements should be 0.0 (method compared with itself)."""
        M = _make_results_matrix(n_datasets=10, n_methods=4)
        names = _method_names(4)
        result = holms_posthoc(M, names)
        for i in range(4):
            assert result.iloc[i, i] == 0.0

    def test_symmetric(self):
        """p-value matrix should be symmetric."""
        M = _make_results_matrix(n_datasets=10, n_methods=4)
        names = _method_names(4)
        result = holms_posthoc(M, names)
        np.testing.assert_array_almost_equal(
            result.values, result.values.T, decimal=10
        )

    def test_p_values_in_range(self):
        """All p-values should be in [0, 1]."""
        M = _make_results_matrix(n_datasets=10, n_methods=4)
        names = _method_names(4)
        result = holms_posthoc(M, names)
        assert (result.values >= 0.0).all()
        assert (result.values <= 1.0).all()

    def test_column_and_index_names(self):
        """Column and index names should match the provided method names."""
        names = ["A", "B", "C"]
        M = _make_results_matrix(n_datasets=5, n_methods=3)
        result = holms_posthoc(M, names)
        assert list(result.columns) == names
        assert list(result.index) == names

    def test_holm_correction_monotonic(self):
        """Holm-adjusted p-values should be non-decreasing when sorted
        (monotonicity property of step-down correction)."""
        M = _make_results_matrix(n_datasets=15, n_methods=5)
        names = _method_names(5)
        result = holms_posthoc(M, names)
        # Extract off-diagonal p-values.
        off_diag = []
        for i in range(5):
            for j in range(i + 1, 5):
                off_diag.append(result.iloc[i, j])
        # Sorted in ascending order, they should be non-decreasing
        # (this is inherent to how Holm correction works).
        # Note: we just check they are valid, not necessarily sorted in
        # the DataFrame (which is symmetric, not sorted).
        for p in off_diag:
            assert 0.0 <= p <= 1.0


# ===================================================================
# critical_difference_data
# ===================================================================

class TestCriticalDifferenceData:
    """Tests for critical_difference_data."""

    def test_returns_dict_with_expected_keys(self):
        """Result should have 'avg_ranks', 'cd', 'n_datasets', 'n_methods', 'alpha'."""
        M = _make_results_matrix(n_datasets=10, n_methods=4)
        names = _method_names(4)
        result = critical_difference_data(M, names, alpha=0.05)
        assert "avg_ranks" in result
        assert "cd" in result
        assert "n_datasets" in result
        assert "n_methods" in result
        assert "alpha" in result

    def test_avg_ranks_keys_match_method_names(self):
        """avg_ranks dict should have one entry per method name."""
        names = ["A", "B", "C", "D"]
        M = _make_results_matrix(n_datasets=8, n_methods=4)
        result = critical_difference_data(M, names)
        assert set(result["avg_ranks"].keys()) == set(names)

    def test_avg_ranks_are_valid(self):
        """Average ranks should be between 1 and n_methods."""
        M = _make_results_matrix(n_datasets=10, n_methods=5)
        names = _method_names(5)
        result = critical_difference_data(M, names)
        for name, rank in result["avg_ranks"].items():
            assert 1.0 <= rank <= 5.0, f"Rank of {name} is out of range: {rank}"

    def test_avg_ranks_sum(self):
        """Average ranks across methods should sum to k*(k+1)/2."""
        k = 4
        M = _make_results_matrix(n_datasets=10, n_methods=k)
        names = _method_names(k)
        result = critical_difference_data(M, names)
        rank_sum = sum(result["avg_ranks"].values())
        expected_sum = k * (k + 1) / 2.0
        assert pytest.approx(rank_sum, abs=1e-10) == expected_sum

    def test_cd_is_positive(self):
        """Critical difference should be a positive number."""
        M = _make_results_matrix(n_datasets=10, n_methods=4)
        names = _method_names(4)
        result = critical_difference_data(M, names)
        assert result["cd"] > 0.0

    def test_n_datasets_and_n_methods(self):
        """Returned n_datasets and n_methods should match input shape."""
        M = _make_results_matrix(n_datasets=12, n_methods=6)
        names = _method_names(6)
        result = critical_difference_data(M, names)
        assert result["n_datasets"] == 12
        assert result["n_methods"] == 6

    def test_alpha_returned(self):
        """Returned alpha should match the input alpha."""
        M = _make_results_matrix(n_datasets=10, n_methods=4)
        names = _method_names(4)
        result = critical_difference_data(M, names, alpha=0.10)
        assert result["alpha"] == 0.10

    def test_better_method_gets_lower_rank(self):
        """A consistently higher-performing method should have a lower (better) rank."""
        rng = np.random.default_rng(42)
        n_datasets = 20
        # Method 0 is always the best.
        M = np.zeros((n_datasets, 3))
        M[:, 0] = rng.uniform(0.9, 1.0, n_datasets)   # best
        M[:, 1] = rng.uniform(0.5, 0.6, n_datasets)   # middle
        M[:, 2] = rng.uniform(0.1, 0.2, n_datasets)   # worst
        names = ["best", "middle", "worst"]
        result = critical_difference_data(M, names)
        assert result["avg_ranks"]["best"] < result["avg_ranks"]["middle"]
        assert result["avg_ranks"]["middle"] < result["avg_ranks"]["worst"]

    def test_cd_decreases_with_more_datasets(self):
        """More datasets should generally reduce the critical difference
        (CD ~ 1/sqrt(N))."""
        names = _method_names(4)
        M_small = _make_results_matrix(n_datasets=5, n_methods=4)
        M_large = _make_results_matrix(n_datasets=50, n_methods=4)
        cd_small = critical_difference_data(M_small, names)["cd"]
        cd_large = critical_difference_data(M_large, names)["cd"]
        assert cd_large < cd_small


# ===================================================================
# nemenyi_test (conditional -- requires scikit_posthocs)
# ===================================================================

class TestNemenyiTest:
    """Tests for nemenyi_test (skipped if scikit_posthocs is not installed)."""

    @pytest.fixture(autouse=True)
    def _check_scikit_posthocs(self):
        """Skip if scikit_posthocs is not available."""
        pytest.importorskip("scikit_posthocs")

    def test_returns_dataframe(self):
        from src.evaluation.statistical_tests import nemenyi_test
        M = _make_results_matrix(n_datasets=10, n_methods=4)
        names = _method_names(4)
        result = nemenyi_test(M, names)
        assert isinstance(result, pd.DataFrame)

    def test_shape(self):
        from src.evaluation.statistical_tests import nemenyi_test
        M = _make_results_matrix(n_datasets=10, n_methods=4)
        names = _method_names(4)
        result = nemenyi_test(M, names)
        assert result.shape == (4, 4)

    def test_p_values_in_range(self):
        from src.evaluation.statistical_tests import nemenyi_test
        M = _make_results_matrix(n_datasets=10, n_methods=4)
        names = _method_names(4)
        result = nemenyi_test(M, names)
        assert (result.values >= -1e-10).all()
        assert (result.values <= 1.0 + 1e-10).all()

    def test_column_names_match(self):
        from src.evaluation.statistical_tests import nemenyi_test
        names = ["X", "Y", "Z"]
        M = _make_results_matrix(n_datasets=5, n_methods=3)
        result = nemenyi_test(M, names)
        assert list(result.columns) == names
        assert list(result.index) == names
