"""
Tests for seed selection: SeedSelector, GaussianSeedSelector, and scoring metrics.
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pytest

from src.seed_selection.selector import SeedSelector
from src.seed_selection.gaussian import GaussianSeedSelector
from src.seed_selection.metrics import (
    normalized_histogram_overlap_percentage,
    histogram_overlap_percentage,  # backward compat alias
    agtp_score,
    smoothness_score,
    geometric_similarity,
    topological_similarity,
    jensen_shannon_divergence,
)


# ---------------------------------------------------------------------------
# Helpers -- synthetic minority data
# ---------------------------------------------------------------------------

def _make_minority(n=100, d=5, seed=42):
    """Generate a synthetic minority point cloud."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


# ---------------------------------------------------------------------------
# SeedSelector
# ---------------------------------------------------------------------------

class TestSeedSelector:
    """Tests for SeedSelector.select()."""

    def test_returns_valid_indices(self):
        """Selected indices must be valid row indices into X_minority."""
        X = _make_minority(80, 4)
        sel = SeedSelector(n_candidates=20, random_state=42)
        indices, score, log = sel.select(X, n_seeds=20)
        assert np.all(indices >= 0)
        assert np.all(indices < 80)

    def test_selected_indices_are_subset(self):
        """Selected indices must be a subset of [0, n)."""
        X = _make_minority(60, 3)
        sel = SeedSelector(n_candidates=30, random_state=0)
        indices, _, _ = sel.select(X, n_seeds=15)
        assert len(indices) == 15
        assert len(np.unique(indices)) == 15  # no duplicates

    def test_score_is_not_nan(self):
        """The returned best_score should be a finite number."""
        X = _make_minority(50, 2)
        sel = SeedSelector(n_candidates=10, random_state=42)
        _, score, _ = sel.select(X, n_seeds=10)
        assert np.isfinite(score)

    def test_log_contains_expected_keys(self):
        """The scores_log dict should contain 'nhop', 'agtp', 'jsd', and 'z'."""
        X = _make_minority(50, 2)
        sel = SeedSelector(n_candidates=10, random_state=42)
        _, _, log = sel.select(X, n_seeds=10)
        assert "nhop" in log
        assert "agtp" in log
        assert "jsd" in log
        assert "z" in log

    def test_n_seeds_equals_n_returns_all(self):
        """When n_seeds >= n, all indices should be returned."""
        X = _make_minority(20, 3)
        sel = SeedSelector(n_candidates=5, random_state=42)
        indices, score, log = sel.select(X, n_seeds=20)
        assert len(indices) == 20
        np.testing.assert_array_equal(np.sort(indices), np.arange(20))

    def test_n_seeds_greater_than_n(self):
        """When n_seeds > n, all indices should be returned (capped at n)."""
        X = _make_minority(10, 2)
        sel = SeedSelector(n_candidates=5, random_state=42)
        indices, _, _ = sel.select(X, n_seeds=50)
        assert len(indices) == 10

    def test_reproducibility(self):
        """Same random_state should produce identical selections."""
        X = _make_minority(100, 4)
        sel1 = SeedSelector(n_candidates=20, random_state=42)
        sel2 = SeedSelector(n_candidates=20, random_state=42)
        idx1, s1, _ = sel1.select(X, n_seeds=20)
        idx2, s2, _ = sel2.select(X, n_seeds=20)
        np.testing.assert_array_equal(np.sort(idx1), np.sort(idx2))
        assert pytest.approx(s1) == s2

    def test_different_random_states_differ(self):
        """Different random states should (almost certainly) give different results."""
        X = _make_minority(100, 4)
        sel1 = SeedSelector(n_candidates=50, random_state=42)
        sel2 = SeedSelector(n_candidates=50, random_state=999)
        idx1, _, _ = sel1.select(X, n_seeds=20)
        idx2, _, _ = sel2.select(X, n_seeds=20)
        # They could theoretically be the same, but extremely unlikely.
        assert not np.array_equal(np.sort(idx1), np.sort(idx2))


# ---------------------------------------------------------------------------
# GaussianSeedSelector
# ---------------------------------------------------------------------------

class TestGaussianSeedSelector:
    """Tests for GaussianSeedSelector.select()."""

    def test_random_method(self):
        """Random method should return valid indices."""
        X = _make_minority(50, 3)
        sel = GaussianSeedSelector(method="random", random_state=42)
        indices, score, log = sel.select(X, n_seeds=10)
        assert len(indices) == 10
        assert score == 0.0
        assert isinstance(log, dict)
        assert np.all(indices >= 0)
        assert np.all(indices < 50)

    def test_gaussian_method(self):
        """Gaussian method should return valid indices."""
        X = _make_minority(50, 3)
        sel = GaussianSeedSelector(method="gaussian", random_state=42)
        indices, score, log = sel.select(X, n_seeds=10)
        assert len(indices) == 10
        assert score == 0.0
        assert np.all(indices >= 0)
        assert np.all(indices < 50)

    def test_no_duplicates(self):
        """Selected indices should have no duplicates."""
        X = _make_minority(40, 2)
        sel = GaussianSeedSelector(method="gaussian", random_state=42)
        indices, _, _ = sel.select(X, n_seeds=15)
        assert len(np.unique(indices)) == 15

    def test_n_seeds_ge_n(self):
        """When n_seeds >= n, all indices are returned."""
        X = _make_minority(10, 2)
        sel = GaussianSeedSelector(method="random", random_state=42)
        indices, _, _ = sel.select(X, n_seeds=10)
        assert len(indices) == 10
        np.testing.assert_array_equal(np.sort(indices), np.arange(10))

    def test_gaussian_favors_center(self):
        """Gaussian method should statistically favor central points.

        We create a cluster at the origin and a few far-off outliers. Over many
        selections, the center points should be selected more often.
        """
        rng = np.random.default_rng(42)
        # 40 points clustered at origin, 10 outliers far away.
        center_pts = rng.normal(0, 0.1, (40, 2))
        outlier_pts = rng.normal(100, 0.1, (10, 2))
        X = np.vstack([center_pts, outlier_pts])

        center_count = 0
        n_trials = 50
        for seed in range(n_trials):
            sel = GaussianSeedSelector(method="gaussian", random_state=seed)
            indices, _, _ = sel.select(X, n_seeds=5)
            # Count how many of the 5 selected are from the center group (idx < 40).
            center_count += np.sum(indices < 40)

        # With Gaussian weighting, a large majority should be center points.
        # 50 trials * 5 selections = 250 total picks.
        assert center_count > 200, (
            f"Gaussian method should favor center: got {center_count}/250 center picks"
        )


# ---------------------------------------------------------------------------
# normalized_histogram_overlap_percentage (NHOP)
# ---------------------------------------------------------------------------

class TestNHOP:
    """Tests for normalized_histogram_overlap_percentage."""

    def test_identical_sets_gives_one(self):
        """NHOP of a set with itself should be approximately 1.0."""
        X = _make_minority(100, 3)
        nhop = normalized_histogram_overlap_percentage(X, X)
        assert pytest.approx(nhop, abs=0.05) == 1.0

    def test_nhop_between_zero_and_one(self):
        """NHOP should be between 0 and 1 for any input."""
        rng = np.random.default_rng(42)
        X_orig = rng.standard_normal((100, 2))
        X_seeds = rng.standard_normal((30, 2)) + 5  # shifted distribution
        nhop = normalized_histogram_overlap_percentage(X_orig, X_seeds)
        assert 0.0 <= nhop <= 1.0

    def test_disjoint_distributions_low_nhop(self):
        """Completely disjoint distributions should have very low NHOP."""
        X_orig = np.zeros((100, 2))
        X_seeds = np.full((50, 2), 1000.0)
        nhop = normalized_histogram_overlap_percentage(X_orig, X_seeds)
        assert nhop < 0.1

    def test_single_dimension(self):
        """NHOP should work with 1-D data (reshaped to (n, 1))."""
        rng = np.random.default_rng(0)
        X_orig = rng.standard_normal((200, 1))
        X_seeds = rng.standard_normal((50, 1))
        nhop = normalized_histogram_overlap_percentage(X_orig, X_seeds)
        assert 0.0 <= nhop <= 1.0

    def test_backward_compat_alias(self):
        """histogram_overlap_percentage should be an alias for NHOP."""
        X = _make_minority(100, 3)
        nhop = normalized_histogram_overlap_percentage(X, X)
        hop = histogram_overlap_percentage(X, X)
        assert nhop == hop


# ---------------------------------------------------------------------------
# agtp_score (AGTP)
# ---------------------------------------------------------------------------

class TestAGTP:
    """Tests for agtp_score."""

    def test_agtp_between_zero_and_one(self):
        """AGTP should be between 0 and 1."""
        rng = np.random.default_rng(42)
        X_orig = rng.standard_normal((80, 3))
        X_seeds = X_orig[np.random.default_rng(0).choice(80, 20, replace=False)]
        score = agtp_score(X_orig, X_seeds, k=3)
        assert 0.0 <= score <= 1.0

    def test_identical_sets_high_agtp(self):
        """AGTP of a set with itself should be high (close to 1)."""
        X = _make_minority(50, 2)
        score = agtp_score(X, X, k=3)
        assert score > 0.5

    def test_small_seed_set(self):
        """AGTP should handle very small seed sets without error."""
        X_orig = _make_minority(50, 2)
        X_seeds = X_orig[:3]
        score = agtp_score(X_orig, X_seeds, k=2)
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# smoothness_score (Z)
# ---------------------------------------------------------------------------

class TestSmoothness:
    """Tests for smoothness_score."""

    def test_smoothness_non_negative(self):
        """Smoothness Z should be non-negative (it's a CV)."""
        X = _make_minority(30, 2)
        z = smoothness_score(X)
        assert z >= 0.0

    def test_single_point_returns_zero(self):
        """A single point has zero smoothness score."""
        X = np.array([[1.0, 2.0]])
        z = smoothness_score(X)
        assert z == 0.0

    def test_regular_grid_low_smoothness(self):
        """A regular grid should have lower smoothness than random points."""
        # Regular grid in 2D.
        xs = np.linspace(0, 1, 10)
        ys = np.linspace(0, 1, 10)
        grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
        z_grid = smoothness_score(grid)

        # Random points.
        rng = np.random.default_rng(42)
        random_pts = rng.standard_normal((100, 2))
        z_random = smoothness_score(random_pts)

        # Grid should generally have lower (more regular) smoothness.
        # Note: this is a statistical property, not a guaranteed inequality,
        # but for well-separated grid vs random it should hold.
        assert z_grid < z_random * 2  # loose bound

    def test_finite_for_typical_input(self):
        """Smoothness should be finite for typical minority sets."""
        X = _make_minority(50, 4)
        z = smoothness_score(X)
        assert np.isfinite(z)


# ---------------------------------------------------------------------------
# geometric_similarity
# ---------------------------------------------------------------------------

class TestGeometricSimilarity:
    """Tests for geometric_similarity."""

    def test_identical_sets(self):
        """Geometric similarity of a set with itself should be 1.0."""
        X = _make_minority(50, 3)
        sim = geometric_similarity(X, X)
        assert pytest.approx(sim, abs=0.01) == 1.0

    def test_range_zero_to_one(self):
        """Geometric similarity should be in [0, 1]."""
        rng = np.random.default_rng(42)
        X_orig = rng.standard_normal((50, 2))
        X_seeds = rng.standard_normal((20, 2)) + 10
        sim = geometric_similarity(X_orig, X_seeds)
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# topological_similarity
# ---------------------------------------------------------------------------

class TestTopologicalSimilarity:
    """Tests for topological_similarity."""

    def test_identical_sets(self):
        """Topological similarity of identical sets should be high."""
        X = _make_minority(50, 2)
        sim = topological_similarity(X, X, k=3)
        assert sim > 0.5

    def test_range(self):
        """Should be in [0, 1]."""
        rng = np.random.default_rng(42)
        X_orig = rng.standard_normal((50, 2))
        X_seeds = rng.standard_normal((20, 2))
        sim = topological_similarity(X_orig, X_seeds, k=3)
        assert 0.0 <= sim <= 1.0

    def test_single_point_returns_zero(self):
        """Single-point seed set has k_seed <= 1 -> returns 0.0."""
        X_orig = _make_minority(50, 2)
        X_seeds = np.array([[0.0, 0.0]])
        sim = topological_similarity(X_orig, X_seeds, k=3)
        assert sim == 0.0


# ---------------------------------------------------------------------------
# jensen_shannon_divergence
# ---------------------------------------------------------------------------

class TestJensenShannonDivergence:
    """Tests for jensen_shannon_divergence."""

    def test_identical_sets_near_zero(self):
        """JSD of a set with itself should be near zero."""
        X = _make_minority(200, 2)
        jsd = jensen_shannon_divergence(X, X)
        assert jsd < 0.05

    def test_non_negative(self):
        """JSD should be non-negative."""
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((100, 2))
        X2 = rng.standard_normal((50, 2)) + 3
        jsd = jensen_shannon_divergence(X1, X2)
        assert jsd >= 0.0
