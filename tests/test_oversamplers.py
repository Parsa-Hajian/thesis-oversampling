"""
Tests for all four oversampler implementations:
  - CircularSMOTE
  - GravityVonMises
  - LocalRegions
  - LayeredSegmentalOversampler
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pytest

from src.core.circular_smote import CircularSMOTE
from src.core.gravity_vonmises import GravityVonMises
from src.core.local_regions import LocalRegions
from src.core.layered_segmental import LayeredSegmentalOversampler


# ---------------------------------------------------------------------------
# Shared test-data factories
# ---------------------------------------------------------------------------

def _make_imbalanced(n_maj=100, n_min=20, n_features=2, seed=42):
    """Create a simple imbalanced binary dataset.

    Majority class (label 0) is centred at the origin; minority class (label 1)
    is centred at (3, 3, ...).
    """
    rng = np.random.default_rng(seed)
    X_maj = rng.standard_normal((n_maj, n_features))
    X_min = rng.standard_normal((n_min, n_features)) + 3.0
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_maj), np.ones(n_min)])
    return X, y


def _make_small_imbalanced(n_features=2, seed=42):
    """Very small imbalanced dataset for edge-case testing."""
    rng = np.random.default_rng(seed)
    X_maj = rng.standard_normal((10, n_features))
    X_min = rng.standard_normal((3, n_features)) + 3.0
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(10), np.ones(3)])
    return X, y


# ---------------------------------------------------------------------------
# Parameterised fixture for all oversamplers
# ---------------------------------------------------------------------------

def _all_oversamplers():
    """Return a list of (name, instance) for the four oversamplers."""
    return [
        ("CircularSMOTE", CircularSMOTE(random_state=42)),
        ("GravityVonMises", GravityVonMises(K=2, random_state=42)),
        ("LocalRegions", LocalRegions(
            N_min=2, certainty_threshold=0.0, k_seed=3, random_state=42
        )),
        ("LayeredSegmental", LayeredSegmentalOversampler(random_state=42)),
    ]


# ===================================================================
# Common tests applied to ALL oversamplers
# ===================================================================

class TestAllOversamplersCommon:
    """Tests that apply identically to every oversampler."""

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_output_shape_increases(self, name, oversampler):
        """Output dataset must be larger than the input (new rows added)."""
        X, y = _make_imbalanced()
        X_res, y_res = oversampler.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0], (
            f"{name}: expected more rows after oversampling"
        )
        assert X_res.shape[1] == X.shape[1], (
            f"{name}: feature dimension should not change"
        )

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_labels_correct(self, name, oversampler):
        """All synthetic samples should carry the minority label."""
        X, y = _make_imbalanced()
        n_original = len(y)
        X_res, y_res = oversampler.fit_resample(X, y)
        synthetic_labels = y_res[n_original:]
        assert np.all(synthetic_labels == 1), (
            f"{name}: synthetic labels should all be the minority class (1)"
        )

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_y_length_matches_x(self, name, oversampler):
        """X_res and y_res must have the same number of rows."""
        X, y = _make_imbalanced()
        X_res, y_res = oversampler.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_balanced_output(self, name, oversampler):
        """With sampling_strategy=1.0, minority count should equal majority count."""
        X, y = _make_imbalanced(n_maj=80, n_min=20)
        X_res, y_res = oversampler.fit_resample(X, y)
        n_maj_out = np.sum(y_res == 0)
        n_min_out = np.sum(y_res == 1)
        assert n_min_out == n_maj_out, (
            f"{name}: expected balanced output, got maj={n_maj_out} min={n_min_out}"
        )

    @pytest.mark.parametrize("n_features", [2, 5, 10])
    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_different_feature_dimensions(self, name, oversampler, n_features):
        """Oversampler should work with 2D, 5D, and 10D data."""
        X, y = _make_imbalanced(n_features=n_features)
        X_res, y_res = oversampler.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]
        assert X_res.shape[1] == n_features

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_small_dataset(self, name, oversampler):
        """Should handle very small datasets without crashing."""
        X, y = _make_small_imbalanced()
        X_res, y_res = oversampler.fit_resample(X, y)
        assert X_res.shape[0] >= X.shape[0]
        assert y_res.shape[0] == X_res.shape[0]

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_original_data_preserved(self, name, oversampler):
        """The first n rows of the resampled set should be the original data."""
        X, y = _make_imbalanced()
        X_res, y_res = oversampler.fit_resample(X, y)
        np.testing.assert_array_equal(X_res[:len(X)], X)
        np.testing.assert_array_equal(y_res[:len(y)], y)

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_absolute_count_strategy(self, name, oversampler):
        """sampling_strategy as an integer should produce that many synthetic samples."""
        X, y = _make_imbalanced(n_maj=80, n_min=20)
        # Create a fresh copy with integer strategy.
        params = oversampler.get_params()
        params["sampling_strategy"] = 10
        cls = type(oversampler)
        os_int = cls(**params)
        X_res, y_res = os_int.fit_resample(X, y)
        n_synthetic = X_res.shape[0] - X.shape[0]
        assert n_synthetic == 10, (
            f"{name}: expected 10 synthetic, got {n_synthetic}"
        )


# ===================================================================
# Reproducibility
# ===================================================================

class TestReproducibility:
    """Same random_state should produce identical outputs."""

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_reproducible_output(self, name, oversampler):
        """Two calls with the same random_state should give identical results."""
        X, y = _make_imbalanced()
        params = oversampler.get_params()
        cls = type(oversampler)

        os1 = cls(**params)
        os2 = cls(**params)

        X_res1, y_res1 = os1.fit_resample(X, y)
        X_res2, y_res2 = os2.fit_resample(X, y)

        np.testing.assert_array_equal(X_res1, X_res2)
        np.testing.assert_array_equal(y_res1, y_res2)


# ===================================================================
# CircularSMOTE specific
# ===================================================================

class TestCircularSMOTE:
    """CircularSMOTE-specific tests."""

    def test_default_k(self):
        """Default k should be 5."""
        cs = CircularSMOTE()
        assert cs.k == 5

    def test_custom_k(self):
        """Custom k parameter should be used."""
        cs = CircularSMOTE(k=3, random_state=42)
        X, y = _make_imbalanced()
        X_res, y_res = cs.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]

    def test_get_params(self):
        """get_params should include 'k' and 'chunk_size'."""
        cs = CircularSMOTE(k=7, chunk_size=5000)
        params = cs.get_params()
        assert params["k"] == 7
        assert params["chunk_size"] == 5000

    def test_chunk_processing(self):
        """Small chunk_size should still produce the correct total."""
        cs = CircularSMOTE(sampling_strategy=1.0, chunk_size=5, random_state=42)
        X, y = _make_imbalanced(n_maj=50, n_min=10)
        X_res, y_res = cs.fit_resample(X, y)
        n_min_out = np.sum(y_res == 1)
        n_maj_out = np.sum(y_res == 0)
        assert n_min_out == n_maj_out

    def test_single_minority_point(self):
        """With only 1 minority point, should still produce output (with jitter)."""
        rng = np.random.default_rng(42)
        X_maj = rng.standard_normal((20, 2))
        X_min = np.array([[5.0, 5.0]])
        X = np.vstack([X_maj, X_min])
        y = np.concatenate([np.zeros(20), np.ones(1)])

        cs = CircularSMOTE(random_state=42)
        X_res, y_res = cs.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]


# ===================================================================
# GravityVonMises specific
# ===================================================================

class TestGravityVonMises:
    """GravityVonMises-specific tests."""

    def test_cross_cluster_false(self):
        """cross_cluster=False should work and produce valid output."""
        gv = GravityVonMises(K=2, cross_cluster=False, random_state=42)
        X, y = _make_imbalanced(n_min=30)
        X_res, y_res = gv.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]
        assert np.all(y_res[len(y):] == 1)

    def test_cross_cluster_true(self):
        """cross_cluster=True should work and produce valid output."""
        gv = GravityVonMises(K=2, cross_cluster=True, random_state=42)
        X, y = _make_imbalanced(n_min=30)
        X_res, y_res = gv.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]
        assert np.all(y_res[len(y):] == 1)

    def test_kmeans_clustering(self):
        """clustering_method='kmeans' should work."""
        gv = GravityVonMises(
            K=3, clustering_method="kmeans", random_state=42
        )
        X, y = _make_imbalanced(n_min=30)
        X_res, _ = gv.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]

    def test_hac_clustering(self):
        """clustering_method='hac' should work."""
        gv = GravityVonMises(
            K=3, clustering_method="hac", random_state=42
        )
        X, y = _make_imbalanced(n_min=30)
        X_res, _ = gv.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]

    def test_kappa_max_effect(self):
        """Different kappa_max values should not crash but may give different results."""
        for kmax in [0.1, 5.0, 50.0]:
            gv = GravityVonMises(K=2, kappa_max=kmax, random_state=42)
            X, y = _make_imbalanced(n_min=30)
            X_res, _ = gv.fit_resample(X, y)
            assert X_res.shape[0] > X.shape[0]

    def test_get_params(self):
        """get_params should include all GravityVonMises-specific parameters."""
        gv = GravityVonMises(
            K=4, k_nn=3, k_seed=7, kappa_max=15.0, gamma=1.5,
            alpha=0.5, clustering_method="hac", cross_cluster=True,
            denoise_method=None, chunk_size=1000,
        )
        params = gv.get_params()
        assert params["K"] == 4
        assert params["k_nn"] == 3
        assert params["k_seed"] == 7
        assert params["kappa_max"] == 15.0
        assert params["gamma"] == 1.5
        assert params["alpha"] == 0.5
        assert params["clustering_method"] == "hac"
        assert params["cross_cluster"] is True
        assert params["denoise_method"] is None
        assert params["chunk_size"] == 1000

    def test_single_cluster(self):
        """K=1 should effectively skip clustering (all labels 0)."""
        gv = GravityVonMises(K=1, random_state=42)
        X, y = _make_imbalanced(n_min=20)
        X_res, y_res = gv.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]

    def test_high_dimensional(self):
        """Should work with high-dimensional data (PCA is used internally)."""
        gv = GravityVonMises(K=2, random_state=42)
        X, y = _make_imbalanced(n_features=20, n_min=30)
        X_res, y_res = gv.fit_resample(X, y)
        assert X_res.shape[1] == 20


# ===================================================================
# LocalRegions specific
# ===================================================================

class TestLocalRegions:
    """LocalRegions-specific tests."""

    def test_certainty_threshold_zero_accepts_all(self):
        """certainty_threshold=0.0 should disable the certainty check."""
        lr = LocalRegions(
            certainty_threshold=0.0, N_min=2, k_seed=3, random_state=42
        )
        X, y = _make_imbalanced(n_min=30)
        X_res, y_res = lr.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]

    def test_high_certainty_threshold(self):
        """High certainty_threshold may reject circles but should still produce output
        (via the fallback)."""
        lr = LocalRegions(
            certainty_threshold=0.99, N_min=2, k_seed=3, random_state=42
        )
        X, y = _make_imbalanced(n_min=30)
        X_res, y_res = lr.fit_resample(X, y)
        # Should still produce output even if all circles are rejected (fallback).
        assert X_res.shape[0] >= X.shape[0]
        assert y_res.shape[0] == X_res.shape[0]

    def test_n_min_parameter(self):
        """Small N_min should allow more circles to be eligible."""
        lr_low = LocalRegions(
            N_min=2, certainty_threshold=0.0, k_seed=5, random_state=42
        )
        lr_high = LocalRegions(
            N_min=50, certainty_threshold=0.0, k_seed=5, random_state=42
        )
        X, y = _make_imbalanced(n_min=20)
        X_res_low, _ = lr_low.fit_resample(X, y)
        X_res_high, _ = lr_high.fit_resample(X, y)
        # Both should produce output. N_min=50 likely rejects all circles -> fallback.
        assert X_res_low.shape[0] >= X.shape[0]
        assert X_res_high.shape[0] >= X.shape[0]

    def test_get_params(self):
        """get_params should include all LocalRegions-specific parameters."""
        lr = LocalRegions(
            k_seed=8, N_min=5, local_k_max=4, min_points_per_cluster=10,
            beta=2.0, certainty_threshold=0.9, clustering_method="kmeans",
            denoise_method=None, chunk_size=5000,
        )
        params = lr.get_params()
        assert params["k_seed"] == 8
        assert params["N_min"] == 5
        assert params["local_k_max"] == 4
        assert params["beta"] == 2.0
        assert params["certainty_threshold"] == 0.9

    def test_single_minority_point_fallback(self):
        """With only 1 minority point, should produce output via fallback."""
        rng = np.random.default_rng(42)
        X_maj = rng.standard_normal((20, 2))
        X_min = np.array([[5.0, 5.0]])
        X = np.vstack([X_maj, X_min])
        y = np.concatenate([np.zeros(20), np.ones(1)])

        lr = LocalRegions(N_min=1, certainty_threshold=0.0, k_seed=3,
                          random_state=42)
        X_res, y_res = lr.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]


# ===================================================================
# LayeredSegmentalOversampler specific
# ===================================================================

class TestLayeredSegmental:
    """LayeredSegmentalOversampler-specific tests."""

    def test_cluster_based_false(self):
        """cluster_based=False (generic mode) should produce valid output."""
        ls = LayeredSegmentalOversampler(cluster_based=False, random_state=42)
        X, y = _make_imbalanced(n_min=30)
        X_res, y_res = ls.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]
        assert np.all(y_res[len(y):] == 1)

    def test_cluster_based_true(self):
        """cluster_based=True should produce valid output."""
        ls = LayeredSegmentalOversampler(
            cluster_based=True, K=3, random_state=42
        )
        X, y = _make_imbalanced(n_min=30)
        X_res, y_res = ls.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]
        assert np.all(y_res[len(y):] == 1)

    def test_cluster_based_with_hac(self):
        """cluster_based=True with clustering_method='hac' should work."""
        ls = LayeredSegmentalOversampler(
            cluster_based=True, K=2, clustering_method="hac", random_state=42
        )
        X, y = _make_imbalanced(n_min=30)
        X_res, _ = ls.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]

    def test_outlier_removal(self):
        """use_outlier_removal=True should work (it's the default)."""
        ls = LayeredSegmentalOversampler(
            use_outlier_removal=True, outlier_keep_frac=0.8, random_state=42
        )
        X, y = _make_imbalanced(n_min=30)
        X_res, _ = ls.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]

    def test_no_outlier_removal(self):
        """use_outlier_removal=False should also work."""
        ls = LayeredSegmentalOversampler(
            use_outlier_removal=False, random_state=42
        )
        X, y = _make_imbalanced(n_min=30)
        X_res, _ = ls.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]

    def test_n_layers_parameter(self):
        """Different n_layers values should all work."""
        for n_lay in [5, 30, 100]:
            ls = LayeredSegmentalOversampler(n_layers=n_lay, random_state=42)
            X, y = _make_imbalanced(n_min=20)
            X_res, _ = ls.fit_resample(X, y)
            assert X_res.shape[0] > X.shape[0]

    def test_sigma_and_ang_std(self):
        """Different sigma and ang_std parameters should work."""
        ls = LayeredSegmentalOversampler(
            sigma=0.1, ang_std=0.2, random_state=42
        )
        X, y = _make_imbalanced(n_min=30)
        X_res, _ = ls.fit_resample(X, y)
        assert X_res.shape[0] > X.shape[0]

    def test_get_params(self):
        """get_params should include all LayeredSegmental-specific parameters."""
        ls = LayeredSegmentalOversampler(
            n_layers=40, sigma=0.05, ang_std=0.1,
            use_outlier_removal=False, outlier_keep_frac=0.9,
            cluster_based=True, K=4, clustering_method="hac",
            denoise_method=None,
        )
        params = ls.get_params()
        assert params["n_layers"] == 40
        assert params["sigma"] == 0.05
        assert params["ang_std"] == 0.1
        assert params["use_outlier_removal"] is False
        assert params["outlier_keep_frac"] == 0.9
        assert params["cluster_based"] is True
        assert params["K"] == 4
        assert params["clustering_method"] == "hac"
        assert params["denoise_method"] is None

    def test_balanced_cluster_based(self):
        """cluster_based=True with sampling_strategy=1.0 should balance classes."""
        ls = LayeredSegmentalOversampler(
            cluster_based=True, K=2, random_state=42, sampling_strategy=1.0
        )
        X, y = _make_imbalanced(n_maj=60, n_min=15)
        X_res, y_res = ls.fit_resample(X, y)
        n_maj_out = np.sum(y_res == 0)
        n_min_out = np.sum(y_res == 1)
        assert n_min_out == n_maj_out


# ===================================================================
# Already-balanced data (n_synth == 0)
# ===================================================================

class TestAlreadyBalanced:
    """When data is already balanced, fit_resample should return copies unchanged."""

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_no_oversampling_when_balanced(self, name, oversampler):
        """If minority count equals majority count, no synthesis should happen."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))
        y = np.concatenate([np.zeros(50), np.ones(50)])
        X_res, y_res = oversampler.fit_resample(X, y)
        assert X_res.shape[0] == X.shape[0], (
            f"{name}: balanced data should not generate synthetic samples"
        )
        np.testing.assert_array_equal(y_res, y)


# ===================================================================
# BaseOversampler contract
# ===================================================================

class TestBaseContract:
    """Tests for BaseOversampler introspection methods."""

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_repr(self, name, oversampler):
        """__repr__ should include the class name."""
        r = repr(oversampler)
        assert type(oversampler).__name__ in r

    @pytest.mark.parametrize("name,oversampler", _all_oversamplers(),
                             ids=[t[0] for t in _all_oversamplers()])
    def test_get_params_has_base_keys(self, name, oversampler):
        """get_params should always include the base keys."""
        params = oversampler.get_params()
        assert "sampling_strategy" in params
        assert "minority_label" in params
        assert "random_state" in params
