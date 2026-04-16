"""
Tests for evaluation metrics and cross-validation pipeline.

Covers:
  - g_mean
  - compute_all_metrics
  - cross_validate_with_oversampling
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.metrics import g_mean, compute_all_metrics, METRIC_NAMES
from src.evaluation.cross_validation import (
    cross_validate_with_oversampling,
    evaluate_pipeline,
)
from src.core.circular_smote import CircularSMOTE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_imbalanced(n_maj=100, n_min=20, n_features=5, seed=42):
    """Simple imbalanced binary dataset for testing."""
    rng = np.random.default_rng(seed)
    X_maj = rng.standard_normal((n_maj, n_features))
    X_min = rng.standard_normal((n_min, n_features)) + 3.0
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_maj), np.ones(n_min)])
    return X, y


# ===================================================================
# g_mean
# ===================================================================

class TestGMean:
    """Tests for the g_mean function."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield g_mean = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        assert pytest.approx(g_mean(y_true, y_pred)) == 1.0

    def test_all_wrong_predictions(self):
        """All-wrong predictions should yield g_mean = 0.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        assert pytest.approx(g_mean(y_true, y_pred)) == 0.0

    def test_known_value(self):
        """Test g_mean with known sensitivity and specificity.

        y_true: 4 zeros, 4 ones
        y_pred: TN=3, FP=1, FN=1, TP=3
        sensitivity = 3/4 = 0.75
        specificity = 3/4 = 0.75
        g_mean = sqrt(0.75 * 0.75) = 0.75
        """
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        expected = np.sqrt(0.75 * 0.75)
        assert pytest.approx(g_mean(y_true, y_pred), abs=1e-10) == expected

    def test_all_predicted_majority(self):
        """If all predictions are majority, sensitivity = 0 so g_mean = 0."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0])
        assert pytest.approx(g_mean(y_true, y_pred)) == 0.0

    def test_all_predicted_minority(self):
        """If all predictions are minority, specificity = 0 so g_mean = 0."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1])
        assert pytest.approx(g_mean(y_true, y_pred)) == 0.0

    def test_returns_float(self):
        """g_mean should return a Python float."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = g_mean(y_true, y_pred)
        assert isinstance(result, float)

    def test_asymmetric_case(self):
        """Test with different sensitivity and specificity values.

        y_true: 5 zeros, 5 ones
        y_pred: TN=5, FP=0, FN=2, TP=3
        sensitivity = 3/5 = 0.6
        specificity = 5/5 = 1.0
        g_mean = sqrt(0.6 * 1.0) = sqrt(0.6)
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        expected = np.sqrt(0.6 * 1.0)
        assert pytest.approx(g_mean(y_true, y_pred), abs=1e-10) == expected


# ===================================================================
# compute_all_metrics
# ===================================================================

class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_returns_all_expected_keys(self):
        """Result dict should contain all metric names."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        result = compute_all_metrics(y_true, y_pred, y_prob)

        expected_keys = {"auc", "g_mean", "f_measure", "balanced_accuracy",
                         "precision", "sensitivity"}
        assert set(result.keys()) == expected_keys

    def test_matches_metric_names_constant(self):
        """All keys in METRIC_NAMES should be present in the result."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = compute_all_metrics(y_true, y_pred)
        for name in METRIC_NAMES:
            assert name in result

    def test_perfect_predictions(self):
        """Perfect predictions should yield 1.0 for most metrics."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        result = compute_all_metrics(y_true, y_pred, y_prob)

        assert pytest.approx(result["g_mean"]) == 1.0
        assert pytest.approx(result["f_measure"]) == 1.0
        assert pytest.approx(result["balanced_accuracy"]) == 1.0
        assert pytest.approx(result["precision"]) == 1.0
        assert pytest.approx(result["sensitivity"]) == 1.0
        assert pytest.approx(result["auc"]) == 1.0

    def test_auc_nan_without_proba(self):
        """AUC should be NaN when y_prob is None."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = compute_all_metrics(y_true, y_pred, y_prob=None)
        assert np.isnan(result["auc"])

    def test_all_values_are_floats(self):
        """All metric values should be Python floats (or NaN)."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = compute_all_metrics(y_true, y_pred)
        for key, val in result.items():
            assert isinstance(val, float), f"Metric '{key}' is not float: {type(val)}"

    def test_sensitivity_is_recall(self):
        """Sensitivity should equal recall of the positive class."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])
        result = compute_all_metrics(y_true, y_pred)
        # TP=2, FN=1 -> sensitivity = 2/3
        assert pytest.approx(result["sensitivity"], abs=1e-10) == 2.0 / 3.0


# ===================================================================
# cross_validate_with_oversampling
# ===================================================================

class TestCrossValidateWithOversampling:
    """Tests for cross_validate_with_oversampling."""

    def test_returns_correct_structure(self):
        """Result should have 'folds', 'median', 'mean', 'std' keys."""
        X, y = _make_imbalanced()
        clf = DecisionTreeClassifier(random_state=42)
        os = CircularSMOTE(random_state=42)
        result = cross_validate_with_oversampling(
            X, y, os, clf, n_folds=3, random_state=42
        )
        assert "folds" in result
        assert "median" in result
        assert "mean" in result
        assert "std" in result

    def test_folds_length_matches_n_folds(self):
        """Number of fold results should equal n_folds."""
        X, y = _make_imbalanced()
        clf = DecisionTreeClassifier(random_state=42)
        os = CircularSMOTE(random_state=42)
        result = cross_validate_with_oversampling(
            X, y, os, clf, n_folds=4, random_state=42
        )
        assert len(result["folds"]) == 4

    def test_each_fold_has_all_metrics(self):
        """Each fold result should contain all metric names."""
        X, y = _make_imbalanced()
        clf = DecisionTreeClassifier(random_state=42)
        os = CircularSMOTE(random_state=42)
        result = cross_validate_with_oversampling(
            X, y, os, clf, n_folds=3, random_state=42
        )
        for fold_metrics in result["folds"]:
            for name in METRIC_NAMES:
                assert name in fold_metrics

    def test_aggregated_metrics_have_all_keys(self):
        """median, mean, std should each contain all metric names."""
        X, y = _make_imbalanced()
        clf = DecisionTreeClassifier(random_state=42)
        result = cross_validate_with_oversampling(
            X, y, None, clf, n_folds=3, random_state=42
        )
        for agg_key in ["median", "mean", "std"]:
            for name in METRIC_NAMES:
                assert name in result[agg_key]

    def test_none_oversampler(self):
        """Passing oversampler=None should work (no oversampling)."""
        X, y = _make_imbalanced()
        clf = DecisionTreeClassifier(random_state=42)
        result = cross_validate_with_oversampling(
            X, y, None, clf, n_folds=3, random_state=42
        )
        assert len(result["folds"]) == 3

    def test_metrics_are_finite(self):
        """All non-AUC metrics in folds should be finite."""
        X, y = _make_imbalanced()
        clf = DecisionTreeClassifier(random_state=42)
        os = CircularSMOTE(random_state=42)
        result = cross_validate_with_oversampling(
            X, y, os, clf, n_folds=3, random_state=42
        )
        for fold_metrics in result["folds"]:
            for name in ["g_mean", "f_measure", "balanced_accuracy",
                         "precision", "sensitivity"]:
                assert np.isfinite(fold_metrics[name])

    def test_mean_is_average_of_folds(self):
        """The 'mean' aggregate should be the actual mean of fold values."""
        X, y = _make_imbalanced()
        clf = DecisionTreeClassifier(random_state=42)
        result = cross_validate_with_oversampling(
            X, y, None, clf, n_folds=3, random_state=42
        )
        for name in ["g_mean", "f_measure", "balanced_accuracy"]:
            fold_values = [f[name] for f in result["folds"]]
            expected_mean = np.nanmean(fold_values)
            assert pytest.approx(result["mean"][name], abs=1e-10) == expected_mean

    def test_classifier_factory_via_evaluate_pipeline(self):
        """evaluate_pipeline should accept factory callables."""
        X, y = _make_imbalanced()
        result = evaluate_pipeline(
            X, y,
            oversampler_factory=lambda: CircularSMOTE(random_state=42),
            classifier_factory=lambda: DecisionTreeClassifier(random_state=42),
            n_folds=3,
            random_state=42,
        )
        assert "folds" in result
        assert len(result["folds"]) == 3

    def test_evaluate_pipeline_no_oversampler(self):
        """evaluate_pipeline with oversampler_factory=None should work."""
        X, y = _make_imbalanced()
        result = evaluate_pipeline(
            X, y,
            oversampler_factory=None,
            classifier_factory=lambda: DecisionTreeClassifier(random_state=42),
            n_folds=3,
        )
        assert "folds" in result
