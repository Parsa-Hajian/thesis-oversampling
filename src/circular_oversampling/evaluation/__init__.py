"""Evaluation pipeline: classifiers, metrics, cross-validation, and statistical tests."""

from src.evaluation.classifiers import get_all_classifiers, get_classifier
from src.evaluation.cross_validation import (
    cross_validate_with_oversampling,
    evaluate_pipeline,
)
from src.evaluation.metrics import METRIC_NAMES, compute_all_metrics, g_mean
from src.evaluation.statistical_tests import (
    critical_difference_data,
    friedman_test,
    holms_posthoc,
    nemenyi_test,
)

__all__ = [
    "get_classifier",
    "get_all_classifiers",
    "compute_all_metrics",
    "g_mean",
    "METRIC_NAMES",
    "cross_validate_with_oversampling",
    "evaluate_pipeline",
    "friedman_test",
    "holms_posthoc",
    "nemenyi_test",
    "critical_difference_data",
]
