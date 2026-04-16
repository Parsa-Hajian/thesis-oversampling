"""
Evaluation metrics for imbalanced binary classification.

All metrics are computed on the **minority** (positive) class by convention.
The module exposes individual metric functions as well as a single
:func:`compute_all_metrics` entry point used by the cross-validation loop.

Metrics
-------
* **AUC** -- Area Under the ROC Curve (requires probability estimates).
* **G-Mean** -- Geometric mean of sensitivity (TPR) and specificity (TNR).
* **F-Measure** -- F1 score for the positive class.
* **Balanced Accuracy** -- Average of per-class recall values.
* **Precision** -- Positive predictive value for the positive class.
* **Sensitivity** -- Recall / true positive rate for the positive class.
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---- Individual metrics ---------------------------------------------------

def g_mean(y_true, y_pred):
    """Geometric mean of sensitivity and specificity.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.

    Returns
    -------
    gmean : float
        Value in [0, 1].  Higher is better.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Binary case: cm is 2x2  [[TN, FP], [FN, TP]]
    # For multi-class we still compute the product of per-class recalls.
    per_class_recall = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return float(np.sqrt(np.prod(per_class_recall)))


def auc_score(y_true, y_prob):
    """Area Under the ROC Curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probability of the positive class.

    Returns
    -------
    auc : float
        Value in [0, 1].  Returns ``np.nan`` if *y_prob* is ``None`` or if
        only one class is present in *y_true*.
    """
    if y_prob is None:
        return np.nan
    # Guard against single-class folds (rare but possible with small data).
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


# ---- Aggregate computation -----------------------------------------------

def compute_all_metrics(y_true, y_pred, y_prob=None):
    """Compute the full suite of evaluation metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.
    y_prob : array-like of shape (n_samples,) or None, default=None
        Predicted probability of the positive class (for AUC).

    Returns
    -------
    results : dict[str, float]
        Keys: ``"auc"``, ``"g_mean"``, ``"f_measure"``,
        ``"balanced_accuracy"``, ``"precision"``, ``"sensitivity"``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "auc": auc_score(y_true, y_prob),
        "g_mean": g_mean(y_true, y_pred),
        "f_measure": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# Canonical ordering used in tables and reports.
METRIC_NAMES = [
    "auc",
    "g_mean",
    "f_measure",
    "balanced_accuracy",
    "precision",
    "sensitivity",
]
