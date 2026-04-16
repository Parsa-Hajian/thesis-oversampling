"""
Stratified K-fold cross-validation with oversampling applied **inside** folds.

This is the correct protocol for evaluating oversampling methods: synthetic
samples must only be generated from the training partition so that the test
partition remains untouched and representative of the true data distribution.

The module provides two public functions:

* :func:`cross_validate_with_oversampling` -- evaluate a single
  (oversampler, classifier) pair.
* :func:`evaluate_pipeline` -- convenience wrapper that accepts names rather
  than instances and handles fresh instantiation per fold.
"""

import logging
import warnings
from copy import deepcopy

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import METRIC_NAMES, compute_all_metrics

logger = logging.getLogger(__name__)


def cross_validate_with_oversampling(
    X,
    y,
    oversampler,
    classifier,
    n_folds=5,
    random_state=42,
):
    """Stratified K-fold CV with oversampling restricted to training folds.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Binary target labels.
    oversampler : object with ``fit_resample(X, y)`` method, or ``None``
        The oversampling strategy.  Pass ``None`` to skip oversampling
        (equivalent to the "none" baseline).
    classifier : sklearn-compatible estimator
        Must support ``.fit()``, ``.predict()``, and ideally
        ``.predict_proba()`` (used for AUC).
    n_folds : int, default=5
        Number of stratified folds.
    random_state : int, default=42
        Random seed for the fold splitter.

    Returns
    -------
    results : dict
        ``"folds"``
            List of per-fold metric dicts (length *n_folds*).
        ``"median"``
            Dict mapping each metric name to its median across folds.
        ``"mean"``
            Dict mapping each metric name to its mean across folds.
        ``"std"``
            Dict mapping each metric name to its standard deviation across
            folds.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Oversample ONLY the training data ---
        if oversampler is not None:
            try:
                X_train_res, y_train_res = oversampler.fit_resample(
                    X_train, y_train
                )
            except Exception as exc:
                logger.warning(
                    "Fold %d: oversampling failed (%s). "
                    "Falling back to original training data.",
                    fold_idx,
                    exc,
                )
                X_train_res, y_train_res = X_train, y_train
        else:
            X_train_res, y_train_res = X_train, y_train

        # --- Fresh copy of classifier for this fold ---
        clf = deepcopy(classifier)

        # --- Train ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train_res, y_train_res)

        # --- Predict ---
        y_pred = clf.predict(X_test)

        # Probability estimates (needed for AUC)
        y_prob = None
        if hasattr(clf, "predict_proba"):
            try:
                proba = clf.predict_proba(X_test)
                # Identify the column for the positive (minority) class.
                # Convention: classes_ is sorted, positive class is last.
                y_prob = proba[:, 1] if proba.shape[1] == 2 else None
            except Exception:
                y_prob = None

        fold_metrics = compute_all_metrics(y_test, y_pred, y_prob)
        fold_results.append(fold_metrics)

        logger.debug(
            "Fold %d: %s",
            fold_idx,
            {k: f"{v:.4f}" for k, v in fold_metrics.items()},
        )

    # --- Aggregate across folds ---
    aggregated = _aggregate_fold_results(fold_results)
    aggregated["folds"] = fold_results
    return aggregated


def evaluate_pipeline(
    X,
    y,
    oversampler_factory,
    classifier_factory,
    n_folds=5,
    random_state=42,
):
    """High-level convenience function that accepts *factories* (callables).

    Unlike :func:`cross_validate_with_oversampling`, this function creates
    fresh oversampler and classifier instances so that each fold starts from
    a clean state without needing ``deepcopy``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    oversampler_factory : callable or None
        Zero-argument callable returning an oversampler instance,
        or ``None`` for no oversampling.
    classifier_factory : callable
        Zero-argument callable returning a fresh classifier instance.
    n_folds : int, default=5
    random_state : int, default=42

    Returns
    -------
    results : dict
        Same structure as :func:`cross_validate_with_oversampling`.
    """
    oversampler = oversampler_factory() if oversampler_factory is not None else None
    classifier = classifier_factory()
    return cross_validate_with_oversampling(
        X, y, oversampler, classifier, n_folds=n_folds, random_state=random_state
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _aggregate_fold_results(fold_results):
    """Compute median, mean, and std for each metric across folds.

    Parameters
    ----------
    fold_results : list[dict[str, float]]
        One dict per fold, as returned by :func:`compute_all_metrics`.

    Returns
    -------
    agg : dict
        Keys ``"median"``, ``"mean"``, ``"std"`` each mapping to a
        ``dict[str, float]``.
    """
    arrays = {}
    for metric in METRIC_NAMES:
        values = np.array(
            [fold[metric] for fold in fold_results], dtype=np.float64
        )
        arrays[metric] = values

    median = {m: float(np.nanmedian(v)) for m, v in arrays.items()}
    mean = {m: float(np.nanmean(v)) for m, v in arrays.items()}
    std = {m: float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
           for m, v in arrays.items()}

    return {"median": median, "mean": mean, "std": std}
