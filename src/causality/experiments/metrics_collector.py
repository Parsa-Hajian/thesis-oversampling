"""
metrics_collector.py -- Per-fold metric computation.

All metrics are computed on the *test* fold only.  No oversampling
information leaks from training folds into the test fold.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold


def _gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    return float(np.sqrt(sens * spec))


def compute_metrics(y_true, y_pred, y_prob):
    """Return dict of metric_name -> value for one fold."""
    sens_val = 0.0
    spec_val = 0.0
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens_val = tp / (tp + fn + 1e-9)
        spec_val = tn / (tn + fp + 1e-9)

    return {
        "auc":      roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "f1":       f1_score(y_true, y_pred, zero_division=0),
        "gmean":    _gmean(y_true, y_pred),
        "bal_acc":  balanced_accuracy_score(y_true, y_pred),
        "sens":     sens_val,
        "spec":     spec_val,
    }


# ---------------------------------------------------------------------------
# Leakage-safe cross-validation
# ---------------------------------------------------------------------------

class LeakageSafeCV:
    """
    5-fold stratified CV with oversampling strictly inside training folds.

    Anti-leakage protocol
    ---------------------
    1. Stratified split is computed on the ORIGINAL (pre-oversample) dataset.
    2. For each fold:
       a. Isolate training fold → oversample ONLY the training fold.
       b. Train classifier on the oversampled training fold.
       c. Evaluate on the *original* test fold (never touched by oversampler).
    3. Fit/transform of any scaler/preprocessor is also done inside the fold
       using only training data; test fold is transformed with the
       fold-fitted scaler.
    4. The oversampler is re-instantiated fresh for every fold to prevent
       any state leakage between folds.

    These four rules prevent three common leakage modes:
    * Distribution leakage  — oversampler seeing test-fold statistics
    * Label leakage         — synthetic points derived from test-fold labels
    * Scale leakage         — normalisation statistics computed on full dataset
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits     = n_splits
        self.random_state = random_state

    def evaluate(self, X, y, clf_factory, oversampler_factory=None,
                 scaler_factory=None):
        """
        Parameters
        ----------
        X, y : array-like
            Dataset.
        clf_factory : callable() -> sklearn estimator
            Called fresh for each fold.
        oversampler_factory : callable() -> fit_resample-capable object | None
            Called fresh for each fold.  If None, no oversampling is applied.
        scaler_factory : callable() -> sklearn transformer | None
            Called fresh for each fold.  Fit on train, transform train+test.

        Returns
        -------
        dict : metric_name -> list of per-fold values
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True,
            random_state=self.random_state
        )

        results = {m: [] for m in ("auc", "f1", "gmean", "bal_acc", "sens", "spec")}

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # 1. Scale inside fold (train statistics only)
            if scaler_factory is not None:
                scaler = scaler_factory()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)          # test uses train stats

            # 2. Oversample training fold only
            if oversampler_factory is not None:
                try:
                    sampler = oversampler_factory()
                    X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
                except Exception:
                    pass  # fall back to unsampled fold on error

            # 3. Train on (possibly oversampled) training fold
            clf = clf_factory()
            clf.fit(X_tr, y_tr)

            # 4. Evaluate on ORIGINAL test fold
            y_pred = clf.predict(X_te)
            try:
                y_prob = clf.predict_proba(X_te)[:, 1]
            except AttributeError:
                y_prob = y_pred.astype(float)

            fold_metrics = compute_metrics(y_te, y_pred, y_prob)
            for k, v in fold_metrics.items():
                results[k].append(v)

        return results

    def mean_metrics(self, X, y, clf_factory, oversampler_factory=None,
                     scaler_factory=None):
        """Same as evaluate() but returns mean across folds."""
        raw = self.evaluate(X, y, clf_factory, oversampler_factory, scaler_factory)
        return {k: float(np.mean(v)) for k, v in raw.items()}
