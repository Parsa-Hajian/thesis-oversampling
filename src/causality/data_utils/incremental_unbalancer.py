"""
incremental_unbalancer.py -- NHOP-guided incremental minority removal.

At each step we remove the minority instance whose absence maximises
the NHOP score of the remaining set relative to the full original
minority (greedy preservation of distributional shape).

This isolates the *imbalance* effect from arbitrary distributional shift:
the surviving minority class always best-represents the original
marginal distributions.
"""

import sys
import os
import numpy as np

# Allow importing from circular-oversampling project
_CO_ROOT = os.path.expanduser("~/Desktop/circular-oversampling")
if _CO_ROOT not in sys.path:
    sys.path.insert(0, _CO_ROOT)

try:
    from src.seed_selection.metrics import nhop_score
    _HAS_NHOP = True
except ImportError:
    _HAS_NHOP = False


def _nhop_score_simple(X_ref: np.ndarray, X_cand: np.ndarray,
                       n_bins: int = 20) -> float:
    """
    Fallback NHOP implementation when circular-oversampling is unavailable.
    Returns mean marginal histogram overlap across all features after
    joint min-max normalisation.
    """
    scores = []
    for j in range(X_ref.shape[1]):
        lo = min(X_ref[:, j].min(), X_cand[:, j].min())
        hi = max(X_ref[:, j].max(), X_cand[:, j].max())
        if hi == lo:
            scores.append(1.0)
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        h_ref, _ = np.histogram(X_ref[:, j], bins=edges, density=False)
        h_cnd, _ = np.histogram(X_cand[:, j], bins=edges, density=False)
        h_ref = h_ref / h_ref.sum() if h_ref.sum() > 0 else h_ref
        h_cnd = h_cnd / h_cnd.sum() if h_cnd.sum() > 0 else h_cnd
        scores.append(np.minimum(h_ref, h_cnd).sum())
    return float(np.mean(scores))


def _score(X_ref: np.ndarray, X_cand: np.ndarray) -> float:
    if _HAS_NHOP:
        try:
            return float(nhop_score(X_ref, X_cand))
        except Exception:
            pass
    return _nhop_score_simple(X_ref, X_cand)


class IncrementalUnbalancer:
    """
    Generates a sequence of progressively imbalanced datasets by greedily
    removing minority instances that are *least informative* (whose removal
    minimises distributional shift relative to the full minority set).

    Parameters
    ----------
    step_frac : float
        Fraction of original minority class to remove per step (default 0.01).
    min_frac : float
        Stop when minority fraction of the original size drops to this value
        (default 0.20, i.e. 80% removed).
    random_state : int
        Random seed for tie-breaking.
    """

    def __init__(self,
                 step_frac: float = 0.01,
                 min_frac: float = 0.20,
                 random_state: int = 42):
        self.step_frac = step_frac
        self.min_frac  = min_frac
        self.random_state = random_state

    def degradation_sequence(self, X: np.ndarray, y: np.ndarray):
        """
        Yield (X_step, y_step, step_idx, removal_frac) tuples from the
        fully-balanced starting point down to min_frac minority remaining.

        step_idx=0 is the original balanced dataset.
        """
        rng = np.random.default_rng(self.random_state)

        # Identify minority class
        counts = np.bincount(y.astype(int))
        min_label = int(np.argmin(counts))
        maj_label = int(np.argmax(counts))

        X_maj = X[y == maj_label]
        X_min = X[y == min_label].copy()
        X_ref = X_min.copy()          # original minority — kept fixed for scoring

        n_orig     = len(X_min)
        n_per_step = max(1, int(np.round(n_orig * self.step_frac)))
        n_stop     = max(2, int(np.round(n_orig * self.min_frac)))

        step = 0
        removal_frac = 0.0

        # Step 0: original balanced dataset
        X_cur = np.vstack([X_maj, X_min])
        y_cur = np.concatenate([
            np.full(len(X_maj), maj_label, dtype=int),
            np.full(len(X_min), min_label, dtype=int),
        ])
        yield X_cur, y_cur, step, removal_frac

        while len(X_min) - n_per_step >= n_stop:
            # Greedy: remove the instance whose absence maximises NHOP(remaining, ref)
            best_score  = -1.0
            best_idx    = None

            # For speed, evaluate a random subset of candidates when n_min is large
            n_min = len(X_min)
            candidates = rng.choice(n_min, size=min(n_min, 50), replace=False)

            for i in candidates:
                X_trial = np.delete(X_min, i, axis=0)
                if len(X_trial) < 2:
                    continue
                s = _score(X_ref, X_trial)
                if s > best_score:
                    best_score = s
                    best_idx   = i

            if best_idx is None:
                break

            X_min = np.delete(X_min, best_idx, axis=0)

            step        += 1
            removal_frac = 1.0 - len(X_min) / n_orig

            X_cur = np.vstack([X_maj, X_min])
            y_cur = np.concatenate([
                np.full(len(X_maj), maj_label, dtype=int),
                np.full(len(X_min), min_label, dtype=int),
            ])
            yield X_cur, y_cur, step, removal_frac

    def get_most_degraded(self, X: np.ndarray, y: np.ndarray):
        """
        Skip straight to the most-degraded state (min_frac remaining).
        Returns (X_degraded, y_degraded).
        """
        last = None
        for state in self.degradation_sequence(X, y):
            last = state
        return last[0], last[1]
