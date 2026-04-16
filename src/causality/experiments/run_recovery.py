"""
run_recovery.py -- Phase 2: imbalanced → balanced recovery sweep.

Starting from the most-degraded state (20% minority remaining), each
oversampler incrementally adds synthetic minority samples step-by-step
until full balance is restored.  Metrics are recorded at each step.

Anti-leakage guarantee
----------------------
* The "starting point" (20% minority) is fixed before any CV split.
* All oversampling occurs INSIDE training folds only (LeakageSafeCV).
* At each recovery step the oversampler is told a target IR; it generates
  only enough samples to reach that target — no full-resample at once.
* Scaler is fit on training fold; test uses training-fold statistics.
* The test fold is NEVER passed to, or seen by, any oversampler at any stage.

Run:
    cd ~/Desktop/imbalance-causality
    /opt/anaconda3/bin/python3 src/experiments/run_recovery.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data_utils.balanced_loader import load_all_balanced
from src.data_utils.incremental_unbalancer import IncrementalUnbalancer
from src.experiments.metrics_collector import LeakageSafeCV
from src.oversamplers.wrapper import get_oversampler_factory, ALL_OVERSAMPLERS

RANDOM_STATE   = 42
N_SPLITS       = 5
STEP_FRAC      = 0.01   # add 1% of original minority per recovery step
MIN_FRAC       = 0.20   # starting IR (20% minority remaining)

CLASSIFIERS = {
    "RF":  lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "DT":  lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "LR":  lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                  random_state=RANDOM_STATE),
}

# Oversamplers to test (skip ones that aren't installed)
OVERSAMPLERS_TO_TEST = ["none", "ros", "smote", "bsmote", "adasyn", "kmsmote", "gvmco"]


class TargetIRSampler:
    """
    Wraps an oversampler so it only generates enough samples to reach
    a specific imbalance ratio (target_ir) rather than always balancing
    to 1:1.  This enables the incremental recovery sweep.
    """

    def __init__(self, base_factory, target_n_minority: int, random_state: int = 42):
        self.base_factory     = base_factory
        self.target_n_minority = target_n_minority
        self.random_state     = random_state

    def fit_resample(self, X, y):
        counts = np.bincount(y.astype(int))
        min_label = int(np.argmin(counts))
        n_current = counts[min_label]
        if n_current >= self.target_n_minority:
            return X, y
        # Let oversampler generate up to target
        try:
            sampler = self.base_factory()
            # Temporarily set sampling_strategy to generate exactly what we need
            n_maj = counts.max()
            ratio  = self.target_n_minority / n_maj
            ratio  = min(ratio, 1.0)
            if hasattr(sampler, "sampling_strategy"):
                sampler.sampling_strategy = ratio
            X_res, y_res = sampler.fit_resample(X, y)
            return X_res, y_res
        except Exception:
            return X, y


def run():
    out_dir = os.path.join(ROOT, "results", "recovery")
    os.makedirs(out_dir, exist_ok=True)

    cv         = LeakageSafeCV(n_splits=N_SPLITS, random_state=RANDOM_STATE)
    unbalancer = IncrementalUnbalancer(
        step_frac=STEP_FRAC,
        min_frac=MIN_FRAC,
        random_state=RANDOM_STATE,
    )

    datasets = load_all_balanced(random_state=RANDOM_STATE)

    for ds_name, X_bal, y_bal in datasets:
        print(f"\n=== {ds_name} ===")

        # Get the most-degraded starting state
        X_start, y_start = unbalancer.get_most_degraded(X_bal, y_bal)
        counts_start = np.bincount(y_start.astype(int))
        min_label    = int(np.argmin(counts_start))
        n_min_start  = counts_start[min_label]
        n_min_target = counts_start.max()   # full balance
        n_steps      = max(1, int(np.round((n_min_target - n_min_start) /
                                            max(1, int(n_min_target * STEP_FRAC)))))

        # Build list of target minority sizes for each recovery step
        target_sizes = np.linspace(n_min_start, n_min_target,
                                   n_steps + 1, dtype=int)

        rows = []

        for ovs_name in OVERSAMPLERS_TO_TEST:
            print(f"  Oversampler: {ovs_name}")
            try:
                base_factory = get_oversampler_factory(ovs_name, RANDOM_STATE)
            except ImportError as e:
                print(f"    Skipping {ovs_name}: {e}")
                continue

            for step_idx, target_n in enumerate(target_sizes):
                recovery_frac = (target_n - n_min_start) / max(1, n_min_target - n_min_start)

                # Factory that generates up to target_n minority samples inside each fold
                ovs_factory = (
                    (lambda tn: lambda: TargetIRSampler(base_factory, tn, RANDOM_STATE))(target_n)
                    if ovs_name != "none" else (lambda: get_oversampler_factory("none")())
                )

                counts_cur = np.bincount(y_start.astype(int))
                ir_cur = counts_cur.max() / max(1, counts_cur.min())

                for clf_name, clf_factory in CLASSIFIERS.items():
                    try:
                        metrics = cv.mean_metrics(
                            X_start, y_start,
                            clf_factory=clf_factory,
                            oversampler_factory=ovs_factory,
                            scaler_factory=lambda: StandardScaler(),
                        )
                    except Exception:
                        metrics = {m: float("nan")
                                   for m in ("auc","f1","gmean","bal_acc","sens","spec")}

                    rows.append({
                        "dataset":       ds_name,
                        "oversampler":   ovs_name,
                        "step":          step_idx,
                        "recovery_frac": recovery_frac,
                        "target_n_min":  int(target_n),
                        "ir_start":      ir_cur,
                        "classifier":    clf_name,
                        **metrics,
                    })
            print(f"    done ({len(target_sizes)} steps)")

        df = pd.DataFrame(rows)
        out_path = os.path.join(out_dir, f"{ds_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    print("Phase 2: Recovery sweep")
    run()
    print("\nDone.")
