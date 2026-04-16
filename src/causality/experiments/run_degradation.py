"""
run_degradation.py -- Phase 1: balanced → progressively imbalanced sweep.

For each dataset:
  1. Load fully-balanced version (IR ≈ 1.0).
  2. Step-by-step remove minority instances (NHOP-guided greedy removal).
  3. At each step run 5 classifiers × leakage-safe 5-fold CV → 6 metrics.
  4. Save results/degradation/<dataset_name>.csv

Anti-leakage guarantee (see metrics_collector.py)
--------------------------------------------------
* StratifiedKFold split is on the CURRENT (already degraded) dataset.
* No oversampling is applied during degradation — classifiers see the
  raw imbalanced data.
* Scaler (StandardScaler) is fit on the training fold only; test fold
  is transformed with training-fold statistics.
* No information about the test fold is used at any point during training.

Run:
    cd ~/Desktop/imbalance-causality
    /opt/anaconda3/bin/python3 src/experiments/run_degradation.py
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

RANDOM_STATE = 42
N_SPLITS     = 5

# Classifiers to evaluate (dict: label -> factory)
CLASSIFIERS = {
    "RF":  lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "DT":  lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "LR":  lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                  random_state=RANDOM_STATE),
}


def run():
    out_dir = os.path.join(ROOT, "results", "degradation")
    os.makedirs(out_dir, exist_ok=True)

    cv = LeakageSafeCV(n_splits=N_SPLITS, random_state=RANDOM_STATE)
    unbalancer = IncrementalUnbalancer(
        step_frac=0.01,
        min_frac=0.20,
        random_state=RANDOM_STATE,
    )

    datasets = load_all_balanced(random_state=RANDOM_STATE)

    for ds_name, X, y in datasets:
        print(f"\n=== {ds_name} (n={len(X)}, d={X.shape[1]}) ===")
        rows = []

        for X_step, y_step, step_idx, removal_frac in \
                unbalancer.degradation_sequence(X, y):

            counts = np.bincount(y_step.astype(int))
            ir = counts.max() / counts.min() if counts.min() > 0 else float("inf")

            print(f"  step={step_idx:3d}  removal={removal_frac:.2f}"
                  f"  IR={ir:.2f}  n_min={counts.min()}", end="")

            for clf_name, clf_factory in CLASSIFIERS.items():
                try:
                    metrics = cv.mean_metrics(
                        X_step, y_step,
                        clf_factory=clf_factory,
                        oversampler_factory=None,       # no oversampling in Phase 1
                        scaler_factory=lambda: StandardScaler(),
                    )
                except Exception as e:
                    metrics = {m: float("nan")
                               for m in ("auc","f1","gmean","bal_acc","sens","spec")}

                row = {
                    "dataset":       ds_name,
                    "step":          step_idx,
                    "removal_frac":  removal_frac,
                    "ir":            ir,
                    "n_minority":    int(counts.min()),
                    "n_majority":    int(counts.max()),
                    "classifier":    clf_name,
                    **{f"{m}": v for m, v in metrics.items()},
                }
                rows.append(row)
            print("  done")

        df = pd.DataFrame(rows)
        out_path = os.path.join(out_dir, f"{ds_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    print("Phase 1: Degradation sweep")
    run()
    print("\nDone.")
