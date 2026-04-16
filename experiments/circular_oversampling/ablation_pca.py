"""
ablation_pca.py -- PCA vs. No-PCA seed selection ablation.

Compares seed selection quality (NHOP, AGTP, JSD, Z) and downstream GVM-CO
F1/AUC with PCA (k_pc=2, default) versus no PCA (k_pc=d, raw features).

Because the external KEEL/UCI dataset files may not be present, this script
generates synthetic analogues using sklearn make_classification with the same
(n_samples, n_features, IR) parameters as the 14 benchmark datasets.

Run:
    cd ~/Desktop/circular-oversampling
    /opt/anaconda3/bin/python3 experiments/ablation_pca.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.seed_selection.selector import SeedSelector
from src.core.gravity_vonmises import GravityVonMises

RANDOM_STATE = 42
N_SEEDS_FRAC = 0.5
N_CANDIDATES = 100

# Synthetic analogues for the 14 benchmark datasets
# (name, n_samples, n_features, ir)
SYNTHETIC_SPECS = [
    ("ecoli1",       336,  7,  3.4),
    ("ecoli2",       336,  7,  5.5),
    ("ecoli3",       336,  7,  8.6),
    ("glass1",       214,  9,  1.8),
    ("glass4",       214,  9, 15.5),
    ("yeast1",       1484, 8,  2.5),
    ("yeast3",       1484, 8,  8.1),
    ("new-thyroid1", 215,  5,  5.1),
    ("haberman",     306,  3,  2.8),
    ("vehicle0",     846, 18,  3.3),
    ("pima",         768,  8,  1.9),
    ("wisconsin",    699,  9,  1.9),
    ("heart",        303, 13,  1.2),
    ("ionosphere",   351, 34,  1.8),
]


def make_synthetic_dataset(n_samples, n_features, ir, seed):
    """Generate a synthetic imbalanced binary dataset with given IR."""
    n_min = max(2, int(n_samples / (ir + 1)))
    n_maj = n_samples - n_min
    weights = [n_maj / n_samples, n_min / n_samples]
    n_informative = max(2, min(n_features - 1, n_features // 2 + 1))
    n_redundant = min(n_features - n_informative, n_features // 4)
    n_redundant = max(0, n_redundant)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        weights=weights,
        random_state=seed,
    )
    return X, y


def evaluate_seed_selection(X_min, n_seeds, n_pcs):
    selector = SeedSelector(
        n_candidates=N_CANDIDATES,
        n_pcs=n_pcs,
        k_clusters=min(5, max(2, len(X_min) // 5)),
        k_topo=5,
        jsd_weight=0.3,
        z_weight=0.5,
        random_state=RANDOM_STATE,
    )
    _, _, scores = selector.select(X_min, n_seeds)
    return scores


def cv_f1_auc(X, y, use_pca, seed):
    """5-fold CV with GVM oversampler, return mean F1 and AUC."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    f1s, aucs = [], []
    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        try:
            gvm = GravityVonMises(use_pca=use_pca, random_state=seed)
            X_res, y_res = gvm.fit_resample(X_tr, y_tr)
            clf = RandomForestClassifier(n_estimators=100, random_state=seed)
            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]
            f1s.append(f1_score(y_te, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_te, y_prob))
        except Exception:
            f1s.append(float("nan"))
            aucs.append(float("nan"))
    return np.nanmean(f1s), np.nanmean(aucs)


def run_ablation():
    results = []

    for idx, (ds_name, n_samples, n_features, ir) in enumerate(SYNTHETIC_SPECS):
        print(f"  {ds_name} (n={n_samples}, d={n_features}, IR={ir}) ...", end=" ", flush=True)
        seed = RANDOM_STATE + idx
        X, y = make_synthetic_dataset(n_samples, n_features, ir, seed)

        counts = np.bincount(y.astype(int))
        minority_label = int(np.argmin(counts))
        X_min = X[y == minority_label]
        n, d = X_min.shape
        n_seeds = max(2, int(n * N_SEEDS_FRAC))
        n_pcs_pca = min(2, d, n - 1)

        scores_pca   = evaluate_seed_selection(X_min, n_seeds, n_pcs=n_pcs_pca)
        scores_nopca = evaluate_seed_selection(X_min, n_seeds, n_pcs=d)

        f1_pca,   auc_pca   = cv_f1_auc(X, y, use_pca=True,  seed=seed)
        f1_nopca, auc_nopca = cv_f1_auc(X, y, use_pca=False, seed=seed)

        results.append({
            "dataset":     ds_name,
            "d":           d,
            "n_min":       n,
            "ir":          ir,
            "nhop_nopca":  scores_nopca["nhop"],
            "agtp_nopca":  scores_nopca["agtp"],
            "jsd_nopca":   scores_nopca["jsd"],
            "z_nopca":     scores_nopca["z"],
            "f1_nopca":    f1_nopca,
            "auc_nopca":   auc_nopca,
            "nhop_pca":    scores_pca["nhop"],
            "agtp_pca":    scores_pca["agtp"],
            "jsd_pca":     scores_pca["jsd"],
            "z_pca":       scores_pca["z"],
            "f1_pca":      f1_pca,
            "auc_pca":     auc_pca,
        })
        print("done")

    df = pd.DataFrame(results)
    out_path = os.path.join(ROOT, "results", "tables", "ablation_pca.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}\n")

    m = df.mean(numeric_only=True)
    print("=== SUMMARY (mean across 14 synthetic datasets) ===")
    print(f"{'Setting':<22} {'NHOP':>7} {'AGTP':>7} {'JSD':>7} {'Z':>7} {'F1':>7} {'AUC':>7}")
    print(f"{'No PCA (k=d)':<22} {m['nhop_nopca']:>7.4f} {m['agtp_nopca']:>7.4f} "
          f"{m['jsd_nopca']:>7.4f} {m['z_nopca']:>7.4f} "
          f"{m['f1_nopca']:>7.4f} {m['auc_nopca']:>7.4f}")
    print(f"{'PCA (k=2)':<22} {m['nhop_pca']:>7.4f} {m['agtp_pca']:>7.4f} "
          f"{m['jsd_pca']:>7.4f} {m['z_pca']:>7.4f} "
          f"{m['f1_pca']:>7.4f} {m['auc_pca']:>7.4f}")
    return df


if __name__ == "__main__":
    print("Running PCA vs. No-PCA seed selection ablation...")
    df = run_ablation()
    print("\nDone.")
