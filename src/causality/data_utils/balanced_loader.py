"""
balanced_loader.py -- Load fully-balanced binary subsets of UCI datasets.

Because real KEEL/UCI files may not be present, this module generates
synthetic stand-ins using sklearn make_classification with matching
(n_samples, n_features) parameters.  All returned datasets have IR ≈ 1.0.
"""

import numpy as np
from sklearn.datasets import make_classification

# (name, n_samples_balanced, n_features)
DATASET_SPECS = [
    ("breast-cancer-wisconsin", 400, 9),
    ("ionosphere",              350, 34),
    ("heart",                   270, 13),
    ("banknote",                600, 4),
    ("sonar",                   208, 60),
    ("pima",                    400, 8),
    ("haberman",                200, 3),
    ("glass",                   146, 9),
    ("vehicle",                 400, 18),
    ("credit",                  400, 23),
]

RANDOM_STATE = 42


def load_balanced(name: str, random_state: int = RANDOM_STATE):
    """
    Return (X, y) balanced binary dataset (IR ≈ 1.0) for `name`.

    If a real .csv file exists at  data/raw/<name>.csv  it is loaded;
    otherwise a synthetic analogue is generated.
    """
    import os
    raw_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw", f"{name}.csv"
    )
    raw_path = os.path.normpath(raw_path)

    if os.path.isfile(raw_path):
        import pandas as pd
        df = pd.read_csv(raw_path)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(int)
        # Re-balance: take equal-sized random samples from each class
        return _balance(X, y, random_state)

    # Synthetic fallback
    spec = {s[0]: s for s in DATASET_SPECS}.get(name)
    if spec is None:
        raise ValueError(f"Unknown dataset: {name}. Add it to DATASET_SPECS.")
    _, n_samples, n_features = spec
    n_informative = max(2, min(n_features - 1, n_features // 2 + 1))
    n_redundant   = max(0, min(n_features - n_informative, n_features // 4))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        weights=[0.5, 0.5],
        random_state=random_state,
    )
    return X, y.astype(int)


def load_all_balanced(random_state: int = RANDOM_STATE):
    """Return list of (name, X, y) for all DATASET_SPECS."""
    return [(name, *load_balanced(name, random_state))
            for name, *_ in DATASET_SPECS]


def _balance(X, y, random_state):
    """Downsample majority class so both classes have equal size."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    n_min = counts.min()
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, size=n_min, replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]
