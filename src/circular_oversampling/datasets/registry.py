"""
Dataset metadata registry for the imbalanced classification benchmark.

Each entry records basic statistics (samples, features, imbalance ratio),
the data source (KEEL or UCI), and the identifier used for the minority
class in the original files.  These metadata are used by the experiment
runner and reporting modules.

Imbalance Ratio (IR)
--------------------
Defined as ``n_majority / n_minority``.  Higher values indicate more severe
class imbalance.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: Dict[str, Dict[str, Any]] = {
    "ecoli1": {
        "samples": 336,
        "features": 7,
        "ir": 3.36,
        "source": "KEEL",
        "minority_class": "positive",
    },
    "ecoli2": {
        "samples": 336,
        "features": 7,
        "ir": 5.46,
        "source": "KEEL",
        "minority_class": "positive",
    },
    "ecoli3": {
        "samples": 336,
        "features": 7,
        "ir": 8.6,
        "source": "KEEL",
        "minority_class": "positive",
    },
    "glass1": {
        "samples": 214,
        "features": 9,
        "ir": 1.82,
        "source": "KEEL",
        "minority_class": "positive",
    },
    "glass4": {
        "samples": 214,
        "features": 9,
        "ir": 15.47,
        "source": "KEEL",
        "minority_class": "positive",
    },
    "yeast1": {
        "samples": 1484,
        "features": 8,
        "ir": 2.46,
        "source": "KEEL",
        "minority_class": "positive",
    },
    "yeast3": {
        "samples": 1484,
        "features": 8,
        "ir": 8.1,
        "source": "KEEL",
        "minority_class": "positive",
    },
    "new-thyroid1": {
        "samples": 215,
        "features": 5,
        "ir": 5.14,
        "source": "KEEL",
        "minority_class": "positive",
    },
    "haberman": {
        "samples": 306,
        "features": 3,
        "ir": 2.78,
        "source": "UCI",
        "minority_class": 1,
    },
    "vehicle0": {
        "samples": 846,
        "features": 18,
        "ir": 3.25,
        "source": "KEEL",
        "minority_class": "positive",
    },
    "pima": {
        "samples": 768,
        "features": 8,
        "ir": 1.87,
        "source": "UCI",
        "minority_class": 1,
    },
    "wisconsin": {
        "samples": 699,
        "features": 9,
        "ir": 1.86,
        "source": "UCI",
        "minority_class": 1,
    },
    "heart": {
        "samples": 303,
        "features": 13,
        "ir": 1.25,
        "source": "UCI",
        "minority_class": 1,
    },
    "ionosphere": {
        "samples": 351,
        "features": 34,
        "ir": 1.79,
        "source": "UCI",
        "minority_class": "b",
    },
}


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_dataset_info(name):
    """Return the metadata dict for a dataset.

    Parameters
    ----------
    name : str
        Dataset name (must match a key in :data:`DATASETS`).

    Returns
    -------
    info : dict
        Keys: ``samples``, ``features``, ``ir``, ``source``,
        ``minority_class``.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASETS.keys())}"
        )
    return DATASETS[name]


def list_datasets():
    """Return an ordered list of all registered dataset names.

    Returns
    -------
    names : list[str]
    """
    return list(DATASETS.keys())


def list_datasets_by_ir(ascending=True):
    """Return dataset names sorted by imbalance ratio.

    Parameters
    ----------
    ascending : bool, default=True
        If ``True``, least imbalanced first.

    Returns
    -------
    names : list[str]
    """
    return sorted(DATASETS, key=lambda n: DATASETS[n]["ir"], reverse=not ascending)


def filter_datasets(source=None, min_ir=None, max_ir=None):
    """Return dataset names matching the given filters.

    Parameters
    ----------
    source : str or None
        Filter by data source (e.g. ``"KEEL"`` or ``"UCI"``).
    min_ir : float or None
        Minimum imbalance ratio (inclusive).
    max_ir : float or None
        Maximum imbalance ratio (inclusive).

    Returns
    -------
    names : list[str]
    """
    result: List[str] = []
    for name, info in DATASETS.items():
        if source is not None and info["source"].upper() != source.upper():
            continue
        if min_ir is not None and info["ir"] < min_ir:
            continue
        if max_ir is not None and info["ir"] > max_ir:
            continue
        result.append(name)
    return result
