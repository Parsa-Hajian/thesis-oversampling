"""
Dataset loading utilities for KEEL, UCI, and standard CSV formats.

The module can load datasets stored locally under the project's ``data/``
directory tree.  It also provides a lightweight downloader for KEEL datasets.

Supported formats
-----------------
* ``.csv`` -- standard comma-separated values with a header row.
* ``.dat`` -- KEEL dataset format (``@attribute`` / ``@data`` sections).
* ``.arff`` -- Weka ARFF format (attribute declarations + data section).

Directory layout
----------------
::

    data/
    ├── raw/            # original downloaded files
    └── processed/      # cleaned numpy arrays (optional cache)
"""

import logging
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(name, data_dir=None):
    """Load a dataset by name and return ``(X, y)`` as numpy arrays.

    The function searches for the dataset file under ``data_dir/raw/`` (or
    the project-level ``data/raw/`` by default) by probing several common
    file extensions.

    Parameters
    ----------
    name : str
        Dataset name (without extension), e.g. ``"ecoli1"``.
    data_dir : str or Path or None, default=None
        Root data directory.  Defaults to ``<project>/data``.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target labels (encoded as integers: 0 = majority, 1 = minority).

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be located.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    raw_dir = data_dir / "raw"

    # Try common extensions.
    for ext in (".dat", ".csv", ".arff"):
        filepath = raw_dir / f"{name}{ext}"
        if filepath.exists():
            logger.info("Loading %s from %s", name, filepath)
            return _load_by_extension(filepath)

    # Also try directly in data_dir (flat layout).
    for ext in (".dat", ".csv", ".arff"):
        filepath = data_dir / f"{name}{ext}"
        if filepath.exists():
            logger.info("Loading %s from %s", name, filepath)
            return _load_by_extension(filepath)

    raise FileNotFoundError(
        f"Dataset '{name}' not found. Searched in {raw_dir} and {data_dir} "
        f"for extensions .dat, .csv, .arff."
    )


def load_keel_dat(filepath):
    """Parse a KEEL ``.dat`` file into a feature matrix and label vector.

    KEEL format overview::

        @relation name
        @attribute attr1 real [min, max]
        @attribute class {negative, positive}
        @data
        0.72,0.49,...,negative
        ...

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.dat`` file.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (float64).
    y : ndarray of shape (n_samples,)
        Binary labels encoded as 0 (majority) / 1 (minority).
    """
    filepath = Path(filepath)

    attribute_names = []
    attribute_types = []  # "numeric" or set of class labels
    data_lines = []
    in_data = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip blank lines and comments.
            if not line or line.startswith("%"):
                continue

            if line.lower().startswith("@data"):
                in_data = True
                continue

            if in_data:
                # Some KEEL files use both comma and space separators.
                data_lines.append(line)
                continue

            # Parse @attribute declarations.
            match = re.match(
                r"@[Aa]ttribute\s+(\S+)\s+(.*)", line
            )
            if match:
                attr_name = match.group(1)
                attr_type = match.group(2).strip()
                attribute_names.append(attr_name)

                if attr_type.lower() in ("real", "integer", "numeric"):
                    attribute_types.append("numeric")
                elif attr_type.startswith("{"):
                    # Nominal: extract class labels.
                    labels = [
                        lbl.strip()
                        for lbl in attr_type.strip("{}").split(",")
                    ]
                    attribute_types.append(labels)
                else:
                    attribute_types.append("numeric")

    if not data_lines:
        raise ValueError(f"No data lines found in {filepath}.")

    # Build DataFrame.
    rows = []
    for dl in data_lines:
        values = [v.strip() for v in dl.split(",")]
        rows.append(values)

    df = pd.DataFrame(rows, columns=attribute_names)

    # Identify the target column (last attribute or the one named "class"/"Class").
    target_col = attribute_names[-1]
    for col in attribute_names:
        if col.lower() in ("class", "clase"):
            target_col = col
            break

    # Separate features and target.
    feature_cols = [c for c in attribute_names if c != target_col]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values.astype(
        np.float64
    )
    y_raw = df[target_col].values

    # Encode target: map "positive" -> 1, "negative" -> 0.
    # For other label schemes, the minority class gets label 1.
    y = _encode_binary_labels(y_raw, attribute_types[-1])

    # Drop rows with NaN features (rare but possible in noisy files).
    mask = ~np.isnan(X).any(axis=1)
    if not mask.all():
        n_dropped = (~mask).sum()
        logger.warning(
            "Dropped %d rows with NaN features from %s.", n_dropped, filepath
        )
        X, y = X[mask], y[mask]

    return X, y


def load_csv(filepath):
    """Load a CSV dataset.  Last column is assumed to be the target.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath)

    # Target = last column.
    X = df.iloc[:, :-1].values.astype(np.float64)
    y_raw = df.iloc[:, -1].values
    y = _encode_binary_labels(y_raw)

    return X, y


def load_arff(filepath):
    """Load a Weka ARFF file.

    The parser handles both numeric and nominal attributes.  The last
    attribute is treated as the target.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    """
    filepath = Path(filepath)

    attribute_names = []
    attribute_types = []
    data_lines = []
    in_data = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower = line.lower()
            if lower.startswith("@data"):
                in_data = True
                continue

            if in_data:
                data_lines.append(line)
                continue

            if lower.startswith("@attribute"):
                match = re.match(
                    r"@[Aa]ttribute\s+['\"]?(\S+?)['\"]?\s+(.*)", line
                )
                if match:
                    attribute_names.append(match.group(1))
                    attr_type = match.group(2).strip()
                    if attr_type.lower() in (
                        "real",
                        "integer",
                        "numeric",
                    ):
                        attribute_types.append("numeric")
                    elif attr_type.startswith("{"):
                        labels = [
                            lbl.strip()
                            for lbl in attr_type.strip("{}").split(",")
                        ]
                        attribute_types.append(labels)
                    else:
                        attribute_types.append("numeric")

    if not data_lines:
        raise ValueError(f"No data lines found in {filepath}.")

    rows = []
    for dl in data_lines:
        values = [v.strip() for v in dl.split(",")]
        rows.append(values)

    df = pd.DataFrame(rows, columns=attribute_names)
    target_col = attribute_names[-1]
    feature_cols = attribute_names[:-1]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values.astype(
        np.float64
    )
    y_raw = df[target_col].values
    y = _encode_binary_labels(y_raw, attribute_types[-1])

    mask = ~np.isnan(X).any(axis=1)
    if not mask.all():
        n_dropped = (~mask).sum()
        logger.warning(
            "Dropped %d rows with NaN features from %s.", n_dropped, filepath
        )
        X, y = X[mask], y[mask]

    return X, y


def download_dataset(name, dest_dir=None):
    """Download a KEEL dataset if not already present locally.

    Parameters
    ----------
    name : str
        Dataset name (e.g. ``"ecoli1"``).
    dest_dir : str or Path or None, default=None
        Destination directory.  Defaults to ``<project>/data/raw``.

    Returns
    -------
    filepath : Path
        Local path to the downloaded file.
    """
    dest_dir = Path(dest_dir) if dest_dir is not None else DATA_DIR / "raw"
    dest_dir.mkdir(parents=True, exist_ok=True)
    filepath = dest_dir / f"{name}.dat"

    if filepath.exists():
        logger.info("Dataset '%s' already exists at %s.", name, filepath)
        return filepath

    # KEEL dataset repository URL pattern.
    url = (
        f"https://sci2s.ugr.es/keel/dataset/data/imbalanced/{name}.dat"
    )
    logger.info("Downloading %s from %s ...", name, url)

    try:
        urllib.request.urlretrieve(url, filepath)
        logger.info("Saved to %s.", filepath)
    except Exception as exc:
        # Try alternative URL patterns.
        alt_urls = [
            f"https://sci2s.ugr.es/keel/dataset/data/imbalanced/{name}-5-fold/{name}.dat",
        ]
        downloaded = False
        for alt_url in alt_urls:
            try:
                urllib.request.urlretrieve(alt_url, filepath)
                logger.info("Saved to %s (from alt URL).", filepath)
                downloaded = True
                break
            except Exception:
                continue

        if not downloaded:
            raise FileNotFoundError(
                f"Could not download dataset '{name}'. Last error: {exc}"
            ) from exc

    return filepath


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_by_extension(filepath):
    """Dispatch to the appropriate loader based on file extension."""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext == ".dat":
        return load_keel_dat(filepath)
    elif ext == ".csv":
        return load_csv(filepath)
    elif ext == ".arff":
        return load_arff(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _encode_binary_labels(y_raw, label_spec=None):
    """Encode raw labels into binary 0/1 integers.

    Heuristics
    ----------
    * If *label_spec* is a list (from KEEL/ARFF nominal declarations), the
      label containing "positive" maps to 1, "negative" to 0.
    * If labels are already numeric (0/1 or similar), they are cast directly.
    * Otherwise the least-frequent class is mapped to 1 (minority).

    Parameters
    ----------
    y_raw : ndarray of shape (n_samples,)
        Raw label values (strings or numbers).
    label_spec : list[str] or str or None
        Optional label specification from the file header.

    Returns
    -------
    y : ndarray of shape (n_samples,) with dtype int
    """
    # Attempt to convert to numeric first.
    try:
        y_numeric = pd.to_numeric(pd.Series(y_raw), errors="raise")
        unique_vals = sorted(y_numeric.unique())
        if set(unique_vals).issubset({0, 1}):
            return y_numeric.values.astype(int)
        if len(unique_vals) == 2:
            # Map smaller value to 0 (majority assumption), larger to 1.
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            return y_numeric.map(mapping).values.astype(int)
    except (ValueError, TypeError):
        pass

    # String labels.
    y_str = np.array([str(v).strip().lower() for v in y_raw])
    unique_labels = np.unique(y_str)

    # Check for KEEL-style "positive"/"negative".
    if "positive" in unique_labels and "negative" in unique_labels:
        return np.where(y_str == "positive", 1, 0).astype(int)

    # Check label_spec for ordering hints.
    if isinstance(label_spec, list) and len(label_spec) == 2:
        for i, lbl in enumerate(label_spec):
            if "positive" in lbl.lower():
                pos_label = lbl.strip().lower()
                return np.where(y_str == pos_label, 1, 0).astype(int)

    # Fallback: minority (least frequent) -> 1.
    if len(unique_labels) == 2:
        counts = {lbl: np.sum(y_str == lbl) for lbl in unique_labels}
        minority = min(counts, key=counts.get)
        return np.where(y_str == minority, 1, 0).astype(int)

    raise ValueError(
        f"Cannot encode labels into binary. Found {len(unique_labels)} "
        f"unique labels: {unique_labels}"
    )
