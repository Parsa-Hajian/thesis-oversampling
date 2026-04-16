"""
General-purpose helper utilities for the circular-oversampling pipeline.

Provides functions for minority-class identification and computation of the
number of synthetic samples required by a given sampling strategy.
"""

import numpy as np

# Small constant to avoid division-by-zero in numerical routines.
EPS = 1e-12


def minority_label(y, minority_lab=None):
    """Identify the minority class label in a target array.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target labels.
    minority_lab : int or str, optional
        If provided, this value is returned directly (useful when the caller
        already knows which class is the minority).

    Returns
    -------
    label : int or str
        The label of the minority (least-frequent) class.
    """
    if minority_lab is not None:
        return minority_lab
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmin(counts)]


def n_synth_from_strategy(y, min_lab, sampling_strategy):
    """Compute how many synthetic minority points to generate.

    The function supports two modes controlled by *sampling_strategy*:

    1. **Absolute count** -- an integer is interpreted as the exact number of
       synthetic samples to produce.
    2. **Ratio** -- a float ``r`` means "generate enough synthetic samples so
       that ``n_minority == floor(r * n_majority)``".  If the minority already
       meets or exceeds that target, zero is returned.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target labels.
    min_lab : int or str
        The label that identifies the minority class.
    sampling_strategy : int or float
        When *int*: the absolute number of synthetic points to create.
        When *float*: the desired minority-to-majority ratio.

    Returns
    -------
    n_synth : int
        Non-negative number of synthetic samples to generate.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 0, 0, 0, 1, 1])
    >>> n_synth_from_strategy(y, min_lab=1, sampling_strategy=1.0)
    3
    >>> n_synth_from_strategy(y, min_lab=1, sampling_strategy=10)
    10
    """
    y = np.asarray(y)
    n_min = int(np.sum(y == min_lab))
    n_maj = int(y.size - n_min)

    # Absolute count mode.
    if isinstance(sampling_strategy, (int, np.integer)):
        return max(0, int(sampling_strategy))

    # Ratio mode.
    ratio = float(sampling_strategy)
    target_min = int(np.floor(ratio * n_maj))
    return max(0, target_min - n_min)
