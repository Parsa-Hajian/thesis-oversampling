"""
Non-parametric statistical tests for comparing oversampling methods.

The standard workflow when benchmarking *k* oversampling methods across *n*
datasets is:

1. Run the **Friedman test** to determine whether there is a statistically
   significant difference among the methods.
2. If the Friedman test rejects the null hypothesis, apply **post-hoc** tests
   (Holm or Nemenyi) to identify which pairs of methods differ.
3. Optionally compute **average ranks** and the **critical difference** for
   a CD diagram.

References
----------
* Demsar, J. (2006). Statistical comparisons of classifiers over multiple
  data sets. *JMLR*, 7, 1--30.
* Garcia, S. & Herrera, F. (2008). An extension on "Statistical comparisons
  of classifiers over multiple data sets" for all pairwise comparisons.
  *JMLR*, 9, 2677--2694.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare


def friedman_test(results_matrix):
    """Friedman non-parametric test for comparing multiple methods.

    Tests the null hypothesis that all methods perform equally (i.e., their
    average ranks are equal) across a set of datasets.

    Parameters
    ----------
    results_matrix : array-like of shape (n_datasets, n_methods)
        Each row is a dataset, each column is a method.  Values are the
        performance metric (higher is better).

    Returns
    -------
    statistic : float
        Friedman chi-squared statistic.
    p_value : float
        Associated p-value.

    Raises
    ------
    ValueError
        If fewer than 3 datasets or fewer than 3 methods are provided
        (the Friedman test requires at least 3 groups).
    """
    results_matrix = np.asarray(results_matrix, dtype=np.float64)
    n_datasets, n_methods = results_matrix.shape

    if n_datasets < 3:
        raise ValueError(
            f"Friedman test requires at least 3 datasets, got {n_datasets}."
        )
    if n_methods < 3:
        raise ValueError(
            f"Friedman test requires at least 3 methods, got {n_methods}."
        )

    # scipy expects each column as a separate argument.
    columns = [results_matrix[:, j] for j in range(n_methods)]
    statistic, p_value = friedmanchisquare(*columns)
    return float(statistic), float(p_value)


def holms_posthoc(results_matrix, method_names):
    """Holm's step-down post-hoc test following a significant Friedman test.

    Computes pairwise comparisons between all methods and adjusts p-values
    using the Holm procedure to control the family-wise error rate.

    Parameters
    ----------
    results_matrix : array-like of shape (n_datasets, n_methods)
        Performance metric values (higher is better).
    method_names : list[str]
        Names for each method (column).

    Returns
    -------
    p_values : pandas.DataFrame
        Symmetric DataFrame of Holm-adjusted pairwise p-values.
    """
    results_matrix = np.asarray(results_matrix, dtype=np.float64)
    n_datasets, n_methods = results_matrix.shape
    method_names = list(method_names)

    # Compute average ranks (rank 1 = best, i.e., highest metric value).
    ranks = np.zeros_like(results_matrix)
    for i in range(n_datasets):
        # Rank in descending order: highest value gets rank 1.
        ranks[i] = stats.rankdata(-results_matrix[i])

    avg_ranks = ranks.mean(axis=0)

    # Pairwise z-statistic under the Friedman null.
    # z_ij = (R_i - R_j) / sqrt( k*(k+1) / (6*N) )
    k = n_methods
    N = n_datasets
    denom = np.sqrt(k * (k + 1) / (6.0 * N))

    n_comparisons = k * (k - 1) // 2
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            z = abs(avg_ranks[i] - avg_ranks[j]) / denom
            p = 2.0 * stats.norm.sf(z)  # two-sided p-value
            pairs.append((i, j, z, p))

    # Sort by raw p-value (ascending).
    pairs.sort(key=lambda t: t[3])

    # Apply Holm's step-down correction.
    adjusted = np.ones(len(pairs))
    for step, (i, j, z, p) in enumerate(pairs):
        adjusted[step] = min(1.0, p * (n_comparisons - step))

    # Enforce monotonicity (Holm correction is step-down).
    for step in range(1, len(adjusted)):
        adjusted[step] = max(adjusted[step], adjusted[step - 1])

    # Build a symmetric DataFrame.
    p_matrix = pd.DataFrame(
        np.ones((k, k)), index=method_names, columns=method_names
    )
    for idx, (i, j, _, _) in enumerate(pairs):
        p_matrix.iloc[i, j] = adjusted[idx]
        p_matrix.iloc[j, i] = adjusted[idx]
    np.fill_diagonal(p_matrix.values, 0.0)

    return p_matrix


def nemenyi_test(results_matrix, method_names):
    """Nemenyi post-hoc test for all pairwise comparisons.

    Uses the ``scikit-posthocs`` implementation under the hood.

    Parameters
    ----------
    results_matrix : array-like of shape (n_datasets, n_methods)
        Performance metric values (higher is better).
    method_names : list[str]
        Names for each method (column).

    Returns
    -------
    p_values : pandas.DataFrame
        Symmetric DataFrame of pairwise p-values.
    """
    import scikit_posthocs as sp

    results_matrix = np.asarray(results_matrix, dtype=np.float64)
    method_names = list(method_names)

    # scikit-posthocs expects a DataFrame in long format.
    n_datasets, n_methods = results_matrix.shape
    records = []
    for d in range(n_datasets):
        for m in range(n_methods):
            records.append({
                "dataset": d,
                "method": method_names[m],
                "value": results_matrix[d, m],
            })
    df_long = pd.DataFrame(records)

    p_matrix = sp.posthoc_nemenyi_friedman(
        df_long, y_col="value", group_col="method", block_col="dataset"
    )

    # Ensure the columns/index follow the requested method order.
    p_matrix = p_matrix.reindex(index=method_names, columns=method_names)
    return p_matrix


def critical_difference_data(results_matrix, method_names, alpha=0.05):
    """Compute average ranks and critical difference for a CD diagram.

    Parameters
    ----------
    results_matrix : array-like of shape (n_datasets, n_methods)
        Performance metric values (higher is better).
    method_names : list[str]
        Names for each method (column).
    alpha : float, default=0.05
        Significance level for the critical difference.

    Returns
    -------
    info : dict
        ``"avg_ranks"``
            Dict mapping method name to its average rank (lower is better).
        ``"cd"``
            The Nemenyi critical difference value.
        ``"n_datasets"``
            Number of datasets used.
        ``"n_methods"``
            Number of methods compared.
        ``"alpha"``
            Significance level used.
    """
    results_matrix = np.asarray(results_matrix, dtype=np.float64)
    method_names = list(method_names)
    n_datasets, n_methods = results_matrix.shape

    # Compute ranks per dataset (rank 1 = best = highest metric value).
    ranks = np.zeros_like(results_matrix)
    for i in range(n_datasets):
        ranks[i] = stats.rankdata(-results_matrix[i])

    avg_ranks = ranks.mean(axis=0)

    # Critical difference (Nemenyi test):
    #   CD = q_alpha * sqrt( k*(k+1) / (6*N) )
    # where q_alpha is the Studentized range critical value divided by sqrt(2).
    q_alpha = _nemenyi_critical_value(n_methods, alpha)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets))

    rank_dict = {name: float(rank) for name, rank in zip(method_names, avg_ranks)}

    return {
        "avg_ranks": rank_dict,
        "cd": float(cd),
        "n_datasets": n_datasets,
        "n_methods": n_methods,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Studentized range q-values / sqrt(2) for Nemenyi test at alpha = 0.05.
# Indexed by k (number of groups).  Values from Demsar (2006), Table 5.
_Q_ALPHA_005 = {
    2: 1.960,
    3: 2.344,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
}

_Q_ALPHA_010 = {
    2: 1.645,
    3: 2.052,
    4: 2.291,
    5: 2.460,
    6: 2.589,
    7: 2.693,
    8: 2.780,
    9: 2.855,
    10: 2.920,
    11: 2.978,
    12: 3.030,
    13: 3.077,
    14: 3.120,
    15: 3.159,
}


def _nemenyi_critical_value(k, alpha=0.05):
    """Look up the Nemenyi critical value q_alpha for *k* groups.

    Parameters
    ----------
    k : int
        Number of methods/groups.
    alpha : float
        Significance level (0.05 or 0.10 supported via lookup table;
        for other values a conservative scipy approximation is used).

    Returns
    -------
    q : float
        The critical value q_alpha / sqrt(2) used in the CD formula.
    """
    if alpha == 0.05 and k in _Q_ALPHA_005:
        return _Q_ALPHA_005[k]
    if alpha == 0.10 and k in _Q_ALPHA_010:
        return _Q_ALPHA_010[k]

    # Fallback: approximate using the Studentized range distribution.
    # q_alpha = q(alpha, k, inf) / sqrt(2)
    # scipy does not have a native Studentized range for inf df, so we use
    # the normal-based approximation via the Tukey distribution with large df.
    from scipy.stats import studentized_range

    q_raw = studentized_range.ppf(1 - alpha, k, df=1e6)
    return q_raw / np.sqrt(2)
