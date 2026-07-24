# -*- coding: utf-8 -*-
"""
nonparametric.py — Friedman test, Nemenyi post-hoc, Holm correction
=====================================================================
Non-parametric statistical comparison of k imputation methods across
N variables (or scenarios), following the benchmark evaluation framework
of Demšar (2006).

Pipeline:
  1. Compute per-variable method rankings (rank 1 = best).
  2. Friedman test: H₀ = all methods have equal average ranks.
  3. If significant: Nemenyi post-hoc pairwise comparisons with
     Holm step-down correction.
  4. Compute critical difference (CD) for CD diagram.

References:
    Demšar, J. (2006). Statistical comparisons of classifiers over
    multiple data sets.
    Journal of Machine Learning Research 7:1–30.

    Friedman, M. (1940). A comparison of alternative tests of
    significance for the problem of m rankings.
    Annals of Mathematical Statistics 11(1):86–92.

    Holm, S. (1979). A simple sequentially rejective multiple test
    procedure.  Scandinavian Journal of Statistics 6(2):65–70.
"""

from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np
from scipy.stats import chi2, friedmanchisquare, norm


def friedman_test(rankings: np.ndarray) -> dict:
    """Friedman test for differences among k methods ranked over N datasets.

    Args:
        rankings: Array of shape ``(N, k)`` containing integer ranks
                  (rank 1 = best method).  Ties should be resolved by
                  averaging before calling this function.

    Returns:
        dict with keys: ``chi2``, ``p_value``, ``N``, ``k``,
        ``eta_sq`` (η² = effect size), ``avg_ranks``.

    Raises:
        ValueError: If ``rankings`` has fewer than 3 rows or 2 columns.
    """
    N, k = rankings.shape
    if N < 3:
        raise ValueError(f"Friedman test requires N >= 3 datasets; got {N}.")
    if k < 2:
        raise ValueError(f"Friedman test requires k >= 2 methods; got {k}.")

    cols = [rankings[:, j] for j in range(k)]
    chi2_stat, p_val = friedmanchisquare(*cols)

    # η² = χ²_F / [N(k−1)] — effect size for Friedman test
    # Reference: Kendall (1970) Rank Correlation Methods, §6.5
    eta_sq = float(chi2_stat) / (N * (k - 1))

    avg_ranks = rankings.mean(axis=0).tolist()

    return {
        "chi2": float(chi2_stat),
        "p_value": float(p_val),
        "N": N,
        "k": k,
        "eta_sq": float(eta_sq),
        "avg_ranks": avg_ranks,
    }


def nemenyi_posthoc(
    avg_ranks: np.ndarray,
    method_names: List[str],
    N: int,
    alpha: float = 0.05,
) -> List[dict]:
    """Nemenyi pairwise post-hoc test with normal approximation.

    Computes all k(k-1)/2 pairwise z-statistics and two-tailed p-values
    from the rank differences, then applies Holm step-down correction.

    The SE of rank differences under H₀ is (Demšar 2006, Eq. 5)::

        SE = sqrt( k(k+1) / 6N )

    Args:
        avg_ranks:    Array of mean ranks, shape ``(k,)``.
        method_names: Method labels (length k).
        N:            Number of datasets (variables / scenarios).
        alpha:        Family-wise error rate (default 0.05).

    Returns:
        List of dicts sorted by p-value (ascending), each with:
        ``method_A``, ``method_B``, ``rank_diff``, ``z``, ``p_raw``,
        ``p_holm``, ``significant``.
    """
    k = len(avg_ranks)
    se = np.sqrt(k * (k + 1) / (6.0 * N))

    pairs = []
    for i, j in combinations(range(k), 2):
        diff = abs(float(avg_ranks[i]) - float(avg_ranks[j]))
        z = diff / se
        p_raw = 2.0 * (1.0 - norm.cdf(abs(z)))
        pairs.append({
            "method_A": method_names[i],
            "method_B": method_names[j],
            "rank_diff": round(diff, 4),
            "z": round(z, 4),
            "p_raw": float(p_raw),
        })

    # Holm step-down correction
    pairs.sort(key=lambda x: x["p_raw"])
    m = len(pairs)
    for rank, pair in enumerate(pairs, start=1):
        pair["p_holm"] = min(1.0, pair["p_raw"] * (m - rank + 1))
        pair["significant"] = pair["p_holm"] < alpha

    return pairs


def nemenyi_cd(k: int, N: int, alpha: float = 0.05) -> float:
    """Critical Difference (CD) for the Nemenyi test (Demšar 2006, Table 5).

    CD = q_α × sqrt( k(k+1) / 6N )

    where q_α is the critical value of the Studentised range distribution
    divided by sqrt(2), tabulated in Demšar (2006) for α = 0.05 and 0.10.

    Args:
        k:     Number of methods.
        N:     Number of datasets.
        alpha: Significance level (0.05 or 0.10).

    Returns:
        Critical difference value.
    """
    # q_α values from Demšar (2006), Table 5 (for α = 0.05)
    q_table_05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_table_10 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459,
        6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920,
    }
    q_table = q_table_05 if abs(alpha - 0.05) < 0.01 else q_table_10
    q = q_table.get(k, q_table_05.get(min(k, 10), 2.569))
    return float(q * np.sqrt(k * (k + 1) / (6.0 * N)))


def compute_rankings(
    values: np.ndarray,
    lower_is_better: bool = True,
) -> np.ndarray:
    """Convert a (N, k) performance matrix to integer ranks (1 = best).

    Args:
        values:           Performance matrix ``(N, k)`` where N = variables
                          and k = methods.
        lower_is_better:  If True (default), rank 1 = lowest value.

    Returns:
        Integer rank matrix ``(N, k)``, same shape as ``values``.
        Ties are resolved by averaging (fractional ranks).
    """
    from scipy.stats import rankdata

    N, k = values.shape
    ranks = np.empty_like(values, dtype=float)
    for i in range(N):
        row = values[i]
        if lower_is_better:
            ranks[i] = rankdata(row, method="average")
        else:
            ranks[i] = rankdata(-row, method="average")
    return ranks
