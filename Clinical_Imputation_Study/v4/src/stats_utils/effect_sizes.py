# -*- coding: utf-8 -*-
"""
effect_sizes.py — Effect size measures for imputation comparison
================================================================
All effect sizes implemented here follow their original published
definitions.  No placeholder values are used.

Implemented:
  - Cohen's d (standardised mean difference for paired samples)
  - Cliff's δ (non-parametric probability-based effect size)
  - η² for Kruskal-Wallis test
  - η² for Friedman test

References:
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral
    Sciences (2nd ed.).  Lawrence Erlbaum Associates.

    Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer
    ordinal questions.  Psychological Bulletin 114(3):494–509.

    Tomczak, M. & Tomczak, E. (2014). The need to report effect size
    estimates revisited. An overview of some recommended measures of
    effect size.  Trends in Sport Sciences 1(21):19–25.

    Kendall, M.G. (1970). Rank Correlation Methods (4th ed.).
    Griffin, London.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.stats import kruskal


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for two independent samples.

    Computes the pooled standard deviation estimator::

        d = (μ_x − μ_y) / s_pooled

    where s_pooled = sqrt( [(n_x-1)s_x² + (n_y-1)s_y²] / (n_x + n_y - 2) )

    Range: (−∞, +∞).  |d| = 0.2 small, 0.5 medium, 0.8 large (Cohen 1988).

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        Cohen's d (float).  Returns 0.0 if pooled SD is near zero.

    Raises:
        ValueError: If either array has fewer than 2 elements.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Cohen's d requires at least 2 observations per group.")

    n_x, n_y = len(x), len(y)
    s2_x = np.var(x, ddof=1)
    s2_y = np.var(y, ddof=1)
    s_pooled = np.sqrt(((n_x - 1) * s2_x + (n_y - 1) * s2_y) / (n_x + n_y - 2))

    if s_pooled < 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / s_pooled)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's δ: non-parametric dominance-based effect size.

    Computed as::

        δ = P(X > Y) − P(Y > X)
          = [Σ_{i,j} sign(x_i − y_j)] / (n_x × n_y)

    Range: [−1, +1].
      δ =  1  → x always greater than y
      δ = −1  → y always greater than x
      δ =  0  → no dominance (distributions identical)

    Magnitude thresholds (Cliff 1993; Romano et al. 2006):
      |δ| < 0.147 → negligible
      |δ| < 0.330 → small
      |δ| < 0.474 → medium
      |δ| ≥ 0.474 → large

    Args:
        x: First sample array.
        y: Second sample array.

    Returns:
        Cliff's δ ∈ [−1, 1].

    Raises:
        ValueError: If either array is empty.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Cliff's delta requires non-empty arrays.")

    # Efficient computation via broadcasting: sign matrix (n_x, n_y)
    sign_matrix = np.sign(x[:, None] - y[None, :])
    delta = float(sign_matrix.mean())
    return delta


def cliffs_delta_magnitude(delta: float) -> str:
    """Return the verbal magnitude of Cliff's δ.

    Reference: Romano et al. (2006) ESEM — magnitude thresholds.

    Args:
        delta: Cliff's δ value.

    Returns:
        String: ``"negligible"``, ``"small"``, ``"medium"``, or ``"large"``.
    """
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    elif d < 0.330:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


def eta_squared_kruskal(groups: List[np.ndarray]) -> float:
    """η² effect size for Kruskal-Wallis test.

    Formula (Tomczak & Tomczak 2014; Eq. 6)::

        η² = (H − k + 1) / (n − k)

    where H is the Kruskal-Wallis statistic, k the number of groups,
    and n the total number of observations.

    Args:
        groups: List of k arrays, each containing group observations.

    Returns:
        η² ∈ [0, 1].  Returns NaN if the KW test cannot be computed.

    Raises:
        ValueError: If fewer than 2 groups are provided.
    """
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis η² requires at least 2 groups.")

    k = len(groups)
    n = sum(len(g) for g in groups)
    if n <= k:
        return np.nan

    try:
        H, _ = kruskal(*groups)
    except Exception:
        return np.nan

    eta_sq = (H - k + 1) / (n - k)
    return float(max(0.0, eta_sq))


def eta_squared_friedman(chi2_stat: float, N: int, k: int) -> float:
    """η² effect size for Friedman test.

    Formula (Kendall 1970, §6.5)::

        η²_F = χ²_F / [N(k − 1)]

    Range: [0, 1].  Values ≥ 0.26 are considered large (Cohen 1988 convention
    adapted for non-parametric tests).

    Args:
        chi2_stat: Friedman χ² statistic.
        N:         Number of datasets (rows in the rank matrix).
        k:         Number of methods (columns).

    Returns:
        η² ∈ [0, 1].

    Raises:
        ValueError: If N < 2 or k < 2.
    """
    if N < 2 or k < 2:
        raise ValueError("N >= 2 and k >= 2 are required for Friedman η².")
    denom = N * (k - 1)
    if denom < 1e-12:
        return np.nan
    return float(min(1.0, max(0.0, chi2_stat / denom)))
