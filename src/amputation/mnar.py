# -*- coding: utf-8 -*-
"""
mnar.py — Missing Not At Random (MNAR) amputation
===================================================
Missingness of variable j depends on the *unobserved* value of j itself.
This is the most challenging scenario because standard imputation methods
(which assume MAR) are not valid under MNAR.

Mechanism (standard self-selection model):
    P(R_ij = 0 | X_ij) = sigmoid(α_j + β * z_ij)
    where z_ij = (X_ij - μ_j) / σ_j  (standardised value of X_ij itself)
    and β = 2.0 (moderate effect, widely used in simulation studies).

    α_j is calibrated by bisection to achieve the marginal target ``rate``.

This is the canonical MNAR "self-selection" model used in simulation
benchmarks (Little 1988; Sterne et al. 2009).

References:
    Little, R.J.A. (1988). A test of missing completely at random for
    multivariate data with missing values.
    JASA 83(404):1198–1202.

    Sterne, J.A.C. et al. (2009). Multiple imputation for missing data in
    epidemiological and clinical research.  BMJ 338:b2393.
"""

from __future__ import annotations

import numpy as np

# Fixed logistic slope — moderate MNAR effect
# β = 2 gives P ≈ 0.88 at z = +2 (values far above mean are more likely missing)
MNAR_BETA: float = 2.0


def ampute_mnar(
    X: np.ndarray,
    rate: float,
    seed: int | None = None,
    beta: float = MNAR_BETA,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate MNAR missing data via logistic self-selection model.

    For each variable j, the probability that observation i is missing
    is a logistic function of the *standardised value of X_ij itself*::

        P(R_ij = 0) = sigmoid(α_j + β * (X_ij - μ_j) / σ_j)

    Higher values of X_ij are systematically more likely to be missing.
    The intercept α_j is calibrated to achieve the target marginal rate.

    Args:
        X:    Complete data matrix ``(N, P)``.
        rate: Target marginal missing rate in ``(0, 1)``.
        seed: Optional integer seed.
        beta: Logistic slope controlling MNAR severity (default 2.0).

    Returns:
        X_miss: Array with NaN at missing positions, shape ``(N, P)``.
        mask:   Boolean mask ``(N, P)``, True where missing.

    Raises:
        ValueError: If ``rate`` is outside ``(0, 1)``.
        ValueError: If ``X`` contains NaN.
    """
    if not (0.0 < rate < 1.0):
        raise ValueError(f"rate must be in (0, 1); got {rate}.")
    if np.isnan(X).any():
        raise ValueError("Input matrix X must be complete before amputation.")

    rng = np.random.default_rng(seed)
    N, P = X.shape
    mask = np.zeros((N, P), dtype=bool)

    for j in range(P):
        col = X[:, j].astype(float)
        mu = np.mean(col)
        sigma = np.std(col, ddof=1)
        if sigma < 1e-8:
            # Constant column — fall back to MCAR
            mask[:, j] = rng.random(N) < rate
            continue

        # Standardised score for MNAR
        z = (col - mu) / sigma

        # Calibrate intercept so that mean(sigmoid(α + β*z)) ≈ rate
        alpha = _calibrate_intercept(beta * z, rate)
        prob = _sigmoid(alpha + beta * z)
        mask[:, j] = rng.random(N) < prob

    X_miss = X.astype(float, copy=True)
    X_miss[mask] = np.nan
    return X_miss, mask


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _calibrate_intercept(
    score: np.ndarray,
    target: float,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """Bisection search for intercept α such that mean(sigmoid(α + score)) = target.

    Args:
        score:    Linear predictor ``β * z``, shape ``(N,)``.
        target:   Target mean missing probability.
        tol:      Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        Float intercept α.
    """
    lo, hi = -30.0, 30.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        mean_p = float(np.mean(_sigmoid(mid + score)))
        if abs(mean_p - target) < tol:
            return mid
        if mean_p < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
