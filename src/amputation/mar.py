# -*- coding: utf-8 -*-
"""
mar.py — Missing At Random (MAR) amputation
============================================
Missingness of variable j depends on the *observed* values of the
other variables through a logistic model, not on the missing value
itself.  This is the standard assumption for most imputation methods.

Mechanism (per variable j):
    score_i = Σ_k≠j  w_k * x_ik   (linear combination of other columns)
    P(R_ij = 0) = sigmoid(α_j + score_i)
    where α_j is calibrated so that the marginal missing rate ≈ ``rate``.

Reference:
    Little, R.J.A. & Rubin, D.B. (2002).
    Statistical Analysis with Missing Data, 2nd ed.  Wiley.
"""

from __future__ import annotations

import numpy as np


def ampute_mar(
    X: np.ndarray,
    rate: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate MAR missing data via logistic dependence on observed predictors.

    For each variable j, the missingness probability of observation i is a
    logistic function of a weighted sum of the *other* columns.  The
    intercept α_j is calibrated to match the target ``rate`` on average.

    Args:
        X:    Complete data matrix ``(N, P)``.
        rate: Target marginal missing rate in ``(0, 1)``.
        seed: Optional integer seed.

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

    # Standardise columns once to stabilise logistic scores
    col_means = np.mean(X, axis=0)
    col_stds = np.std(X, axis=0, ddof=1)
    col_stds[col_stds < 1e-8] = 1.0
    X_std = (X - col_means) / col_stds

    for j in range(P):
        # Predictor columns: all except j
        pred_idx = [k for k in range(P) if k != j]
        if not pred_idx:
            # Single-column edge case: fall back to MCAR
            mask[:, j] = rng.random(N) < rate
            continue

        # Random weights ∈ [-1, 1]
        w = rng.uniform(-1.0, 1.0, len(pred_idx))
        score = X_std[:, pred_idx] @ w  # shape (N,)

        # Calibrate intercept α via bisection so that
        # mean(sigmoid(α + score)) ≈ rate
        alpha = _calibrate_intercept(score, rate)
        prob = _sigmoid(alpha + score)
        mask[:, j] = rng.random(N) < prob

    X_miss = X.astype(float, copy=True)
    X_miss[mask] = np.nan
    return X_miss, mask


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _calibrate_intercept(
    score: np.ndarray,
    target_rate: float,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """Find intercept α such that mean(sigmoid(α + score)) ≈ target_rate.

    Uses bisection search, which is exact and always converges.

    Args:
        score:       Linear predictor values, shape ``(N,)``.
        target_rate: Desired mean missing probability.
        tol:         Convergence tolerance on |mean(prob) - target|.
        max_iter:    Maximum bisection iterations.

    Returns:
        Calibrated float intercept α.
    """
    lo, hi = -20.0, 20.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        mean_prob = float(np.mean(_sigmoid(mid + score)))
        if abs(mean_prob - target_rate) < tol:
            return mid
        if mean_prob < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
