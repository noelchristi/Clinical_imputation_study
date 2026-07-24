# -*- coding: utf-8 -*-
"""
mcar.py — Missing Completely At Random (MCAR) amputation
=========================================================
Each cell is independently masked with probability ``rate``,
regardless of any observed or unobserved value.  This is the
weakest and most favourable missingness assumption.

Reference:
    Rubin, D.B. (1976). Inference and missing data.
    Biometrika 63(3):581–592.
"""

from __future__ import annotations

import numpy as np


def ampute_mcar(
    X: np.ndarray,
    rate: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate MCAR missing data by independently masking each cell.

    Each cell (i, j) is set to NaN with probability ``rate``,
    independently of all other cells and of the cell value itself.

    Args:
        X:    Complete data matrix of shape ``(N, P)``.  Must contain no NaN.
        rate: Target marginal missing rate in the open interval ``(0, 1)``.
        seed: Optional integer seed for reproducibility.

    Returns:
        X_miss: Copy of ``X`` with NaN at masked positions, shape ``(N, P)``.
        mask:   Boolean array of shape ``(N, P)``; ``True`` where data are missing.

    Raises:
        ValueError: If ``rate`` is outside ``(0, 1)``.
        ValueError: If ``X`` contains NaN before amputation.
    """
    if not (0.0 < rate < 1.0):
        raise ValueError(f"rate must be in (0, 1); got {rate}.")
    if np.isnan(X).any():
        raise ValueError("Input matrix X must be complete (no NaN) before amputation.")

    rng = np.random.default_rng(seed)
    # Draw Bernoulli(rate) mask: True = missing
    mask = rng.random(X.shape) < rate
    X_miss = X.astype(float, copy=True)
    X_miss[mask] = np.nan
    return X_miss, mask
