# -*- coding: utf-8 -*-
"""
rubin.py — Rubin's combination rules for multiple imputation
=============================================================
Rubin's rules allow valid inference to be drawn from M independently
imputed datasets.  The key components are:

  Q̄  = (1/M) Σ_m Q̂_m              (pooled point estimate)
  W  = (1/M) Σ_m W_m               (within-imputation variance)
  B  = 1/(M−1) Σ_m (Q̂_m − Q̄)²    (between-imputation variance)
  T  = W + (1 + 1/M) B             (total variance)
  λ  = (1 + 1/M) B / T             (fraction of missing information)
  ν  = (M − 1) / λ²                (degrees of freedom, Rubin 1987)
  FMI = λ                          (fraction of missing information)
  RE  = 1 / (1 + FMI/M)            (relative efficiency)

CRITICAL correctness requirement:
  - Each of the M datasets must contain the **same N individuals** as the
    original incomplete dataset.
  - Bootstrapping rows before imputation violates this requirement because
    the between-imputation variance B then reflects sampling variation
    rather than imputation uncertainty.

Reference:
    Rubin, D.B. (1987).
    Multiple Imputation for Nonresponse in Surveys.
    Wiley, New York.  Chapters 3–4.

    Barnard, J. & Rubin, D.B. (1999). Small-sample degrees of freedom with
    multiple imputation.  Biometrika 86(4):948–955.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def rubin_rules(
    imputed_datasets: List[np.ndarray],
    variable_indices: Optional[List[int]] = None,
    col_names: Optional[List[str]] = None,
) -> List[dict]:
    """Apply Rubin's combination rules to M imputed datasets.

    Computes pooled estimates of the column means and their associated
    variance components, following Rubin (1987) §3.3.

    Args:
        imputed_datasets: List of M complete arrays, each of shape ``(N, P)``.
                          All arrays must have identical shape and correspond
                          to the same N individuals (no bootstrapping).
        variable_indices: Indices of variables to process.
                          If None, all P columns are processed.
        col_names:        Optional column names for output labelling.

    Returns:
        List of dicts, one per variable, with keys:
        ``variable_idx``, ``variable_name``,
        ``Q_bar`` (pooled mean), ``W`` (within-var), ``B`` (between-var),
        ``T`` (total var), ``lambda`` (FMI), ``nu`` (df), ``RE``.

    Raises:
        ValueError: If M < 2.
        ValueError: If imputed datasets have different shapes.
    """
    M = len(imputed_datasets)
    if M < 2:
        raise ValueError(f"Rubin's rules require M >= 2 imputed datasets; got {M}.")

    shapes = {arr.shape for arr in imputed_datasets}
    if len(shapes) > 1:
        raise ValueError(f"All imputed datasets must have the same shape; found {shapes}.")

    N, P = imputed_datasets[0].shape
    idx = variable_indices if variable_indices is not None else list(range(P))

    # Stack to (M, N, P) for vectorised computation
    stack = np.stack(imputed_datasets, axis=0)  # (M, N, P)

    results = []
    for j in idx:
        col_stack = stack[:, :, j]  # (M, N)

        # Per-imputation means and variances
        Q_hat = np.mean(col_stack, axis=1)    # (M,) — dataset-level means
        W_m = np.var(col_stack, axis=1, ddof=1)  # (M,) — within-dataset variances

        Q_bar = float(np.mean(Q_hat))          # pooled mean
        W = float(np.mean(W_m))                # within-imputation variance

        # Between-imputation variance (must use ddof=1)
        B = float(np.var(Q_hat, ddof=1))       # between-imputation variance

        # Total variance (Rubin 1987, Eq. 3.1.5)
        T = W + (1.0 + 1.0 / M) * B

        # Fraction of missing information (FMI) ≈ λ
        lambda_ = ((1.0 + 1.0 / M) * B) / T if T > 1e-12 else 0.0

        # Degrees of freedom (Rubin 1987, Eq. 3.1.6)
        nu = (M - 1) / (lambda_ ** 2) if lambda_ > 1e-12 else np.inf

        # Relative efficiency (Rubin 1987, Eq. 4.2.8)
        RE = 1.0 / (1.0 + lambda_ / M)

        name = col_names[j] if col_names and j < len(col_names) else f"var_{j}"
        results.append({
            "variable_idx": j,
            "variable_name": name,
            "M": M,
            "Q_bar": round(Q_bar, 6),
            "W": round(W, 6),
            "B": round(B, 6),
            "T": round(T, 6),
            "lambda": round(float(lambda_), 6),
            "nu": round(float(nu), 2) if not np.isinf(nu) else "Inf",
            "FMI": round(float(lambda_), 6),
            "RE": round(float(RE), 6),
        })

    return results
