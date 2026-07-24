# -*- coding: utf-8 -*-
"""
variance_correlation.py — Variance ratio and correlation preservation
======================================================================

Variance Ratio (R_V)
--------------------
Two complementary metrics for variance preservation:

1. **Cell-level variance ratio** (R_V_cell):
   Measures variance preservation on imputed cells only::

       R_V_cell = Var(X_imp[mask, j]) / Var(X_true[mask, j])

   where mask indicates imputed positions.
   
   - R_V = 1  → perfect variance preservation on imputed cells
   - R_V < 1  → variance underestimation (common for Median/Mode, KNN)
   - R_V > 1  → variance inflation (can occur with MICE under MNAR)
   - R_V = 0  → deterministic imputation (Median/Mode)

2. **Global variance ratio** (R_V_global):
   Measures variance preservation on the complete column after imputation::

       R_V_global = Var(X_imp[:, j]) / Var(X_true[:, j])

   This metric captures the overall distributional impact of imputation
   by comparing the full reconstructed column to the original complete data.
   
   - R_V_global ≈ 1  → imputation preserves global variance
   - R_V_global < 1  → global variance attenuation
   - R_V_global > 1  → global variance inflation

Both metrics are complementary and should be reported together:
- Cell-level R_V: assesses quality of imputed values themselves
- Global R_V: assesses overall distributional impact

API Compatibility:
  - ``mean_R_V``: alias for ``mean_R_V_cell`` (backward compatibility)
  - ``mean_R_V_cell``: mean cell-level variance ratio
  - ``mean_R_V_global``: mean global variance ratio (new in v4.1)

Correlation Preservation (Δr)
------------------------------
The Pearson correlation matrix computed on the imputed dataset minus
the correlation matrix of the original complete data::

    Δr[i,j] = r_imputed[i,j] − r_original[i,j]

Values near zero indicate good correlation preservation.

References:
    White, I.R., Royston, P. & Wood, A.M. (2011).
    Multiple imputation using chained equations.
    Statistics in Medicine 30(4):377–399.
"""

from __future__ import annotations

from typing import List

import numpy as np


def variance_ratio(
    X_true: np.ndarray,
    X_imp: np.ndarray,
    mask: np.ndarray,
    continuous_idx: List[int],
    min_cells: int = 3,
) -> dict:
    """Compute per-variable and aggregate variance ratios (cell-level & global).

    Returns two complementary metrics:
    
    1. **Cell-level R_V**: Variance ratio computed on imputed cells only.
       Assesses the quality of imputed values themselves.
    
    2. **Global R_V**: Variance ratio computed on the complete column.
       Assesses the overall distributional impact of imputation.

    Args:
        X_true:         Complete reference matrix ``(N, P)``.
        X_imp:          Imputed matrix ``(N, P)``.
        mask:           Boolean mask ``(N, P)`` — True where imputed.
        continuous_idx: Indices of continuous variables.
        min_cells:      Minimum imputed cells needed to compute cell-level
                        variance (≥ 3 required for unbiased sample variance).

    Returns:
        dict with keys:
          ``"per_variable"``: list of dicts with columns:
              - var_idx: column index
              - R_V_cell: cell-level variance ratio (on imputed positions)
              - R_V_global: global variance ratio (on full column)
          ``"mean_R_V"``: alias for mean_R_V_cell (backward compatibility).
          ``"mean_R_V_cell"``: mean cell-level R_V across eligible variables.
          ``"mean_R_V_global"``: mean global R_V across all variables.
    """
    per_var = []
    for j in continuous_idx:
        col_mask = mask[:, j]
        n_imp = int(col_mask.sum())
        
        # ── Global variance ratio (full column) ──────────────────────────
        var_true_global = float(np.var(X_true[:, j], ddof=1))
        var_imp_global = float(np.var(X_imp[:, j], ddof=1))
        
        if var_true_global < 1e-12:
            # Reference variance is (nearly) zero: ratio is undefined
            continue
        
        r_v_global = var_imp_global / var_true_global
        
        # ── Cell-level variance ratio (imputed cells only) ───────────────
        r_v_cell = np.nan
        if n_imp >= min_cells:
            var_true_cell = float(np.var(X_true[col_mask, j], ddof=1))
            var_imp_cell = float(np.var(X_imp[col_mask, j], ddof=1))
            
            if var_true_cell >= 1e-12:
                r_v_cell = var_imp_cell / var_true_cell
        
        per_var.append({
            "var_idx": j,
            "R_V_cell": r_v_cell,
            "R_V_global": r_v_global,
        })

    # Aggregate metrics
    cell_vals = [x["R_V_cell"] for x in per_var if not np.isnan(x["R_V_cell"])]
    global_vals = [x["R_V_global"] for x in per_var]
    
    mean_r_v_cell = float(np.mean(cell_vals)) if cell_vals else np.nan
    mean_r_v_global = float(np.mean(global_vals)) if global_vals else np.nan
    
    return {
        "per_variable": per_var,
        "mean_R_V": mean_r_v_cell,           # backward compatibility
        "mean_R_V_cell": mean_r_v_cell,
        "mean_R_V_global": mean_r_v_global,
    }


def correlation_differences(
    X_orig: np.ndarray,
    X_imp: np.ndarray,
    var_indices: List[int],
) -> np.ndarray:
    """Compute element-wise correlation difference matrix Δr = r_imp − r_orig.

    Both correlation matrices are computed on the **complete columns** of
    the respective datasets using Pearson correlation.  This measures the
    degree to which the imputed dataset distorts the inter-variable
    relationships present in the original data.

    Args:
        X_orig:      Original complete matrix ``(N, P)``.
        X_imp:       Imputed matrix ``(N, P)``.
        var_indices: Indices of the variables to include (typically continuous).

    Returns:
        Square numpy array ``(k, k)`` where k = len(var_indices).
        Entry [i, j] = r_imp[i,j] − r_orig[i,j].
    """
    if len(var_indices) < 2:
        return np.array([[0.0]])

    orig_sub = X_orig[:, var_indices].astype(float)
    imp_sub = X_imp[:, var_indices].astype(float)

    # numpy corrcoef returns the full correlation matrix
    r_orig = np.corrcoef(orig_sub, rowvar=False)
    r_imp = np.corrcoef(imp_sub, rowvar=False)

    return r_imp - r_orig


def mean_absolute_correlation_error(delta_r: np.ndarray) -> float:
    """Mean absolute off-diagonal element of Δr.

    A scalar summary of correlation preservation quality.
    Perfect preservation → 0.0.

    Args:
        delta_r: Square correlation difference matrix ``(k, k)``.

    Returns:
        Mean |Δr| over all k(k-1)/2 upper-triangle pairs.
    """
    k = delta_r.shape[0]
    if k < 2:
        return 0.0
    upper_idx = np.triu_indices(k, k=1)
    return float(np.mean(np.abs(delta_r[upper_idx])))
