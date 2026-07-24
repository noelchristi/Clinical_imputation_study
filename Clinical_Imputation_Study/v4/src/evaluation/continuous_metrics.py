# -*- coding: utf-8 -*-
"""
continuous_metrics.py — Reconstruction metrics for continuous variables
========================================================================
All metrics are computed **exclusively on imputed cells** (i.e., the
cells that were set to NaN by the amputation step).  This is the only
statistically valid comparison: the observed cells are identical across
all methods and must not dilute the metric (bug identified in audit).

Metrics implemented:
  - RMSE   (Root Mean Squared Error)
  - NRMSE  (Normalised RMSE, range-normalised — Stekhoven 2012)
  - MAE    (Mean Absolute Error)
  - Bias   (mean signed error: Ê[X_imp - X_true])
  - KS     (Kolmogorov-Smirnov two-sample test on the full column distribution)
  - Wilcoxon (paired signed-rank test on imputed cells)

References:
    Stekhoven & Bühlmann (2012) Bioinformatics 28(1):112–118.
    Wilcoxon, F. (1945). Individual comparisons by ranking methods.
    Biometrics Bulletin 1(6):80–83.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.stats import ks_2samp, wilcoxon
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_continuous_variable(
    X_true: np.ndarray,
    X_imp: np.ndarray,
    col_mask: np.ndarray,
    var_idx: int,
) -> dict:
    """Compute all continuous metrics for one variable at one scenario.

    Args:
        X_true:    Complete reference matrix ``(N, P)``.
        X_imp:     Imputed matrix ``(N, P)``.
        col_mask:  Boolean vector ``(N,)`` — True where the cell was amputed
                   (i.e., was missing and therefore imputed).
        var_idx:   Column index of the variable being evaluated.

    Returns:
        Dictionary of metric name → float.  Returns empty dict if fewer
        than 2 imputed cells (metric undefined).
    """
    n_imputed = int(col_mask.sum())
    if n_imputed < 2:
        return {}

    # ── Only imputed cells ─────────────────────────────────────────────────
    true_vals = X_true[col_mask, var_idx]
    imp_vals = X_imp[col_mask, var_idx]

    # RMSE
    rmse = float(np.sqrt(mean_squared_error(true_vals, imp_vals)))

    # NRMSE: range-normalised (Stekhoven 2012)
    col_range = float(np.max(X_true[:, var_idx]) - np.min(X_true[:, var_idx]))
    nrmse = rmse / col_range if col_range > 1e-8 else np.nan

    # MAE
    mae = float(mean_absolute_error(true_vals, imp_vals))

    # Bias = mean(imputed - true)
    bias = float(np.mean(imp_vals - true_vals))

    # KS test on full column distributions (distributional fidelity)
    ks_stat, ks_p = np.nan, np.nan
    try:
        ks_stat, ks_p = ks_2samp(X_true[:, var_idx], X_imp[:, var_idx])
    except Exception:
        pass

    # Wilcoxon paired signed-rank test on imputed cells
    wilcox_stat, wilcox_p = np.nan, np.nan
    if n_imputed >= 5:
        try:
            diff = imp_vals - true_vals
            if np.any(diff != 0):
                wilcox_stat, wilcox_p = wilcoxon(
                    true_vals, imp_vals, zero_method="zsplit"
                )
            else:
                wilcox_p = 1.0
        except Exception:
            pass

    return {
        "RMSE": rmse,
        "NRMSE": nrmse,
        "MAE": mae,
        "Bias": bias,
        "KS_stat": float(ks_stat) if not np.isnan(ks_stat) else np.nan,
        "KS_pvalue": float(ks_p) if not np.isnan(ks_p) else np.nan,
        "Wilcoxon_pvalue": float(wilcox_p) if not np.isnan(wilcox_p) else np.nan,
    }


def evaluate_all_continuous(
    X_true: np.ndarray,
    X_imp: np.ndarray,
    mask: np.ndarray,
    continuous_idx: List[int],
    col_names: List[str],
) -> List[dict]:
    """Evaluate all continuous variables and return a list of result records.

    Args:
        X_true:         Complete reference matrix ``(N, P)``.
        X_imp:          Imputed matrix ``(N, P)``.
        mask:           Boolean mask ``(N, P)`` — True where data were missing.
        continuous_idx: Indices of continuous variables to evaluate.
        col_names:      Column names (length P).

    Returns:
        List of dicts, one per continuous variable (includes variable name).
    """
    results = []
    for j in continuous_idx:
        col_mask = mask[:, j]
        if col_mask.sum() < 2:
            continue
        metrics = evaluate_continuous_variable(X_true, X_imp, col_mask, j)
        if metrics:
            metrics["variable"] = col_names[j]
            metrics["var_idx"] = j
            results.append(metrics)
    return results
