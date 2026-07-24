# -*- coding: utf-8 -*-
"""
categorical_metrics.py — Reconstruction metrics for categorical variables
=========================================================================
All metrics are computed on the imputed cells only (cells that were
set to NaN during amputation), matching the approach for continuous metrics.

Metrics:
  - F1 (macro-weighted)
  - Accuracy
  - Cohen's κ (Kappa)
  - MCC (Matthews Correlation Coefficient)
  - Chi-squared test on the full-column contingency table (distributional)

References:
    Cohen, J. (1960). A coefficient of agreement for nominal scales.
    Educational and Psychological Measurement 20(1):37–46.

    Matthews, B.W. (1975). Comparison of the predicted and observed
    secondary structure of T4 phage lysozyme.
    BBA - Protein Structure 405(2):442–451.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)


def evaluate_categorical_variable(
    X_true: np.ndarray,
    X_imp: np.ndarray,
    col_mask: np.ndarray,
    var_idx: int,
) -> dict:
    """Compute all categorical metrics for one variable at one scenario.

    Args:
        X_true:    Complete reference matrix ``(N, P)``.
        X_imp:     Imputed matrix ``(N, P)``.
        col_mask:  Boolean vector ``(N,)`` — True where the cell was imputed.
        var_idx:   Column index of the variable.

    Returns:
        Dictionary of metric name → float.  Empty dict if < 2 imputed cells.
    """
    n_imputed = int(col_mask.sum())
    if n_imputed < 2:
        return {}

    # ── Only imputed cells ─────────────────────────────────────────────────
    max_val = int(np.nanmax(X_true[:, var_idx]))
    true_cat = X_true[col_mask, var_idx].astype(int)
    imp_cat = np.clip(np.round(X_imp[col_mask, var_idx]), 0, max_val).astype(int)

    f1 = float(f1_score(true_cat, imp_cat, average="weighted", zero_division=0))
    acc = float(accuracy_score(true_cat, imp_cat))

    kappa = np.nan
    try:
        kappa = float(cohen_kappa_score(true_cat, imp_cat))
    except Exception:
        pass

    mcc = np.nan
    try:
        mcc = float(matthews_corrcoef(true_cat, imp_cat))
    except Exception:
        pass

    # Chi-squared on full column (distributional fidelity)
    chi2_p = np.nan
    n_classes = len(np.unique(true_cat))
    if n_classes >= 2:
        try:
            all_true = X_true[:, var_idx].astype(int)
            all_imp = np.clip(np.round(X_imp[:, var_idx]), 0, max_val).astype(int)
            ct = _contingency_from_arrays(all_true, all_imp)
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                _, chi2_p, _, _ = chi2_contingency(ct)
        except Exception:
            pass

    return {
        "F1_weighted": f1,
        "Accuracy": acc,
        "Cohens_Kappa": kappa,
        "MCC": mcc,
        "Chi2_pvalue": float(chi2_p) if not np.isnan(chi2_p) else np.nan,
    }


def _contingency_from_arrays(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build contingency table without pandas overhead."""
    _, x_inv = np.unique(x, return_inverse=True)
    _, y_inv = np.unique(y, return_inverse=True)
    ct = np.zeros((x_inv.max() + 1, y_inv.max() + 1), dtype=np.int64)
    np.add.at(ct, (x_inv, y_inv), 1)
    return ct


def evaluate_all_categorical(
    X_true: np.ndarray,
    X_imp: np.ndarray,
    mask: np.ndarray,
    categorical_idx: List[int],
    col_names: List[str],
) -> List[dict]:
    """Evaluate all categorical variables and return a list of result records.

    Args:
        X_true:          Complete reference matrix ``(N, P)``.
        X_imp:           Imputed matrix ``(N, P)``.
        mask:            Boolean mask ``(N, P)`` — True where data were missing.
        categorical_idx: Indices of categorical variables to evaluate.
        col_names:       Column names (length P).

    Returns:
        List of dicts, one per categorical variable.
    """
    results = []
    for j in categorical_idx:
        col_mask = mask[:, j]
        if col_mask.sum() < 2:
            continue
        metrics = evaluate_categorical_variable(X_true, X_imp, col_mask, j)
        if metrics:
            metrics["variable"] = col_names[j]
            metrics["var_idx"] = j
            results.append(metrics)
    return results
