# -*- coding: utf-8 -*-
"""
preprocessing.py — Descriptive statistics, audit, and data quality checks
==========================================================================
Provides functions used in the pre-simulation audit phase:

  * Normality testing (Shapiro-Wilk)
  * Skewness / kurtosis
  * Quasi-constant variable detection
  * Full descriptive statistics table

Reference for normality and distributional testing:
    Shapiro & Wilk (1965) Biometrika 52(3–4):591–611.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import shapiro, skew, kurtosis, iqr


# ── Thresholds ────────────────────────────────────────────────────────────────
QUASI_CONSTANT_THRESHOLD: float = 0.98   # dominant category proportion
SHAPIRO_MAX_N: int = 5000                # Shapiro-Wilk valid up to ~5000 obs


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_descriptive_stats(X: np.ndarray, col_names: List[str],
                              continuous_idx: List[int]) -> pd.DataFrame:
    """Compute descriptive statistics for continuous variables.

    Metrics: N, Mean, SD, Median, IQR, Min, Max,
             Skewness, Excess-Kurtosis, Shapiro-W, Shapiro-p.

    Args:
        X:              Complete data matrix ``(N, P)`` (no NaN expected).
        col_names:      Column names (length P).
        continuous_idx: Indices of continuous columns.

    Returns:
        DataFrame with one row per continuous variable.
    """
    rows = []
    for j in continuous_idx:
        col = X[:, j]
        col_clean = col[~np.isnan(col)]
        n = len(col_clean)

        skewness = float(skew(col_clean)) if n >= 3 else np.nan
        kurt = float(kurtosis(col_clean)) if n >= 3 else np.nan

        sw_w, sw_p = np.nan, np.nan
        if 3 <= n <= SHAPIRO_MAX_N:
            sw_w, sw_p = shapiro(col_clean)

        rows.append({
            "Variable": col_names[j],
            "N": n,
            "Mean": round(float(np.mean(col_clean)), 4),
            "SD": round(float(np.std(col_clean, ddof=1)), 4),
            "Median": round(float(np.median(col_clean)), 4),
            "IQR": round(float(iqr(col_clean)), 4),
            "Min": round(float(np.min(col_clean)), 4),
            "Max": round(float(np.max(col_clean)), 4),
            "Skewness": round(skewness, 4),
            "ExKurtosis": round(kurt, 4),
            "Shapiro_W": round(float(sw_w), 4) if not np.isnan(sw_w) else np.nan,
            "Shapiro_p": round(float(sw_p), 4) if not np.isnan(sw_p) else np.nan,
            "Normal_p05": (sw_p > 0.05) if not np.isnan(sw_p) else None,
        })
    return pd.DataFrame(rows)


def detect_quasi_constant(X: np.ndarray, col_names: List[str],
                          threshold: float = QUASI_CONSTANT_THRESHOLD) -> List[str]:
    """Identify variables where a single value dominates.

    A variable is quasi-constant when the modal frequency >= ``threshold``.
    Such variables are uninformative predictors in imputation models and
    should be excluded from multivariate analyses (reported but not removed
    automatically — the caller decides).

    Args:
        X:         Complete data matrix ``(N, P)``.
        col_names: Column names.
        threshold: Proportion threshold (default 0.98).

    Returns:
        List of column names flagged as quasi-constant.
    """
    flagged: List[str] = []
    n = X.shape[0]
    for j, name in enumerate(col_names):
        col = X[:, j]
        col_clean = col[~np.isnan(col)]
        if len(col_clean) == 0:
            continue
        values, counts = np.unique(col_clean, return_counts=True)
        modal_freq = counts.max() / len(col_clean)
        if modal_freq >= threshold:
            flagged.append(name)
    return flagged


def audit_data(X: np.ndarray, col_names: List[str],
               continuous_idx: List[int],
               categorical_idx: List[int]) -> dict:
    """Run a full pre-simulation data quality audit.

    Checks:
      * Number of subjects and variables
      * Proportion of non-Gaussian continuous variables (Shapiro p < 0.05)
      * Quasi-constant variable detection
      * Native missing rate (expected 0 for the reference dataset)

    Args:
        X:               Complete data matrix ``(N, P)``.
        col_names:       Column names.
        continuous_idx:  Indices of continuous variables.
        categorical_idx: Indices of categorical (binary + ordinal) variables.

    Returns:
        dict with audit results suitable for JSON serialisation.
    """
    N, P = X.shape
    native_miss = int(np.isnan(X).sum())
    pct_native_miss = round(100.0 * native_miss / X.size, 2)

    # Normality
    n_non_gaussian = 0
    gaussian_flags: Dict[str, bool] = {}
    for j in continuous_idx:
        col = X[:, j]
        col_clean = col[~np.isnan(col)]
        n = len(col_clean)
        is_gaussian = True
        if 3 <= n <= SHAPIRO_MAX_N:
            _, p = shapiro(col_clean)
            is_gaussian = p > 0.05
        gaussian_flags[col_names[j]] = bool(is_gaussian)
        if not is_gaussian:
            n_non_gaussian += 1

    pct_non_gaussian = (
        round(100.0 * n_non_gaussian / len(continuous_idx), 1)
        if continuous_idx else 0.0
    )

    # Quasi-constant
    quasi_const = detect_quasi_constant(X, col_names)

    return {
        "N_subjects": N,
        "P_variables": P,
        "N_continuous": len(continuous_idx),
        "N_categorical": len(categorical_idx),
        "native_missing_cells": native_miss,
        "native_missing_pct": pct_native_miss,
        "pct_non_gaussian": pct_non_gaussian,
        "gaussian_by_variable": gaussian_flags,
        "quasi_constant_variables": quasi_const,
        "n_quasi_constant": len(quasi_const),
    }
