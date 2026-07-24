# -*- coding: utf-8 -*-
"""
baseline.py — Median/Mode imputation (baseline method)
=======================================================
Simple univariate imputation: each missing value is replaced by the
column median (continuous variables) or the column mode (categorical
variables).  This is the standard baseline in imputation benchmarks.

No inter-variable relationships are modelled.  This method is known to
underestimate variance and distort correlations when missing rates are
non-trivial (Sterne et al. 2009).

Reference:
    Sterne, J.A.C. et al. (2009). Multiple imputation for missing data in
    epidemiological and clinical research: potential and pitfalls.
    BMJ 338:b2393.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.impute import SimpleImputer


def impute_median_mode(
    X_miss: np.ndarray,
    continuous_idx: List[int],
    categorical_idx: List[int],
) -> np.ndarray:
    """Replace missing values with column median (continuous) or mode (categorical).

    Args:
        X_miss:          Incomplete data matrix ``(N, P)`` with NaN at missing cells.
        continuous_idx:  Column indices of continuous variables.
        categorical_idx: Column indices of binary/ordinal variables.

    Returns:
        X_imp: Complete matrix ``(N, P)`` with no NaN.

    Raises:
        ValueError: If a column index appears in both lists.
    """
    overlap = set(continuous_idx) & set(categorical_idx)
    if overlap:
        raise ValueError(f"Column indices {overlap} appear in both continuous and categorical lists.")

    X_imp = X_miss.astype(float, copy=True)

    if continuous_idx:
        imp_cont = SimpleImputer(strategy="median")
        X_imp[:, continuous_idx] = imp_cont.fit_transform(X_miss[:, continuous_idx])

    if categorical_idx:
        imp_cat = SimpleImputer(strategy="most_frequent")
        X_imp[:, categorical_idx] = imp_cat.fit_transform(X_miss[:, categorical_idx])

    return X_imp
