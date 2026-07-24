# -*- coding: utf-8 -*-
"""
knn_imputer.py — K-Nearest Neighbours imputation
=================================================
Each missing value is estimated as the weighted mean of the corresponding
values in the k most similar (nearest) observations, where distance is
computed using all available features (partial Euclidean distance).

The scikit-learn KNNImputer implements the exact formulation of
Troyanskaya et al. (2001): for each missing cell, the k donors are
selected based on the Euclidean distance over all *jointly observed*
features between the target row and candidate rows.

Reference:
    Troyanskaya, O. et al. (2001).
    Missing value estimation methods for DNA microarrays.
    Bioinformatics 17(6):520–525.
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import KNNImputer


def impute_knn(
    X_miss: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Impute missing values using K-Nearest Neighbours.

    Uses uniform (unweighted) donor averaging.  Standardisation of
    features before distance computation is not applied here so that
    the comparison is fair relative to the original variable scales;
    callers should standardise if desired.

    Args:
        X_miss: Incomplete data matrix ``(N, P)`` with NaN at missing cells.
        k:      Number of neighbours (default 5, as in Stekhoven 2012 comparison).

    Returns:
        X_imp: Complete matrix ``(N, P)`` with no NaN.

    Raises:
        ValueError: If k < 1.
        ValueError: If k >= N (not enough donors).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    N = X_miss.shape[0]
    if k >= N:
        raise ValueError(f"k ({k}) must be less than number of rows ({N}).")

    imputer = KNNImputer(n_neighbors=k, weights="uniform", metric="nan_euclidean")
    return imputer.fit_transform(X_miss.astype(float))
