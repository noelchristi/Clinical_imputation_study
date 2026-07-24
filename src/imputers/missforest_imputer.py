# -*- coding: utf-8 -*-
"""
missforest_imputer.py — True MissForest algorithm
==================================================
Faithful implementation of the MissForest algorithm as described in
Stekhoven & Bühlmann (2012).

Key algorithmic properties that distinguish true MissForest from
``IterativeImputer(RandomForestRegressor)``:

  1. **Separate model per variable per iteration**: a fresh Random Forest
     is trained for each column at each iteration, using the current
     imputed state of all other columns as predictors.

  2. **Type-aware prediction**:
     - ``RandomForestRegressor`` for continuous variables.
     - ``RandomForestClassifier`` for binary and ordinal variables.
     Using a regressor for categorical variables is methodologically
     incorrect (predicting 0/1/2/3 as a continuous number).

  3. **Convergence criterion** (Stekhoven 2012, Eq. 1):
     - γ_cont = Σ(X_new − X_old)² / Σ(X_new)²   (continuous variables)
     - γ_cat  = # disagreements / # categorical missing cells
     The algorithm stops when either criterion stops decreasing.
     The *previous* iterate is returned (not the last, potentially
     diverged, one).

  4. **Variable ordering**: variables are updated in ascending order of
     their missing count (least missing first), which provides better
     initial estimates for more heavily missing variables.

Reference:
    Stekhoven, D.J. & Bühlmann, P. (2012).
    MissForest — non-parametric missing value imputation for mixed-type
    data.  Bioinformatics 28(1):112–118.
    https://doi.org/10.1093/bioinformatics/btr597
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def impute_missforest(
    X_miss: np.ndarray,
    continuous_idx: List[int],
    categorical_idx: List[int],
    n_estimators: int = 100,
    max_depth: int | None = None,
    max_iter: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """True MissForest imputation (Stekhoven & Bühlmann 2012).

    Args:
        X_miss:          Incomplete data matrix ``(N, P)`` with NaN.
        continuous_idx:  Indices of continuous variables
                         (RandomForestRegressor used).
        categorical_idx: Indices of binary/ordinal variables
                         (RandomForestClassifier used).
        n_estimators:    Number of trees per forest (default 100).
        max_depth:       Maximum tree depth (``None`` = fully grown trees).
        max_iter:        Maximum imputation cycles (default 10).
        random_state:    Seed for the random forests.

    Returns:
        X_imp: Complete matrix ``(N, P)`` with no NaN.

    Raises:
        ValueError: If column indices overlap between type lists.
    """
    overlap = set(continuous_idx) & set(categorical_idx)
    if overlap:
        raise ValueError(f"Indices {overlap} appear in both continuous and categorical lists.")

    # ── Step 1: Initial fill (mean/mode) ─────────────────────────────────────
    X_imp = _initial_fill(X_miss, continuous_idx, categorical_idx)

    # ── Step 2: Sort variables by missing count (ascending) ──────────────────
    miss_counts = np.isnan(X_miss).sum(axis=0)          # shape (P,)
    sort_order = np.argsort(miss_counts)                 # ascending
    active_vars = [int(j) for j in sort_order if miss_counts[j] > 0]

    if not active_vars:
        # No missing values; return as-is
        return X_imp

    # ── Step 3: Iterative imputation with convergence check ──────────────────
    gamma_cont_prev = np.inf
    gamma_cat_prev = np.inf
    X_best = X_imp.copy()   # best iterate (returned on divergence)

    for _iter in range(max_iter):
        X_old = X_imp.copy()

        for j in active_vars:
            obs_rows = ~np.isnan(X_miss[:, j])   # rows where j was observed
            mis_rows = np.isnan(X_miss[:, j])    # rows where j is missing

            # Predictor columns: all columns except j
            pred_cols = [k for k in range(X_imp.shape[1]) if k != j]
            X_train = X_imp[np.ix_(obs_rows, pred_cols)]
            y_train = X_imp[obs_rows, j]
            X_pred = X_imp[np.ix_(mis_rows, pred_cols)]

            if j in categorical_idx:
                preds = _rf_classify(
                    X_train, y_train, X_pred,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                )
            else:
                preds = _rf_regress(
                    X_train, y_train, X_pred,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                )

            X_imp[mis_rows, j] = preds

        # ── Convergence criterion (Stekhoven 2012, Eq. 1) ────────────────────
        gamma_cont = _gamma_continuous(X_imp, X_old, continuous_idx, miss_counts)
        gamma_cat = _gamma_categorical(X_imp, X_old, categorical_idx, miss_counts)

        cont_ok = (not continuous_idx) or (gamma_cont <= gamma_cont_prev)
        cat_ok = (not categorical_idx) or (gamma_cat <= gamma_cat_prev)

        if cont_ok and cat_ok:
            X_best = X_imp.copy()
            gamma_cont_prev = gamma_cont
            gamma_cat_prev = gamma_cat
        else:
            # Both criteria must stop decreasing before we stop
            # Return the best observed iterate
            break

    return X_best


# ── Internal helpers ───────────────────────────────────────────────────────────

def _initial_fill(
    X_miss: np.ndarray,
    continuous_idx: List[int],
    categorical_idx: List[int],
) -> np.ndarray:
    """Fill NaN with column mean (continuous) or mode (categorical).

    Initial imputation must produce a complete matrix so that the first
    Random Forest can be trained.  Mean/mode is standard (Stekhoven 2012 §2).
    """
    X = X_miss.astype(float, copy=True)
    for j in continuous_idx:
        col = X[:, j]
        fill = np.nanmean(col)
        col[np.isnan(col)] = fill
        X[:, j] = col
    for j in categorical_idx:
        col = X[:, j]
        observed = col[~np.isnan(col)]
        if len(observed) == 0:
            continue
        vals, counts = np.unique(observed, return_counts=True)
        fill = float(vals[np.argmax(counts)])
        col[np.isnan(col)] = fill
        X[:, j] = col
    return X


def _rf_regress(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
) -> np.ndarray:
    """Fit RandomForestRegressor and predict for continuous variables."""
    if X_pred.shape[0] == 0:
        return np.array([])
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf.predict(X_pred)


def _rf_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
) -> np.ndarray:
    """Fit RandomForestClassifier and predict for categorical variables.

    If only one class is present in training, return that class for all
    missing rows (degenerate case).
    """
    if X_pred.shape[0] == 0:
        return np.array([])
    classes = np.unique(y_train.astype(int))
    if len(classes) < 2:
        return np.full(X_pred.shape[0], classes[0], dtype=float)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train.astype(int))
    return rf.predict(X_pred).astype(float)


def _gamma_continuous(
    X_new: np.ndarray,
    X_old: np.ndarray,
    cont_idx: List[int],
    miss_counts: np.ndarray,
) -> float:
    """Compute γ_cont convergence criterion (Stekhoven 2012, Eq. 1a).

    γ_cont = Σ_{j ∈ F_cont} Σ_{i: R_ij=0} (X_new[i,j] - X_old[i,j])²
             ─────────────────────────────────────────────────────────
             Σ_{j ∈ F_cont} Σ_{i: R_ij=0} X_new[i,j]²
    """
    if not cont_idx:
        return 0.0
    num, den = 0.0, 0.0
    for j in cont_idx:
        n_miss = int(miss_counts[j])
        if n_miss == 0:
            continue
        # Only summed over missing cells — but we compare X_new vs X_old
        # at ALL positions (imputed cells only differ at missing positions)
        diff = X_new[:, j] - X_old[:, j]
        num += float(np.sum(diff**2))
        den += float(np.sum(X_new[:, j]**2))
    return num / den if den > 1e-12 else 0.0


def _gamma_categorical(
    X_new: np.ndarray,
    X_old: np.ndarray,
    cat_idx: List[int],
    miss_counts: np.ndarray,
) -> float:
    """Compute γ_cat convergence criterion (Stekhoven 2012, Eq. 1b).

    γ_cat = # imputed cells where X_new ≠ X_old
            ─────────────────────────────────────
            total number of missing categorical cells
    """
    if not cat_idx:
        return 0.0
    total = sum(int(miss_counts[j]) for j in cat_idx)
    if total == 0:
        return 0.0
    disagree = sum(
        int(np.sum(X_new[:, j].astype(int) != X_old[:, j].astype(int)))
        for j in cat_idx
    )
    return disagree / total
