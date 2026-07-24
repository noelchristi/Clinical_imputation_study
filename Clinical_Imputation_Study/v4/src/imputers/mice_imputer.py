# -*- coding: utf-8 -*-
"""
mice_imputer.py — MICE with Predictive Mean Matching (PMM)
===========================================================
Multivariate Imputation by Chained Equations (MICE), also known as
"fully conditional specification" (FCS).

This implementation follows the R ``mice`` package specification
(van Buuren & Groothuis-Oudshoorn 2011) using:

  * **Continuous variables**: Predictive Mean Matching (PMM, k=5 donors)
    — ensures imputed values come from the empirical distribution; avoids
    out-of-range predictions.

  * **Binary variables**:  Logistic regression with Bernoulli draw from
    predicted probability.

  * **Ordinal variables**: Multinomial logistic regression with draw from
    predicted class probabilities (proportional-odds approximation).

The algorithm performs ``max_iter`` full passes (one per variable per
pass) over all variables with missing data.

IMPORTANT distinction from ``IterativeImputer(BayesianRidge)``:
  - Bayesian Ridge is a regularised linear model that can produce
    out-of-range values and does not match the empirical distribution.
  - PMM always returns an *observed* value, preserving the marginal
    distribution and respecting variable bounds.

References:
    van Buuren, S. & Groothuis-Oudshoorn, K. (2011).
    mice: Multivariate imputation by chained equations in R.
    Journal of Statistical Software 45(3):1–67.

    van Buuren, S. (2018).
    Flexible Imputation of Missing Data (2nd ed.).
    CRC Press.  https://stefvanbuuren.name/fimd/
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import check_random_state


# Number of donors for PMM
PMM_K_DONORS: int = 5


def impute_mice(
    X_miss: np.ndarray,
    continuous_idx: List[int],
    binary_idx: List[int],
    ordinal_idx: List[int],
    max_iter: int = 10,
    k_donors: int = PMM_K_DONORS,
    random_state: int = 42,
) -> np.ndarray:
    """MICE imputation with PMM for continuous, logistic for binary/ordinal.

    Args:
        X_miss:         Incomplete data matrix ``(N, P)`` with NaN.
        continuous_idx: Indices of continuous variables (PMM applied).
        binary_idx:     Indices of binary variables (logistic regression).
        ordinal_idx:    Indices of ordinal variables (multinomial logistic).
        max_iter:       Number of full imputation cycles (default 10).
        k_donors:       Number of PMM donors (default 5, van Buuren 2018 §3.4).
        random_state:   Integer seed for reproducibility.

    Returns:
        X_imp: Complete matrix ``(N, P)`` with no NaN.

    Raises:
        ValueError: If any column index overlaps between the type lists.
    """
    all_idx = continuous_idx + binary_idx + ordinal_idx
    if len(all_idx) != len(set(all_idx)):
        raise ValueError("Column indices must not overlap between variable-type lists.")

    rng = check_random_state(random_state)
    X_imp = _initial_fill(X_miss, continuous_idx, binary_idx + ordinal_idx)
    miss_matrix = np.isnan(X_miss)
    n_cols = X_miss.shape[1]
    pred_cols_map = [np.r_[0:j, j + 1:n_cols] for j in range(n_cols)]

    # Variables with at least one missing value — only these are updated
    miss_vars = np.flatnonzero(miss_matrix.any(axis=0)).tolist()

    for _iteration in range(max_iter):
        for j in miss_vars:
            obs_mask = ~miss_matrix[:, j]   # rows observed for j
            mis_mask = miss_matrix[:, j]    # rows missing for j

            if obs_mask.sum() < 2:
                # Not enough observed values to fit any model
                continue

            # Predictor matrix: all columns except j (already filled)
            X_pred_cols = pred_cols_map[j]
            X_obs = X_imp[np.ix_(obs_mask, X_pred_cols)]
            X_mis = X_imp[np.ix_(mis_mask, X_pred_cols)]
            y_obs = X_imp[obs_mask, j]

            if j in continuous_idx:
                preds = _pmm(X_obs, y_obs, X_mis, k=k_donors, rng=rng)
            elif j in binary_idx:
                preds = _logreg_binary(X_obs, y_obs, X_mis, rng=rng)
            else:
                # Ordinal: multinomial logistic
                preds = _logreg_multinomial(X_obs, y_obs, X_mis, rng=rng)

            X_imp[mis_mask, j] = preds

    return X_imp


# ── Internal imputation functions ─────────────────────────────────────────────

def _initial_fill(
    X_miss: np.ndarray,
    continuous_idx: List[int],
    categorical_idx: List[int],
) -> np.ndarray:
    """Fill missing values with column median (continuous) or mode (categorical).

    This initialisation is standard in MICE implementations (van Buuren 2018 §4.5).

    Args:
        X_miss:          Incomplete matrix.
        continuous_idx:  Continuous column indices.
        categorical_idx: Categorical column indices.

    Returns:
        Initialised complete matrix.
    """
    X = X_miss.astype(float, copy=True)
    for j in continuous_idx:
        col = X[:, j]
        fill = np.nanmedian(col)
        col[np.isnan(col)] = fill
        X[:, j] = col
    for j in categorical_idx:
        col = X[:, j]
        observed = col[~np.isnan(col)]
        if len(observed) == 0:
            continue
        vals, counts = np.unique(observed, return_counts=True)
        fill = vals[np.argmax(counts)]
        col[np.isnan(col)] = fill
        X[:, j] = col
    # Fill any remaining columns (e.g., nominal) not covered above — use mode/median
    all_filled = set(continuous_idx) | set(categorical_idx)
    for j in range(X.shape[1]):
        if j in all_filled:
            continue
        col = X[:, j]
        if not np.isnan(col).any():
            continue
        observed = col[~np.isnan(col)]
        if len(observed) == 0:
            continue
        if np.issubdtype(observed.dtype, np.integer) or np.all(observed == observed.astype(int)):
            vals, counts = np.unique(observed.astype(int), return_counts=True)
            fill = float(vals[np.argmax(counts)])
        else:
            fill = float(np.nanmedian(observed))
        col[np.isnan(col)] = fill
        X[:, j] = col
    return X


def _pmm(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_mis: np.ndarray,
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Predictive Mean Matching imputation for continuous variables.

    Algorithm (van Buuren 2018, §3.4):
      1. Fit OLS on observed data: β̂ = (X_obs'X_obs)^{-1} X_obs'y_obs
      2. Compute ŷ_obs = X_obs β̂ and ŷ_mis = X_mis β̂
      3. For each missing i, find the k observed j with smallest |ŷ_obs_j - ŷ_mis_i|
      4. Randomly sample one donor j* from those k candidates
      5. Imputed value = y_obs[j*]  (the actual observed value, not the prediction)

    This guarantees imputed values are within the observed data range.

    Args:
        X_obs: Predictor matrix for observed rows ``(n_obs, P-1)``.
        y_obs: Observed outcome values ``(n_obs,)``.
        X_mis: Predictor matrix for missing rows ``(n_mis, P-1)``.
        k:     Number of donors.
        rng:   Random state for donor sampling.

    Returns:
        Array of imputed values ``(n_mis,)``.
    """
    if X_obs.shape[0] == 0 or X_mis.shape[0] == 0:
        return np.full(X_mis.shape[0], np.nanmean(y_obs))

    model = LinearRegression(fit_intercept=True)
    model.fit(X_obs, y_obs)

    y_hat_obs = model.predict(X_obs)   # predicted for observed
    y_hat_mis = model.predict(X_mis)   # predicted for missing

    n_mis = X_mis.shape[0]
    k_actual = min(k, len(y_obs))
    if k_actual <= 0:
        return np.full(n_mis, np.nanmean(y_obs))

    # Vectorized PMM donor search for all missing rows at once.
    dist = np.abs(y_hat_mis[:, None] - y_hat_obs[None, :])
    donor_pool = np.argpartition(dist, k_actual - 1, axis=1)[:, :k_actual]
    donor_pick = rng.randint(0, k_actual, size=n_mis)
    chosen_idx = donor_pool[np.arange(n_mis), donor_pick]
    return y_obs[chosen_idx].astype(float)


def _logreg_binary(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_mis: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Logistic regression imputation for binary variables.

    Fits logistic regression on observed rows, then draws imputed values
    from the predicted Bernoulli distribution (stochastic imputation,
    as required by proper MICE).

    Args:
        X_obs: Predictor matrix ``(n_obs, P-1)``.
        y_obs: Binary outcome ``(n_obs,)`` with integer labels {lo, hi}.
        X_mis: Predictor matrix ``(n_mis, P-1)``.
        rng:   Random state.

    Returns:
        Array of imputed binary values ``(n_mis,)``.
    """
    classes = np.unique(y_obs.astype(int))
    if len(classes) < 2:
        return np.full(X_mis.shape[0], classes[0])

    model = LogisticRegression(max_iter=3000, solver="newton-cholesky", random_state=0)
    try:
        model.fit(X_obs, y_obs.astype(int))
        prob_pos = model.predict_proba(X_mis)[:, 1]
        draws = (rng.random(X_mis.shape[0]) < prob_pos).astype(float)
        # Map back to original class labels
        lo, hi = float(classes[0]), float(classes[-1])
        return np.where(draws == 1, hi, lo)
    except Exception:
        # Fallback: mode imputation
        vals, counts = np.unique(y_obs, return_counts=True)
        return np.full(X_mis.shape[0], float(vals[np.argmax(counts)]))


def _logreg_multinomial(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_mis: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Multinomial logistic regression for ordinal variables.

    Note: True proportional-odds regression (polr) would be preferable
    for ordered categories.  Multinomial logistic is a valid approximation
    used when category counts are small (van Buuren 2018 §7.3).

    Args:
        X_obs: Predictor matrix ``(n_obs, P-1)``.
        y_obs: Integer-encoded ordinal outcome ``(n_obs,)``.
        X_mis: Predictor matrix ``(n_mis, P-1)``.
        rng:   Random state.

    Returns:
        Array of imputed ordinal values ``(n_mis,)``.
    """
    classes = np.unique(y_obs.astype(int))
    if len(classes) < 2:
        return np.full(X_mis.shape[0], classes[0])

    model = LogisticRegression(max_iter=3000, solver="newton-cholesky", random_state=0)
    try:
        model.fit(X_obs, y_obs.astype(int))
        probs = model.predict_proba(X_mis)   # (n_mis, n_classes)
        # Vectorized class draw from per-row class probabilities.
        cdf = np.cumsum(probs, axis=1)
        u = rng.random((X_mis.shape[0], 1))
        class_idx = np.sum(cdf < u, axis=1)
        class_idx = np.clip(class_idx, 0, len(model.classes_) - 1)
        return model.classes_[class_idx].astype(float)
    except Exception:
        vals, counts = np.unique(y_obs, return_counts=True)
        return np.full(X_mis.shape[0], float(vals[np.argmax(counts)]))
