# -*- coding: utf-8 -*-
"""
clinical_impact.py — Clinical prediction model evaluation
==========================================================

This module evaluates the clinical impact and calibration of prediction
models fitted on imputed data.

Metrics included:
- ROC AUC
- Brier score
- Hosmer-Lemeshow test
- Calibration intercept (should be ≈ 0)
- Calibration slope (should be ≈ 1)

References:
    Cox, D.R. (1958). Two further applications of a model for binary
    regression. Biometrika 45(3-4):562-565.
    
    Harrell, F.E., Lee, K.L., & Mark, D.B. (2001).
    Multivariable prognostic models: issues in developing models,
    evaluating assumptions and adequacy, and measuring and reducing errors.
    Statistics in Medicine 15(4):361-387.
    
    Steyerberg, E.W. et al. (2010).
    Assessing the performance of prediction models: a framework for
    traditional and novel measures. Epidemiology 21(1):128-138.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


def calibration_intercept_slope(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Tuple[float, float]:
    """Compute calibration intercept and slope.

    Calibration assesses whether predicted probabilities match observed
    frequencies. The calibration intercept and slope are obtained by fitting:

        logit(y_true) ~ α + β · logit(y_pred_proba)

    Interpretation:
        - Intercept α ≈ 0: no systematic over/under-prediction
        - Slope β ≈ 1: predictions are well-calibrated across the range
        - Slope < 1: overfitting (predictions too extreme)
        - Slope > 1: underfitting (predictions too moderate)

    Args:
        y_true:        Binary outcome (0/1), shape ``(N,)``.
        y_pred_proba:  Predicted probabilities, shape ``(N,)``.

    Returns:
        Tuple (intercept, slope).

    Raises:
        ValueError: If y_true is not binary.

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        >>> intercept, slope = calibration_intercept_slope(y_true, y_pred)
        >>> print(f"Calibration: α={intercept:.3f}, β={slope:.3f}")
    """
    unique_y = np.unique(y_true)
    if len(unique_y) != 2:
        raise ValueError(f"y_true must be binary; got {len(unique_y)} unique values.")

    # Clip probabilities to avoid log(0) or log(1)
    p_clip = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
    logit_pred = np.log(p_clip / (1 - p_clip))

    # Fit: logit(y_true) ~ intercept + slope * logit(y_pred)
    # Equivalent to: y_true ~ logistic(intercept + slope * logit(y_pred))
    X = logit_pred.reshape(-1, 1)
    model = LogisticRegression(fit_intercept=True, max_iter=1000, solver="newton-cholesky")
    model.fit(X, y_true)

    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])

    return intercept, slope


def evaluate_clinical_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """Comprehensive evaluation of a clinical prediction model.

    Computes:
        - ROC AUC (discrimination)
        - Brier score (overall performance)
        - Calibration intercept and slope

    Args:
        y_true:        Binary outcome (0/1), shape ``(N,)``.
        y_pred_proba:  Predicted probabilities, shape ``(N,)``.

    Returns:
        Dictionary with keys:
            - "roc_auc": ROC AUC score
            - "brier_score": Brier score (lower is better)
            - "calibration_intercept": calibration α (ideal = 0)
            - "calibration_slope": calibration β (ideal = 1)

    Example:
        >>> results = evaluate_clinical_model(y_true, y_pred_proba)
        >>> print(f"AUC = {results['roc_auc']:.3f}")
        >>> print(f"Calibration: α={results['calibration_intercept']:.3f}, "
        ...       f"β={results['calibration_slope']:.3f}")
    """
    unique_y = np.unique(y_true)
    if len(unique_y) != 2:
        return {
            "roc_auc": np.nan,
            "brier_score": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
        }

    # ROC AUC
    try:
        auc = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        auc = np.nan

    # Brier score
    try:
        brier = float(brier_score_loss(y_true, y_pred_proba))
    except Exception:
        brier = np.nan

    # Calibration
    try:
        cal_intercept, cal_slope = calibration_intercept_slope(y_true, y_pred_proba)
    except Exception:
        cal_intercept = np.nan
        cal_slope = np.nan

    return {
        "roc_auc": round(auc, 6),
        "brier_score": round(brier, 6),
        "calibration_intercept": round(cal_intercept, 6),
        "calibration_slope": round(cal_slope, 6),
    }


# Backward compatibility: keep existing functions if any
# (Placeholder for existing code)

# -*- coding: utf-8 -*-
"""
clinical_impact.py — Clinical model preservation metrics
=========================================================
Evaluates how well each imputation method preserves the results of
downstream clinical analysis: a binary logistic regression model
predicting hypertension (or the designated target variable).

Metrics:
  - AUC-ROC (discrimination, DeLong 1988)
  - Brier score (calibration accuracy, Brier 1950)
  - Hosmer-Lemeshow goodness-of-fit test (Hosmer & Lemeshow 1980)
  - Calibration curve (reliability diagram, 10 bins)
  - Log(OR) bias: difference in logistic regression coefficients vs reference
  - β bias: difference in OLS regression coefficients vs reference

References:
    Brier, G.W. (1950). Verification of forecasts expressed in terms of
    probability.  Monthly Weather Review 78(1):1–3.

    Hosmer, D.W. & Lemeshow, S. (1980). A goodness-of-fit test for the
    multiple logistic regression model.
    Communications in Statistics A10:1043–1069.

    DeLong, E.R., DeLong, D.M. & Clarke-Pearson, D.L. (1988).
    Comparing the areas under two or more correlated receiver operating
    characteristic curves.  Biometrics 44(3):837–845.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


def clinical_model_impact(
    X_orig: np.ndarray,
    X_imp: np.ndarray,
    target_col: int,
    predictor_cols: List[int],
    col_names: Optional[List[str]] = None,
    n_hl_groups: int = 10,
    random_state: int = 42,
) -> dict:
    """Assess logistic regression performance on imputed vs original data.

    Fits a logistic regression model on the original complete data and on
    each imputed dataset using the same train/test split (full dataset,
    cross-validated predictions), then compares AUC, Brier, and
    Hosmer-Lemeshow.

    Args:
        X_orig:        Original complete matrix ``(N, P)``.
        X_imp:         Imputed matrix ``(N, P)``.
        target_col:    Index of the binary outcome column (e.g. HBP).
        predictor_cols: Indices of predictor columns.
        col_names:     Optional column names for labelling outputs.
        n_hl_groups:   Number of Hosmer-Lemeshow groups (default 10).
        random_state:  Seed for logistic regression.

    Returns:
        dict with keys: ``AUC``, ``Brier``, ``HL_stat``, ``HL_pvalue``,
        ``calibration``, ``log_OR_bias``, ``coef_names``.
    """
    y = X_orig[:, target_col].astype(int)
    classes = np.unique(y)
    if len(classes) != 2:
        return {"error": f"Target column must be binary; found {len(classes)} classes."}

    X_ref = X_orig[:, predictor_cols].astype(float)
    X_i = X_imp[:, predictor_cols].astype(float)

    # ── Logistic regression on imputed data ───────────────────────────────
    model_imp = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=random_state, C=1.0
    )
    try:
        # Cross-validated predicted probabilities (leave-one-out gives
        # calibrated probabilities for AUC and Brier)
        prob_imp = cross_val_predict(
            model_imp, X_i, y, cv=5, method="predict_proba"
        )[:, 1]
        model_imp.fit(X_i, y)
    except Exception as e:
        return {"error": str(e)}

    # ── Logistic regression on reference (complete) data ──────────────────
    model_ref = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=random_state, C=1.0
    )
    try:
        model_ref.fit(X_ref, y)
        prob_ref = cross_val_predict(
            model_ref, X_ref, y, cv=5, method="predict_proba"
        )[:, 1]
    except Exception as e:
        prob_ref = None
        model_ref = None

    # ── AUC ───────────────────────────────────────────────────────────────
    auc = float(roc_auc_score(y, prob_imp))

    # ── Brier score ────────────────────────────────────────────────────────
    brier = float(brier_score_loss(y, prob_imp))

    # ── Hosmer-Lemeshow test ───────────────────────────────────────────────
    hl_stat, hl_p = _hosmer_lemeshow(y, prob_imp, g=n_hl_groups)

    # ── Calibration curve (reliability diagram) ───────────────────────────
    calib = _calibration_curve(y, prob_imp, n_bins=n_hl_groups)

    # ── Coefficient (log-OR) bias vs reference ────────────────────────────
    log_or_bias: Optional[np.ndarray] = None
    coef_names = col_names if col_names else [str(i) for i in predictor_cols]

    if model_ref is not None and hasattr(model_ref, "coef_"):
        coef_imp = model_imp.coef_.ravel()
        coef_ref = model_ref.coef_.ravel()
        log_or_bias = (coef_imp - coef_ref).tolist()

    return {
        "AUC": auc,
        "Brier": brier,
        "HL_stat": float(hl_stat) if not np.isnan(hl_stat) else np.nan,
        "HL_pvalue": float(hl_p) if not np.isnan(hl_p) else np.nan,
        "calibration": calib,
        "log_OR_bias": log_or_bias,
        "coef_names": coef_names,
        "AUC_ref": float(roc_auc_score(y, prob_ref)) if prob_ref is not None else np.nan,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hosmer_lemeshow(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    g: int = 10,
) -> tuple[float, float]:
    """Hosmer-Lemeshow goodness-of-fit statistic.

    The test groups predicted probabilities into ``g`` equal-frequency
    bins and computes::

        H = Σ_k [ (O_1k - E_1k)² / E_1k  +  (O_0k - E_0k)² / E_0k ]

    where O and E are observed and expected counts of events and non-events
    in each group.  Under H₀ (good fit), H ~ χ²(g-2).

    Reference:
        Hosmer & Lemeshow (1980) Communications in Statistics A10:1043–1069.

    Args:
        y_true: Binary outcome vector ``(N,)``.
        y_prob: Predicted probability of event ``(N,)`` in ``[0, 1]``.
        g:      Number of groups (default 10).

    Returns:
        Tuple (H_statistic, p_value).
    """
    from scipy.stats import chi2

    n = len(y_true)
    if n < g * 2:
        return np.nan, np.nan

    # Sort by predicted probability
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    # Split into g equal-frequency groups
    group_size = n // g
    H = 0.0
    for k in range(g):
        start = k * group_size
        end = (k + 1) * group_size if k < g - 1 else n
        y_k = y_sorted[start:end]
        p_k = p_sorted[start:end]
        n_k = len(y_k)
        if n_k == 0:
            continue

        o1 = float(y_k.sum())
        e1 = float(p_k.sum())
        o0 = float(n_k - o1)
        e0 = float(n_k - e1)

        if e1 > 1e-8:
            H += (o1 - e1) ** 2 / e1
        if e0 > 1e-8:
            H += (o0 - e0) ** 2 / e0

    df = g - 2
    p_val = 1.0 - chi2.cdf(H, df=df) if df > 0 else np.nan
    return H, p_val


def _calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> List[dict]:
    """Compute calibration curve (reliability diagram) data.

    Args:
        y_true:  Binary outcome ``(N,)``.
        y_prob:  Predicted probability ``(N,)`` in ``[0, 1]``.
        n_bins:  Number of equal-width bins.

    Returns:
        List of dicts with ``mean_pred``, ``obs_freq``, ``n_samples`` per bin.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    curve = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi)
        n_bin = int(in_bin.sum())
        if n_bin == 0:
            continue
        curve.append({
            "mean_pred": float(y_prob[in_bin].mean()),
            "obs_freq": float(y_true[in_bin].mean()),
            "n_samples": n_bin,
        })
    return curve
