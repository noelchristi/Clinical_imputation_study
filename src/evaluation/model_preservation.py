# -*- coding: utf-8 -*-
"""
evaluation/model_preservation.py — Clinical Model Preservation After Imputation
================================================================================

Evaluates the fidelity of statistical models fitted on imputed data relative to
models fitted on the original complete data (ground truth).

Supported models:
    • Logistic regression  → binary outcome (e.g., hypertension status)
    • Linear regression     → continuous outcome (e.g., BMI)

Metrics computed per model type:
    • Coefficient comparison:  absolute bias, relative bias, RMSE on β vector
    • Odds Ratio (logistic):   bias in OR, 95% CI coverage
    • Standard error ratio:    SE_imputed / SE_original → quantifies uncertainty inflation/deflation
    • Significance change rate: proportion of predictors whose p < 0.05 status flips after imputation
    • Calibration:             AUC (ROC), Brier score, Hosmer-Lemeshow goodness-of-fit test
    • Goodness-of-fit:         R² (linear), McFadden pseudo-R² (logistic)

References
----------
[1] Rubin DB. Multiple Imputation for Nonresponse in Surveys. Wiley, 1987.
    → Standard error combination rules, variance decomposition.
[2] Hosmer DW, Lemeshow S, Sturdivant RX. Applied Logistic Regression. 3rd ed. Wiley, 2013.
    → Hosmer-Lemeshow test (Section 5.2), pseudo-R² (Section 4.6).
[3] Brier GW. Verification of forecasts expressed in terms of probability. Mon Wea Rev. 1950;78:1-3.
    → Brier score definition.
[4] Steyerberg EW et al. Assessing the performance of prediction models. Epidemiology. 2010;21(1):128-138.
    → AUC, calibration framework for clinical prediction models.
[5] Van Calster B et al. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17:230.
    → Modern calibration assessment recommendations.

Assumptions
-----------
• The imputed dataset has no remaining missing values.
• The original dataset is complete (serves as the reference/gold standard).
• Predictor indices in original and imputed datasets are aligned (same column order).
• Binary outcome is encoded as 0/1 integers.

Limitations
------------
• Hosmer-Lemeshow test is sensitive to the number of groups (g = 10) and sample size;
  with N < 100, the test may be unreliable.
• Brier score and AUC are computed on the training set (no train/test split),
  representing apparent performance — optimistic relative to external validation.
• Single imputation underestimates standard errors (SE ratio < 1 is expected);
  Rubin's rules require M ≥ 2 imputed datasets for proper variance combination.
"""

import numpy as np
from scipy import stats
from scipy.stats import chi2 as chi2_dist
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    r2_score,
    accuracy_score,
    mean_squared_error,
)


# ══════════════════════════════════════════════════════════════════════
# 1. COEFFICIENT-LEVEL METRICS
# ══════════════════════════════════════════════════════════════════════

def coefficient_bias(
    beta_ref: np.ndarray,
    beta_imp: np.ndarray,
) -> dict:
    """Compute bias metrics between reference and imputed coefficient vectors.

    Parameters
    ----------
    beta_ref : np.ndarray, shape (p,)
        Coefficients from model fitted on original complete data.
    beta_imp : np.ndarray, shape (p,)
        Coefficients from model fitted on imputed data.

    Returns
    -------
    dict with keys:
        abs_bias_max  : maximum absolute coefficient difference
        abs_bias_mean : mean absolute coefficient difference
        rel_bias_mean : mean relative bias (|β_imp − β_ref| / |β_ref|)
        rmse_beta     : root mean squared error of the β vector
        correlation   : Pearson correlation between β_ref and β_imp
    """
    beta_ref = np.asarray(beta_ref, dtype=np.float64).ravel()
    beta_imp = np.asarray(beta_imp, dtype=np.float64).ravel()

    if len(beta_ref) != len(beta_imp):
        raise ValueError(f"Dimension mismatch: ref={len(beta_ref)}, imp={len(beta_imp)}")

    diff = beta_imp - beta_ref
    abs_diff = np.abs(diff)

    # Relative bias: guard against division by zero for β_ref ≈ 0
    denom = np.abs(beta_ref)
    safe_denom = np.where(denom < 1e-8, 1.0, denom)
    rel_bias = abs_diff / safe_denom

    # RMSE on β vector
    rmse_beta = np.sqrt(np.mean(diff ** 2))

    # Pearson correlation (consistency of direction/magnitude)
    if np.std(beta_ref) > 1e-12 and np.std(beta_imp) > 1e-12:
        corr = np.corrcoef(beta_ref, beta_imp)[0, 1]
    else:
        corr = np.nan

    return {
        "abs_bias_max": float(np.max(abs_diff)),
        "abs_bias_mean": float(np.mean(abs_diff)),
        "rel_bias_mean": float(np.mean(rel_bias)),
        "rmse_beta": float(rmse_beta),
        "correlation": float(corr) if not np.isnan(corr) else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════
# 2. ODDS RATIO METRICS (Logistic Regression)
# ══════════════════════════════════════════════════════════════════════

def odds_ratio_bias(
    beta_ref: np.ndarray,
    beta_imp: np.ndarray,
) -> dict:
    """Compare Odds Ratios (OR = exp(β)) between reference and imputed models.

    Parameters
    ----------
    beta_ref, beta_imp : np.ndarray
        Logistic regression coefficient vectors (excluding intercept).

    Returns
    -------
    dict with:
        or_ref, or_imp       : vectors of ORs
        or_rel_bias_mean     : mean relative bias in OR
        or_sign_agreement    : proportion of coefficients with same sign direction
        or_rank_corr         : Spearman rank correlation between OR vectors
    """
    beta_ref = np.asarray(beta_ref, dtype=np.float64).ravel()
    beta_imp = np.asarray(beta_imp, dtype=np.float64).ravel()

    or_ref = np.exp(beta_ref)
    or_imp = np.exp(beta_imp)

    # Relative bias in OR
    abs_denom = np.abs(or_ref)
    safe_denom = np.where(abs_denom < 1e-8, 1.0, abs_denom)
    or_rel_bias = np.abs(or_imp - or_ref) / safe_denom

    # Sign agreement (direction of association preserved?)
    sign_agree = np.mean(np.sign(beta_ref) == np.sign(beta_imp))

    # Spearman rank correlation (rank ordering of predictor importance)
    if np.std(or_ref) > 1e-12 and np.std(or_imp) > 1e-12:
        rank_corr, _ = stats.spearmanr(or_ref, or_imp)
    else:
        rank_corr = np.nan

    return {
        "or_ref": or_ref.tolist(),
        "or_imp": or_imp.tolist(),
        "or_rel_bias_mean": float(np.mean(or_rel_bias)),
        "or_sign_agreement": float(sign_agree),
        "or_rank_corr": float(rank_corr) if not np.isnan(rank_corr) else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════
# 3. STANDARD ERROR & INFERENCE METRICS
# ══════════════════════════════════════════════════════════════════════

def standard_error_metrics(
    se_ref: np.ndarray,
    se_imp: np.ndarray,
    beta_ref: np.ndarray,
    beta_imp: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Evaluate standard error inflation/deflation and significance changes.

    Rubin (1987, Ch. 3): single imputation systematically underestimates SE.
    This function quantifies the magnitude of this bias.

    Parameters
    ----------
    se_ref, se_imp : np.ndarray
        Standard errors of coefficients (excluding intercept).
    beta_ref, beta_imp : np.ndarray
        Coefficient vectors.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    dict with:
        se_ratio_mean          : mean(SE_imp / SE_ref) — <1 = underestimation
        se_ratio_median        : median ratio
        wald_p_ref, wald_p_imp : Wald test p-values
        significance_change    : proportion of predictors flipping p < α status
        n_significant_ref      : count of significant predictors in reference model
        n_significant_imp      : count of significant predictors in imputed model
    """
    se_ref = np.asarray(se_ref, dtype=np.float64).ravel()
    se_imp = np.asarray(se_imp, dtype=np.float64).ravel()
    beta_ref = np.asarray(beta_ref, dtype=np.float64).ravel()
    beta_imp = np.asarray(beta_imp, dtype=np.float64).ravel()

    # SE ratio (guard against SE_ref ≈ 0)
    safe_se_ref = np.where(se_ref < 1e-12, 1e-12, se_ref)
    se_ratio = se_imp / safe_se_ref

    # Wald test p-values (two-sided)
    # p = 2 * Φ(−|β| / SE)  where Φ is the standard normal CDF
    z_ref = np.abs(beta_ref) / safe_se_ref
    z_imp = np.abs(beta_imp) / np.where(se_imp < 1e-12, 1e-12, se_imp)
    p_ref = 2.0 * stats.norm.sf(z_ref)
    p_imp = 2.0 * stats.norm.sf(z_imp)

    # Significance status
    sig_ref = (p_ref < alpha).astype(int)
    sig_imp = (p_imp < alpha).astype(int)

    # Significance change: proportion where status flipped
    significance_change = np.mean(sig_ref != sig_imp)

    return {
        "se_ratio_mean": float(np.mean(se_ratio)),
        "se_ratio_median": float(np.median(se_ratio)),
        "wald_p_ref": p_ref.tolist(),
        "wald_p_imp": p_imp.tolist(),
        "significance_change": float(significance_change),
        "n_significant_ref": int(np.sum(sig_ref)),
        "n_significant_imp": int(np.sum(sig_imp)),
    }


# ══════════════════════════════════════════════════════════════════════
# 4. CONFIDENCE INTERVAL METRICS
# ══════════════════════════════════════════════════════════════════════

def confidence_interval_metrics(
    beta_ref: np.ndarray,
    se_ref: np.ndarray,
    beta_imp: np.ndarray,
    se_imp: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Compute 95% CI properties and assess coverage/width changes.

    Parameters
    ----------
    beta_ref, se_ref, beta_imp, se_imp : np.ndarray
        Coefficient and standard error vectors.
    alpha : float
        Significance level (1 − α confidence level).

    Returns
    -------
    dict with:
        ci_width_ref, ci_width_imp : CI widths (upper − lower)
        ci_width_ratio_mean        : mean(CI_imp / CI_ref)
        ci_overlap_mean            : mean overlap proportion between ref and imp CIs
        ci_coverage                : proportion of β_ref falling within CI_imp
    """
    beta_ref = np.asarray(beta_ref, dtype=np.float64).ravel()
    se_ref = np.asarray(se_ref, dtype=np.float64).ravel()
    beta_imp = np.asarray(beta_imp, dtype=np.float64).ravel()
    se_imp = np.asarray(se_imp, dtype=np.float64).ravel()

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)  # ≈ 1.96 for α = 0.05

    # CI boundaries
    lower_ref = beta_ref - z_alpha * se_ref
    upper_ref = beta_ref + z_alpha * se_ref
    lower_imp = beta_imp - z_alpha * se_imp
    upper_imp = beta_imp + z_alpha * se_imp

    # Widths
    width_ref = upper_ref - lower_ref
    width_imp = upper_imp - lower_imp
    width_ratio = width_imp / np.where(width_ref < 1e-12, 1e-12, width_ref)

    # Overlap: proportion of ref CI contained within imp CI
    # Overlap length = min(upper_ref, upper_imp) − max(lower_ref, lower_imp)
    overlap_lower = np.maximum(lower_ref, lower_imp)
    overlap_upper = np.minimum(upper_ref, upper_imp)
    overlap_length = np.maximum(0.0, overlap_upper - overlap_lower)
    overlap_prop = overlap_length / np.where(width_ref < 1e-12, 1e-12, width_ref)

    # Coverage: β_ref ∈ CI_imp ?
    covered = (beta_ref >= lower_imp) & (beta_ref <= upper_imp)

    return {
        "ci_width_ref_mean": float(np.mean(width_ref)),
        "ci_width_imp_mean": float(np.mean(width_imp)),
        "ci_width_ratio_mean": float(np.mean(width_ratio)),
        "ci_overlap_mean": float(np.mean(overlap_prop)),
        "ci_coverage": float(np.mean(covered)),
    }


# ══════════════════════════════════════════════════════════════════════
# 5. HOSMER-LEMESHOW GOODNESS-OF-FIT TEST
# ══════════════════════════════════════════════════════════════════════

def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    g: int = 10,
) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test for binary logistic regression.

    Reference: Hosmer DW, Lemeshow S, Sturdivant RX. Applied Logistic Regression.
               3rd ed. Wiley, 2013. Section 5.2.2.

    Assumptions:
        • Binary outcome y_true ∈ {0, 1}.
        • Predicted probabilities y_prob ∈ [0, 1].
        • Observations are independent.
        • g groups (default 10) is standard; alternative g = floor(N/10) for small N.

    Limitations:
        • Power depends on sample size and number of groups.
        • With large N, even trivial miscalibration yields significant p.
        • With small N (N < 100), test lacks power.
        • Does not indicate the direction or magnitude of miscalibration.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Observed binary outcomes.
    y_prob : np.ndarray, shape (N,)
        Predicted probabilities from logistic model.
    g : int
        Number of groups (default 10).

    Returns
    -------
    dict with:
        chi2_stat   : Hosmer-Lemeshow χ² statistic
        p_value     : p-value from χ²(g − 2) distribution
        dof         : degrees of freedom (g − 2)
        n_total     : total observations used
        table       : list of (n_g, obs_g, exp_g) per group
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    # Remove NaN
    valid = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[valid]
    y_prob = y_prob[valid]
    N = len(y_true)

    if N < g:
        g = max(2, N // 2)

    # Sort by predicted probability and split into g equal-sized groups
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    # Decile boundaries
    group_edges = np.linspace(0, N, g + 1, dtype=int)

    hl_stat = 0.0
    table = []

    for k in range(g):
        start, end = group_edges[k], group_edges[k + 1]
        if start == end:
            continue

        obs_k = y_true_sorted[start:end].sum()
        exp_k = y_prob_sorted[start:end].sum()
        n_k = end - start

        if exp_k > 0.0 and (n_k - exp_k) > 0.0:
            hl_stat += (obs_k - exp_k) ** 2 / exp_k
            hl_stat += ((n_k - obs_k) - (n_k - exp_k)) ** 2 / (n_k - exp_k)

        table.append({
            "group": k + 1,
            "n": int(n_k),
            "observed_events": int(obs_k),
            "expected_events": float(exp_k),
        })

    # Degrees of freedom: g − 2 (Hosmer & Lemeshow, 2013, p. 159)
    dof = max(1, g - 2) if N >= g else 1
    p_value = 1.0 - chi2_dist.cdf(hl_stat, dof)

    return {
        "chi2_stat": float(hl_stat),
        "p_value": float(p_value),
        "dof": int(dof),
        "n_total": int(N),
        "table": table,
    }


# ══════════════════════════════════════════════════════════════════════
# 6. CALIBRATION CURVE POINTS
# ══════════════════════════════════════════════════════════════════════

def calibration_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration curve data points (observed proportion per predicted-probability bin).

    Reference: Van Calster B et al. Calibration: the Achilles heel of predictive
               analytics. BMC Med. 2019;17:230.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Binary outcomes.
    y_prob : np.ndarray, shape (N,)
        Predicted probabilities.
    n_bins : int
        Number of bins for calibration assessment.

    Returns
    -------
    dict with:
        bin_centers     : mean predicted probability per bin
        bin_observed    : observed event proportion per bin
        bin_counts      : number of observations per bin
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    valid = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_observed = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        n_bin = int(np.sum(mask))
        if n_bin > 0:
            bin_centers.append(float(np.mean(y_prob[mask])))
            bin_observed.append(float(np.mean(y_true[mask])))
        else:
            bin_centers.append(float((lo + hi) / 2))
            bin_observed.append(np.nan)
        bin_counts.append(n_bin)

    return {
        "bin_centers": bin_centers,
        "bin_observed": bin_observed,
        "bin_counts": bin_counts,
    }


# ══════════════════════════════════════════════════════════════════════
# 7. LOGISTIC REGRESSION EVALUATION (FULL PIPELINE)
# ══════════════════════════════════════════════════════════════════════

def evaluate_logistic_model(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_imp: np.ndarray,
    predictor_names: list = None,
    alpha: float = 0.05,
) -> dict:
    """Complete logistic regression preservation evaluation.

    Fits two logistic models (reference on original data, target on imputed data),
    then computes all coefficient-level, inference-level, and calibration metrics.

    The design matrix includes an intercept term automatically via
    scikit-learn's LogisticRegression (fit_intercept=True).

    Parameters
    ----------
    X_ref : np.ndarray, shape (N, p)
        Predictor matrix — original complete data.
    y_ref : np.ndarray, shape (N,)
        Binary outcome — original data.
    X_imp : np.ndarray, shape (N, p)
        Predictor matrix — imputed data (same subjects, same column order).
    predictor_names : list of str, optional
        Variable names for interpretable output.
    alpha : float
        Significance threshold.

    Returns
    -------
    dict with nested keys: coefficient_bias, odds_ratio, standard_error,
    confidence_interval, calibration, performance_summary.
    """
    X_ref = np.asarray(X_ref, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64).ravel()
    X_imp = np.asarray(X_imp, dtype=np.float64)

    if X_ref.shape != X_imp.shape:
        raise ValueError(
            f"Shape mismatch: X_ref {X_ref.shape} vs X_imp {X_imp.shape}"
        )
    if len(y_ref) != X_ref.shape[0]:
        raise ValueError(
            f"N mismatch: y_ref has {len(y_ref)} rows, X_ref has {X_ref.shape[0]}"
        )

    # Standardize predictors for numerical stability
    scaler_ref = StandardScaler()
    scaler_imp = StandardScaler()
    X_ref_scaled = scaler_ref.fit_transform(X_ref)
    X_imp_scaled = scaler_imp.fit_transform(X_imp)

    # ---- Fit reference model ----
    lr_ref = LogisticRegression(
        penalty=None,              # no regularization → unbiased β estimates
        max_iter=2000,
        random_state=42,
    )
    lr_ref.fit(X_ref_scaled, y_ref)
    beta_ref = lr_ref.coef_.ravel()
    y_prob_ref = lr_ref.predict_proba(X_ref_scaled)[:, 1]
    y_pred_ref = lr_ref.predict(X_ref_scaled)

    # ---- Fit imputed model ----
    lr_imp = LogisticRegression(
        penalty=None,
        max_iter=2000,
        random_state=42,
    )
    lr_imp.fit(X_imp_scaled, y_ref)   # same y_ref: we evaluate on ground-truth outcome
    beta_imp = lr_imp.coef_.ravel()
    y_prob_imp = lr_imp.predict_proba(X_imp_scaled)[:, 1]
    y_pred_imp = lr_imp.predict(X_imp_scaled)

    # ---- Standard errors via observed Fisher information ----
    # SE = sqrt(diag(H⁻¹)), where H = Xᵀ W X is the Hessian at convergence
    # W = diag(p * (1 − p)) with p = predicted probabilities
    def _compute_se(X_scaled: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """Compute coefficient standard errors via observed Fisher information."""
        N, p = X_scaled.shape
        X_design = np.column_stack([np.ones(N), X_scaled])  # add intercept column
        p = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
        W = np.diag(p * (1.0 - p))
        try:
            H = X_design.T @ W @ X_design
            H_inv = np.linalg.inv(H)
            se_full = np.sqrt(np.diag(H_inv))
            return se_full[1:]  # exclude intercept SE
        except np.linalg.LinAlgError:
            return np.full(p, np.nan)

    se_ref = _compute_se(X_ref_scaled, y_prob_ref)
    se_imp = _compute_se(X_imp_scaled, y_prob_imp)

    # ---- Assemble results ----
    results = {}

    # Coefficient-level
    results["coefficient_bias"] = coefficient_bias(beta_ref, beta_imp)
    results["odds_ratio"] = odds_ratio_bias(beta_ref, beta_imp)

    # Inference
    results["standard_error"] = standard_error_metrics(
        se_ref, se_imp, beta_ref, beta_imp, alpha
    )
    results["confidence_interval"] = confidence_interval_metrics(
        beta_ref, se_ref, beta_imp, se_imp, alpha
    )

    # Calibration
    results["calibration"] = {
        "auc_ref": float(roc_auc_score(y_ref, y_prob_ref)),
        "auc_imp": float(roc_auc_score(y_ref, y_prob_imp)),
        "brier_ref": float(brier_score_loss(y_ref, y_prob_ref)),
        "brier_imp": float(brier_score_loss(y_ref, y_prob_imp)),
        "hosmer_lemeshow_ref": hosmer_lemeshow_test(y_ref, y_prob_ref),
        "hosmer_lemeshow_imp": hosmer_lemeshow_test(y_ref, y_prob_imp),
        "calibration_curve_ref": calibration_points(y_ref, y_prob_ref),
        "calibration_curve_imp": calibration_points(y_ref, y_prob_imp),
    }

    # Performance summary
    results["performance_summary"] = {
        "accuracy_ref": float(accuracy_score(y_ref, y_pred_ref)),
        "accuracy_imp": float(accuracy_score(y_ref, y_pred_imp)),
        "mcfadden_r2_ref": float(_mcfadden_r2(y_ref, y_prob_ref)),
        "mcfadden_r2_imp": float(_mcfadden_r2(y_ref, y_prob_imp)),
    }

    # Predictor names for interpretability
    if predictor_names is not None:
        results["predictor_names"] = list(predictor_names)
        results["beta_table"] = [
            {
                "predictor": predictor_names[i],
                "beta_ref": float(beta_ref[i]),
                "beta_imp": float(beta_imp[i]),
                "se_ref": float(se_ref[i]) if i < len(se_ref) else np.nan,
                "se_imp": float(se_imp[i]) if i < len(se_imp) else np.nan,
                "or_ref": float(np.exp(beta_ref[i])),
                "or_imp": float(np.exp(beta_imp[i])),
            }
            for i in range(len(beta_ref))
        ]

    return results


def _mcfadden_r2(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """McFadden's pseudo-R² for logistic regression.

    R²_McFadden = 1 − ln(L_model) / ln(L_null)

    Reference: McFadden D. Conditional logit analysis of qualitative choice behavior.
               In: Zarembka P, ed. Frontiers in Econometrics. Academic Press, 1974:105-142.

    Range: typically [0, 1]; 0.2–0.4 considered excellent fit.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    p = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    # Log-likelihood of fitted model
    ll_model = np.sum(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

    # Log-likelihood of null (intercept-only) model
    y_mean = np.mean(y_true)
    p_null = np.clip(y_mean, 1e-15, 1.0 - 1e-15)
    ll_null = np.sum(y_true * np.log(p_null) + (1.0 - y_true) * np.log(1.0 - p_null))

    if ll_null == 0.0:
        return 0.0
    return float(1.0 - ll_model / ll_null)


# ══════════════════════════════════════════════════════════════════════
# 8. LINEAR REGRESSION EVALUATION (FULL PIPELINE)
# ══════════════════════════════════════════════════════════════════════

def evaluate_linear_model(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_imp: np.ndarray,
    predictor_names: list = None,
    alpha: float = 0.05,
) -> dict:
    """Complete linear regression preservation evaluation.

    Fits two OLS models (reference on original, target on imputed), then
    computes coefficient-level metrics, R², and residual diagnostics.

    Standard errors are computed via the standard OLS covariance estimator:
    Var(β̂) = σ̂² (XᵀX)⁻¹, where σ̂² = RSS / (N − p − 1).

    Parameters
    ----------
    X_ref, y_ref, X_imp : np.ndarray
        Reference data and imputed data.
    predictor_names : list of str, optional
    alpha : float
        Significance level.

    Returns
    -------
    dict with coefficient_bias, standard_error, confidence_interval,
    goodness_of_fit, residual_diagnostics.
    """
    X_ref = np.asarray(X_ref, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64).ravel()
    X_imp = np.asarray(X_imp, dtype=np.float64)

    if X_ref.shape != X_imp.shape:
        raise ValueError(f"Shape mismatch: {X_ref.shape} vs {X_imp.shape}")

    scaler_ref = StandardScaler()
    scaler_imp = StandardScaler()
    X_ref_scaled = scaler_ref.fit_transform(X_ref)
    X_imp_scaled = scaler_imp.fit_transform(X_imp)

    # ---- Fit models ----
    lin_ref = LinearRegression()
    lin_ref.fit(X_ref_scaled, y_ref)
    beta_ref = lin_ref.coef_.ravel()
    y_pred_ref = lin_ref.predict(X_ref_scaled)

    lin_imp = LinearRegression()
    lin_imp.fit(X_imp_scaled, y_ref)
    beta_imp = lin_imp.coef_.ravel()
    y_pred_imp = lin_imp.predict(X_imp_scaled)

    # ---- Standard errors (OLS formula) ----
    def _ols_se(X_scaled: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """Compute OLS standard errors: SE = sqrt(σ̂² × diag((XᵀX)⁻¹))."""
        N, p = X_scaled.shape
        X_design = np.column_stack([np.ones(N), X_scaled])
        rss = np.sum(residuals ** 2)
        sigma2_hat = rss / (N - p - 1) if N > p + 1 else rss / max(1, N - 1)
        try:
            XtX_inv = np.linalg.inv(X_design.T @ X_design)
            se_full = np.sqrt(sigma2_hat * np.diag(XtX_inv))
            return se_full[1:]  # exclude intercept
        except np.linalg.LinAlgError:
            return np.full(p, np.nan)

    residuals_ref = y_ref - y_pred_ref
    residuals_imp = y_ref - y_pred_imp
    se_ref = _ols_se(X_ref_scaled, residuals_ref)
    se_imp = _ols_se(X_imp_scaled, residuals_imp)

    # ---- Assemble ----
    results = {}

    results["coefficient_bias"] = coefficient_bias(beta_ref, beta_imp)
    results["standard_error"] = standard_error_metrics(
        se_ref, se_imp, beta_ref, beta_imp, alpha
    )
    results["confidence_interval"] = confidence_interval_metrics(
        beta_ref, se_ref, beta_imp, se_imp, alpha
    )

    # Goodness-of-fit
    results["goodness_of_fit"] = {
        "r2_ref": float(r2_score(y_ref, y_pred_ref)),
        "r2_imp": float(r2_score(y_ref, y_pred_imp)),
        "rmse_ref": float(np.sqrt(mean_squared_error(y_ref, y_pred_ref))),
        "rmse_imp": float(np.sqrt(mean_squared_error(y_ref, y_pred_imp))),
    }

    # Residual diagnostics
    results["residual_diagnostics"] = {
        "mean_residual_ref": float(np.mean(residuals_ref)),
        "mean_residual_imp": float(np.mean(residuals_imp)),
        "std_residual_ref": float(np.std(residuals_ref, ddof=1)),
        "std_residual_imp": float(np.std(residuals_imp, ddof=1)),
        "skew_residual_ref": float(stats.skew(residuals_ref)),
        "skew_residual_imp": float(stats.skew(residuals_imp)),
        "ks_pvalue_ref": float(_normality_test(residuals_ref)),
        "ks_pvalue_imp": float(_normality_test(residuals_imp)),
    }

    if predictor_names is not None:
        results["predictor_names"] = list(predictor_names)
        results["beta_table"] = [
            {
                "predictor": predictor_names[i],
                "beta_ref": float(beta_ref[i]),
                "beta_imp": float(beta_imp[i]),
                "se_ref": float(se_ref[i]) if i < len(se_ref) else np.nan,
                "se_imp": float(se_imp[i]) if i < len(se_imp) else np.nan,
            }
            for i in range(len(beta_ref))
        ]

    return results


def _normality_test(residuals: np.ndarray) -> float:
    """Shapiro-Wilk test for residual normality. Returns p-value."""
    residuals = np.asarray(residuals, dtype=np.float64).ravel()
    valid = ~np.isnan(residuals)
    if valid.sum() < 3 or np.std(residuals[valid]) < 1e-12:
        return np.nan
    _, p = stats.shapiro(residuals[valid][:5000])
    return p


# ══════════════════════════════════════════════════════════════════════
# 9. AGGREGATE EVALUATION ACROSS METHODS
# ══════════════════════════════════════════════════════════════════════

def evaluate_all_methods_logistic(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    imputed_datasets: dict,
    predictor_names: list = None,
    alpha: float = 0.05,
) -> dict:
    """Evaluate logistic model preservation for multiple imputation methods.

    Parameters
    ----------
    X_ref, y_ref : np.ndarray
        Reference predictor matrix and binary outcome.
    imputed_datasets : dict[str, np.ndarray]
        Mapping of method_name → imputed predictor matrix X_imp.
    predictor_names : list of str, optional
    alpha : float

    Returns
    -------
    dict[str, dict] : method_name → full logistic evaluation results.
    """
    results = {}
    for method_name, X_imp in imputed_datasets.items():
        results[method_name] = evaluate_logistic_model(
            X_ref, y_ref, X_imp, predictor_names, alpha
        )
    return results


# ══════════════════════════════════════════════════════════════════════
# 10. SELF-TEST (executed when run directly)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Generate synthetic data for testing
    rng = np.random.RandomState(42)
    N, p = 300, 8

    X_ref = rng.randn(N, p)
    true_beta = rng.randn(p) * 0.5
    logit = X_ref @ true_beta
    y_prob = 1.0 / (1.0 + np.exp(-logit))
    y_ref = (y_prob > 0.5).astype(int)

    # Simulate imputed data with small Gaussian noise
    X_imp = X_ref + rng.randn(N, p) * 0.15

    pred_names = [f"var_{i}" for i in range(p)]

    # ── Logistic evaluation ──
    print("=" * 60)
    print("LOGISTIC REGRESSION PRESERVATION TEST")
    print("=" * 60)

    results = evaluate_logistic_model(X_ref, y_ref, X_imp, pred_names)

    print(f"  AUC: ref={results['calibration']['auc_ref']:.4f}, "
          f"imp={results['calibration']['auc_imp']:.4f}")
    print(f"  Brier: ref={results['calibration']['brier_ref']:.4f}, "
          f"imp={results['calibration']['brier_imp']:.4f}")
    print(f"  McFadden R²: ref={results['performance_summary']['mcfadden_r2_ref']:.4f}, "
          f"imp={results['performance_summary']['mcfadden_r2_imp']:.4f}")
    print(f"  beta bias (mean abs): {results['coefficient_bias']['abs_bias_mean']:.4f}")
    print(f"  SE ratio (mean): {results['standard_error']['se_ratio_mean']:.4f}")
    print(f"  Significance change: {results['standard_error']['significance_change']:.4f}")
    print(f"  CI width ratio: {results['confidence_interval']['ci_width_ratio_mean']:.4f}")
    print(f"  CI overlap: {results['confidence_interval']['ci_overlap_mean']:.4f}")
    print(f"  CI coverage: {results['confidence_interval']['ci_coverage']:.4f}")
    hl = results['calibration']['hosmer_lemeshow_imp']
    print(f"  Hosmer-Lemeshow chi2={hl['chi2_stat']:.2f}, p={hl['p_value']:.4f}, dof={hl['dof']}")

    # ── Linear evaluation ──
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION PRESERVATION TEST")
    print("=" * 60)

    y_cont = X_ref[:, 0] * 0.8 + X_ref[:, 1] * 0.3 + rng.randn(N) * 0.5
    lin_results = evaluate_linear_model(X_ref, y_cont, X_imp, pred_names)

    print(f"  R²: ref={lin_results['goodness_of_fit']['r2_ref']:.4f}, "
          f"imp={lin_results['goodness_of_fit']['r2_imp']:.4f}")
    print(f"  RMSE: ref={lin_results['goodness_of_fit']['rmse_ref']:.4f}, "
          f"imp={lin_results['goodness_of_fit']['rmse_imp']:.4f}")
    print(f"  beta bias (mean abs): {lin_results['coefficient_bias']['abs_bias_mean']:.4f}")
    print(f"  SE ratio (mean): {lin_results['standard_error']['se_ratio_mean']:.4f}")
    print(f"  Residual std: ref={lin_results['residual_diagnostics']['std_residual_ref']:.4f}, "
          f"imp={lin_results['residual_diagnostics']['std_residual_imp']:.4f}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
