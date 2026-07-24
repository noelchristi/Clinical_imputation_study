# -*- coding: utf-8 -*-
"""
rubin.py — Rubin's combination rules for multiple imputation
=============================================================
Rubin's rules allow valid inference to be drawn from M independently
imputed datasets by pooling estimated parameters (regression coefficients,
odds ratios, hazard ratios, etc.) and their variances.

The key components are:

  Q̄  = (1/M) Σ_m Q̂_m              (pooled point estimate)
  W  = (1/M) Σ_m SE²_m             (within-imputation variance)
  B  = 1/(M−1) Σ_m (Q̂_m − Q̄)²    (between-imputation variance)
  T  = W + (1 + 1/M) B             (total variance)
  λ  = (1 + 1/M) B / T             (fraction of missing information)
  ν  = (M − 1) / λ²                (degrees of freedom, Rubin 1987)
  FMI = λ                          (fraction of missing information)
  RE  = 1 / (1 + FMI/M)            (relative efficiency)

CRITICAL correctness requirement:
  - Q̂_m must be estimated parameters (β, OR, log(HR), etc.) from a
    fitted statistical model, NOT raw data means or variances.
  - SE²_m is the squared standard error of Q̂_m from model m.
  - Each of the M datasets must contain the **same N individuals** as the
    original incomplete dataset (no bootstrapping of rows before imputation).

This module provides two functions:

1. **rubin_pool_parameters()**: Proper Rubin pooling for user-supplied
   parameter estimates (e.g., from regression models fitted externally).

2. **rubin_descriptive_statistics()**: Descriptive summary of column means
   across M datasets (NOT proper Rubin pooling, provided for backward
   compatibility and exploratory analysis).

Reference:
    Rubin, D.B. (1987).
    Multiple Imputation for Nonresponse in Surveys.
    Wiley, New York.  Chapters 3–4.

    Barnard, J. & Rubin, D.B. (1999). Small-sample degrees of freedom with
    multiple imputation.  Biometrika 86(4):948–955.
    
    Marshall, A., Altman, D.G., Holder, R.L., & Royston, P. (2009).
    Combining estimates of interest in prognostic modelling studies after
    multiple imputation: current practice and guidelines.
    BMC Medical Research Methodology 9:57.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Literal

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def fit_linear_model_on_imputed_datasets(
    imputed_datasets: List[np.ndarray],
    outcome_idx: int,
    predictor_indices: List[int],
    model_type: Literal["linear", "logistic"] = "linear",
) -> Dict[str, List[float]]:
    """Fit regression models on M imputed datasets and extract parameter estimates.

    This function fits the same regression model on each of the M imputed
    datasets and extracts the regression coefficients and their standard errors.
    These can then be pooled using rubin_pool_parameters().

    Args:
        imputed_datasets:  List of M complete arrays, shape ``(N, P)``.
        outcome_idx:       Column index of the outcome variable (y).
        predictor_indices: List of column indices for predictor variables (X).
        model_type:        "linear" for continuous outcome (OLS),
                           "logistic" for binary outcome (logistic regression).

    Returns:
        Dictionary with keys for each predictor and "intercept":
            - Each value is a dict with keys:
                - "estimates": list of M coefficients
                - "std_errors": list of M standard errors

    Example:
        >>> # Fit logistic regression: HbA1c_high ~ age + BMI + glucose
        >>> results = fit_linear_model_on_imputed_datasets(
        ...     imputed_datasets=datasets,
        ...     outcome_idx=col_HbA1c_binary,
        ...     predictor_indices=[col_age, col_BMI, col_glucose],
        ...     model_type="logistic"
        ... )
        >>> # Pool the coefficient for BMI
        >>> bmi_pooled = rubin_pool_parameters(
        ...     estimates=results["predictor_1"]["estimates"],
        ...     std_errors=results["predictor_1"]["std_errors"],
        ...     parameter_name="BMI_coef"
        ... )
    """
    M = len(imputed_datasets)
    n_predictors = len(predictor_indices)

    # Storage for coefficients and SEs across M datasets
    intercepts = []
    intercept_ses = []
    coef_arrays = [[] for _ in range(n_predictors)]
    se_arrays = [[] for _ in range(n_predictors)]

    for m, dataset in enumerate(imputed_datasets):
        y = dataset[:, outcome_idx]
        X = dataset[:, predictor_indices]

        if model_type == "linear":
            # Fit OLS
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)

            # Extract coefficients
            intercept = float(model.intercept_)
            coefs = model.coef_  # shape (n_predictors,)

            # Compute standard errors
            # SE = sqrt(diag((X'X)^{-1} * MSE))
            y_pred = model.predict(X)
            residuals = y - y_pred
            n, p = X.shape
            dof = max(n - p - 1, 1)
            mse = float(np.sum(residuals ** 2) / dof)

            # Add intercept column
            X_with_intercept = np.column_stack([np.ones(n), X])
            try:
                XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                var_coef = mse * np.diag(XtX_inv)
                se_coef = np.sqrt(var_coef)
                intercept_se = float(se_coef[0])
                predictor_ses = se_coef[1:]
            except np.linalg.LinAlgError:
                # Singular matrix: assign large SE
                intercept_se = np.nan
                predictor_ses = np.full(n_predictors, np.nan)

        elif model_type == "logistic":
            # Fit logistic regression
            from sklearn.linear_model import LogisticRegression as LogReg
            model = LogReg(max_iter=3000, solver="newton-cholesky", random_state=0)
            
            # Check if binary outcome
            unique_y = np.unique(y)
            if len(unique_y) < 2:
                # Degenerate case: all same class
                intercept = np.nan
                intercept_se = np.nan
                coefs = np.full(n_predictors, np.nan)
                predictor_ses = np.full(n_predictors, np.nan)
            else:
                model.fit(X, y)
                intercept = float(model.intercept_[0])
                coefs = model.coef_[0]  # shape (n_predictors,)

                # Compute standard errors using Hessian approximation
                # SE ≈ sqrt(diag((X'WX)^{-1})) where W = diag(p*(1-p))
                p_pred = model.predict_proba(X)[:, 1]
                w = p_pred * (1 - p_pred)
                w = np.clip(w, 1e-8, 1 - 1e-8)  # numerical stability

                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                W_diag = np.diag(w)
                try:
                    XtWX = X_with_intercept.T @ W_diag @ X_with_intercept
                    XtWX_inv = np.linalg.inv(XtWX)
                    se_coef = np.sqrt(np.diag(XtWX_inv))
                    intercept_se = float(se_coef[0])
                    predictor_ses = se_coef[1:]
                except np.linalg.LinAlgError:
                    intercept_se = np.nan
                    predictor_ses = np.full(n_predictors, np.nan)
        else:
            raise ValueError(f"model_type must be 'linear' or 'logistic'; got {model_type}")

        intercepts.append(intercept)
        intercept_ses.append(intercept_se)
        for i, (coef, se) in enumerate(zip(coefs, predictor_ses)):
            coef_arrays[i].append(float(coef))
            se_arrays[i].append(float(se))

    # Package results
    results = {
        "intercept": {
            "estimates": intercepts,
            "std_errors": intercept_ses,
        }
    }
    for i in range(n_predictors):
        results[f"predictor_{i}"] = {
            "estimates": coef_arrays[i],
            "std_errors": se_arrays[i],
        }

    return results


def rubin_pool_parameters(
    estimates: List[float],
    std_errors: List[float],
    parameter_name: str = "Q",
) -> Dict:
    """Apply Rubin's combination rules to pool parameter estimates.

    This is the proper implementation of Rubin (1987) for multiple imputation
    inference. Use this function when you have fitted M statistical models
    (e.g., linear regression, logistic regression, Cox model) on M imputed
    datasets and want to combine the parameter estimates.

    Args:
        estimates:      List of M point estimates (Q̂₁, ..., Q̂ₘ).
                        Examples: regression coefficients β, log odds ratios,
                        hazard ratios, correlation coefficients.
        std_errors:     List of M standard errors (SE₁, ..., SEₘ).
                        These are the standard errors of the estimates from
                        each model (obtained from model.summary() or similar).
        parameter_name: Name of the parameter for output labeling (default "Q").

    Returns:
        Dictionary with keys:
            - "parameter": parameter name
            - "M": number of imputations
            - "Q_bar": pooled point estimate
            - "W": within-imputation variance
            - "B": between-imputation variance
            - "T": total variance
            - "SE": pooled standard error (sqrt(T))
            - "lambda": fraction of missing information
            - "nu": degrees of freedom
            - "FMI": fraction of missing information (alias for lambda)
            - "RE": relative efficiency
            - "CI_lower": 95% confidence interval lower bound
            - "CI_upper": 95% confidence interval upper bound

    Raises:
        ValueError: If M < 2 or if estimates and std_errors have different lengths.

    Example:
        >>> # Fit logistic regression on M=5 imputed datasets
        >>> betas = [0.52, 0.48, 0.55, 0.50, 0.53]  # coefficients for "age"
        >>> ses = [0.12, 0.13, 0.11, 0.12, 0.13]     # standard errors
        >>> result = rubin_pool_parameters(betas, ses, parameter_name="age_coef")
        >>> print(f"Pooled β = {result['Q_bar']:.3f} ± {result['SE']:.3f}")
    """
    M = len(estimates)
    if M < 2:
        raise ValueError(f"Rubin's rules require M >= 2 imputations; got {M}.")
    if len(std_errors) != M:
        raise ValueError(f"estimates and std_errors must have same length; got {M} vs {len(std_errors)}.")

    Q_hat = np.array(estimates, dtype=float)
    SE = np.array(std_errors, dtype=float)

    # Step 1: Pooled point estimate (Rubin 1987, Eq. 3.1.2)
    Q_bar = float(np.mean(Q_hat))

    # Step 2: Within-imputation variance (Rubin 1987, Eq. 3.1.3)
    W = float(np.mean(SE ** 2))

    # Step 3: Between-imputation variance (Rubin 1987, Eq. 3.1.4)
    B = float(np.var(Q_hat, ddof=1))

    # Step 4: Total variance (Rubin 1987, Eq. 3.1.5)
    T = W + (1.0 + 1.0 / M) * B

    # Step 5: Pooled standard error
    SE_pooled = float(np.sqrt(T))

    # Step 6: Fraction of missing information (Rubin 1987, Eq. 3.1.7)
    lambda_ = ((1.0 + 1.0 / M) * B) / T if T > 1e-12 else 0.0

    # Step 7: Degrees of freedom (Rubin 1987, Eq. 3.1.6)
    nu = (M - 1) / (lambda_ ** 2) if lambda_ > 1e-12 else np.inf

    # Step 8: Relative efficiency (Rubin 1987, Eq. 4.2.8)
    RE = 1.0 / (1.0 + lambda_ / M)

    # Step 9: 95% Confidence interval using t-distribution
    # For large nu, t_nu,0.025 ≈ 1.96; for small nu, wider
    if np.isinf(nu):
        t_crit = 1.96  # z-score for normal distribution
    else:
        from scipy.stats import t as t_dist
        t_crit = float(t_dist.ppf(0.975, df=nu))

    CI_lower = Q_bar - t_crit * SE_pooled
    CI_upper = Q_bar + t_crit * SE_pooled

    return {
        "parameter": parameter_name,
        "M": M,
        "Q_bar": round(Q_bar, 6),
        "W": round(W, 6),
        "B": round(B, 6),
        "T": round(T, 6),
        "SE": round(SE_pooled, 6),
        "lambda": round(float(lambda_), 6),
        "nu": round(float(nu), 2) if not np.isinf(nu) else "Inf",
        "FMI": round(float(lambda_), 6),
        "RE": round(float(RE), 6),
        "CI_lower": round(CI_lower, 6),
        "CI_upper": round(CI_upper, 6),
    }


def rubin_descriptive_statistics(
    imputed_datasets: List[np.ndarray],
    variable_indices: Optional[List[int]] = None,
    col_names: Optional[List[str]] = None,
) -> List[dict]:
    """Compute descriptive statistics across M imputed datasets.

    WARNING: This function computes summary statistics on RAW DATA COLUMNS,
    not on estimated parameters from fitted models. This is NOT a proper
    application of Rubin's combination rules for inference.

    Use this function only for:
    - Exploratory data analysis
    - Reporting descriptive statistics in "Table 1" of a paper
    - Backward compatibility with existing code

    For valid statistical inference (hypothesis tests, confidence intervals),
    use **rubin_pool_parameters()** with parameter estimates from fitted models.

    Args:
        imputed_datasets: List of M complete arrays, each of shape ``(N, P)``.
                          All arrays must have identical shape and correspond
                          to the same N individuals (no bootstrapping).
        variable_indices: Indices of variables to process.
                          If None, all P columns are processed.
        col_names:        Optional column names for output labelling.

    Returns:
        List of dicts, one per variable, with keys:
        ``variable_idx``, ``variable_name``,
        ``mean_across_M`` (average of column means),
        ``sd_within`` (average of column SDs),
        ``sd_between`` (SD of column means across M datasets).

    Raises:
        ValueError: If M < 2.
        ValueError: If imputed datasets have different shapes.
    """
    M = len(imputed_datasets)
    if M < 2:
        raise ValueError(f"Need M >= 2 imputed datasets; got {M}.")

    shapes = {arr.shape for arr in imputed_datasets}
    if len(shapes) > 1:
        raise ValueError(f"All imputed datasets must have the same shape; found {shapes}.")

    N, P = imputed_datasets[0].shape
    idx = variable_indices if variable_indices is not None else list(range(P))

    # Stack to (M, N, P) for vectorised computation
    stack = np.stack(imputed_datasets, axis=0)  # (M, N, P)

    results = []
    for j in idx:
        col_stack = stack[:, :, j]  # (M, N)

        # Per-dataset means and standard deviations
        means = np.mean(col_stack, axis=1)      # (M,)
        sds = np.std(col_stack, axis=1, ddof=1)  # (M,)

        mean_across_M = float(np.mean(means))
        sd_within = float(np.mean(sds))
        sd_between = float(np.std(means, ddof=1))

        name = col_names[j] if col_names and j < len(col_names) else f"var_{j}"
        results.append({
            "variable_idx": j,
            "variable_name": name,
            "M": M,
            "mean_across_M": round(mean_across_M, 6),
            "sd_within": round(sd_within, 6),
            "sd_between": round(sd_between, 6),
        })

    return results


# Backward compatibility alias
def rubin_rules(*args, **kwargs):
    """Deprecated: use rubin_pool_parameters() or rubin_descriptive_statistics().
    
    This function is kept for backward compatibility with existing code that
    called rubin_rules(). It now calls rubin_descriptive_statistics().
    
    For proper Rubin pooling of statistical parameters, use
    rubin_pool_parameters() instead.
    """
    import warnings
    warnings.warn(
        "rubin_rules() is deprecated. Use rubin_pool_parameters() for proper "
        "Rubin pooling or rubin_descriptive_statistics() for exploratory analysis.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rubin_descriptive_statistics(*args, **kwargs)
