# -*- coding: utf-8 -*-
"""
bootstrap.py — Bootstrap confidence intervals for imputation quality metrics
=============================================================================

This module provides parametric bootstrap confidence intervals for
imputation quality metrics (RMSE, MAE, F1, Variance Ratio).

The percentile method (Efron & Tibshirani 1993, §13.3) is used by
resampling rows with replacement.

References:
    Efron, B. & Tibshirani, R.J. (1993).
    An Introduction to the Bootstrap. Chapman & Hall/CRC.
    
    Carpenter, J. & Bithell, J. (2000).
    Bootstrap confidence intervals: when, which, what? A practical guide
    for medical statisticians. Statistics in Medicine 19(9):1141-1164.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from sklearn.utils import check_random_state


def bootstrap_ci(
    X_true: np.ndarray,
    X_imp: np.ndarray,
    mask: np.ndarray,
    metric_func: Callable,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    random_state: int = 42,
    **metric_kwargs,
) -> dict:
    """Compute bootstrap confidence interval for an imputation quality metric.

    Uses the percentile method (Efron & Tibshirani 1993, §13.3) to estimate
    confidence intervals by resampling rows with replacement.

    Args:
        X_true:           Complete reference matrix ``(N, P)``.
        X_imp:            Imputed matrix ``(N, P)``.
        mask:             Boolean mask ``(N, P)`` — True where imputed.
        metric_func:      Function computing the metric. Must accept
                          (X_true, X_imp, mask, **kwargs) and return a scalar.
        n_bootstrap:      Number of bootstrap replications (default 200).
                          Increase to 1000+ for publication-quality CIs.
                          Note: computation time scales linearly with n_bootstrap.
        confidence_level: Confidence level (default 0.95 for 95% CI).
        random_state:     Integer seed for reproducibility.
        **metric_kwargs:  Additional arguments passed to metric_func.

    Returns:
        Dictionary with keys:
            - "point_estimate": metric computed on original data
            - "CI_lower": lower bound of confidence interval
            - "CI_upper": upper bound of confidence interval
            - "bootstrap_mean": mean of bootstrap distribution
            - "bootstrap_std": standard deviation of bootstrap distribution
            - "n_bootstrap": number of replications performed

    Example:
        >>> from evaluation.continuous_metrics import rmse_on_imputed_cells
        >>> result = bootstrap_ci(
        ...     X_true=X_complete,
        ...     X_imp=X_imputed,
        ...     mask=imputed_mask,
        ...     metric_func=lambda X_t, X_i, m, idx: rmse_on_imputed_cells(X_t, X_i, m, idx)["mean_rmse"],
        ...     n_bootstrap=200,
        ...     continuous_idx=[0, 1, 2, 3]
        ... )
        >>> print(f"RMSE = {result['point_estimate']:.2f} "
        ...       f"[{result['CI_lower']:.2f}, {result['CI_upper']:.2f}]")
    """
    rng = check_random_state(random_state)
    N = X_true.shape[0]

    # Original metric value
    point_estimate = float(metric_func(X_true, X_imp, mask, **metric_kwargs))

    # Bootstrap replications
    bootstrap_values = []
    for _ in range(n_bootstrap):
        # Resample rows with replacement
        idx = rng.choice(N, size=N, replace=True)
        X_true_boot = X_true[idx, :]
        X_imp_boot = X_imp[idx, :]
        mask_boot = mask[idx, :]

        try:
            val = float(metric_func(X_true_boot, X_imp_boot, mask_boot, **metric_kwargs))
            if not np.isnan(val) and not np.isinf(val):
                bootstrap_values.append(val)
        except Exception:
            # Skip failed bootstrap samples
            continue

    if len(bootstrap_values) < 10:
        # Too few successful replications
        return {
            "point_estimate": point_estimate,
            "CI_lower": np.nan,
            "CI_upper": np.nan,
            "bootstrap_mean": np.nan,
            "bootstrap_std": np.nan,
            "n_bootstrap": len(bootstrap_values),
        }

    # Percentile confidence interval
    alpha = 1.0 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    CI_lower = float(np.percentile(bootstrap_values, lower_percentile))
    CI_upper = float(np.percentile(bootstrap_values, upper_percentile))
    bootstrap_mean = float(np.mean(bootstrap_values))
    bootstrap_std = float(np.std(bootstrap_values, ddof=1))

    return {
        "point_estimate": round(point_estimate, 6),
        "CI_lower": round(CI_lower, 6),
        "CI_upper": round(CI_upper, 6),
        "bootstrap_mean": round(bootstrap_mean, 6),
        "bootstrap_std": round(bootstrap_std, 6),
        "n_bootstrap": len(bootstrap_values),
    }


def bootstrap_ci_multiple_metrics(
    X_true: np.ndarray,
    X_imp: np.ndarray,
    mask: np.ndarray,
    metric_configs: List[dict],
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> List[dict]:
    """Compute bootstrap CIs for multiple metrics efficiently.

    Reuses the same bootstrap samples for all metrics to reduce computation time.
    This is more efficient than calling bootstrap_ci() multiple times.

    Args:
        X_true:          Complete reference matrix ``(N, P)``.
        X_imp:           Imputed matrix ``(N, P)``.
        mask:            Boolean mask ``(N, P)`` — True where imputed.
        metric_configs:  List of dicts, each with keys:
                         - "name": metric name (e.g., "RMSE")
                         - "func": metric function
                         - "kwargs": dict of additional arguments
        n_bootstrap:     Number of bootstrap replications (default 200).
        confidence_level: Confidence level (default 0.95).
        random_state:    Integer seed.

    Returns:
        List of dicts, one per metric, with CI results.

    Example:
        >>> configs = [
        ...     {"name": "RMSE", "func": compute_rmse, "kwargs": {"continuous_idx": [0,1,2]}},
        ...     {"name": "MAE", "func": compute_mae, "kwargs": {"continuous_idx": [0,1,2]}},
        ...     {"name": "F1", "func": compute_f1, "kwargs": {"categorical_idx": [3,4]}},
        ... ]
        >>> results = bootstrap_ci_multiple_metrics(
        ...     X_true, X_imp, mask, configs, n_bootstrap=200
        ... )
        >>> for r in results:
        ...     print(f"{r['metric_name']}: {r['point_estimate']:.3f} "
        ...           f"[{r['CI_lower']:.3f}, {r['CI_upper']:.3f}]")
    """
    rng = check_random_state(random_state)
    N = X_true.shape[0]
    n_metrics = len(metric_configs)

    # Original metric values
    point_estimates = []
    for config in metric_configs:
        try:
            val = float(config["func"](X_true, X_imp, mask, **config.get("kwargs", {})))
        except Exception:
            val = np.nan
        point_estimates.append(val)

    # Bootstrap replications
    bootstrap_values = [[] for _ in range(n_metrics)]

    for _ in range(n_bootstrap):
        # Resample rows with replacement
        idx = rng.choice(N, size=N, replace=True)
        X_true_boot = X_true[idx, :]
        X_imp_boot = X_imp[idx, :]
        mask_boot = mask[idx, :]

        for i, config in enumerate(metric_configs):
            try:
                val = float(config["func"](X_true_boot, X_imp_boot, mask_boot, **config.get("kwargs", {})))
                if not np.isnan(val) and not np.isinf(val):
                    bootstrap_values[i].append(val)
            except Exception:
                continue

    # Compute CIs
    results = []
    alpha = 1.0 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    for i, config in enumerate(metric_configs):
        if len(bootstrap_values[i]) < 10:
            results.append({
                "metric_name": config["name"],
                "point_estimate": point_estimates[i],
                "CI_lower": np.nan,
                "CI_upper": np.nan,
                "bootstrap_mean": np.nan,
                "bootstrap_std": np.nan,
                "n_bootstrap": len(bootstrap_values[i]),
            })
        else:
            CI_lower = float(np.percentile(bootstrap_values[i], lower_percentile))
            CI_upper = float(np.percentile(bootstrap_values[i], upper_percentile))
            bootstrap_mean = float(np.mean(bootstrap_values[i]))
            bootstrap_std = float(np.std(bootstrap_values[i], ddof=1))

            results.append({
                "metric_name": config["name"],
                "point_estimate": round(point_estimates[i], 6),
                "CI_lower": round(CI_lower, 6),
                "CI_upper": round(CI_upper, 6),
                "bootstrap_mean": round(bootstrap_mean, 6),
                "bootstrap_std": round(bootstrap_std, 6),
                "n_bootstrap": len(bootstrap_values[i]),
            })

    return results
