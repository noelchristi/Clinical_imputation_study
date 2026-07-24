# -*- coding: utf-8 -*-
"""
sensitivity.py — Sensitivity analysis and robustness checking
=============================================================
Evaluates the stability of imputation performance:

  1. **Sensitivity analysis**: how RMSE rankings change when the missing
     rate is varied from 10 % to 40 %, assessed via Spearman rank
     correlation of method rankings across scenarios.

  2. **Robustness bootstrap**: 95 % bootstrap confidence intervals for
     RMSE and AUC, using B = 1000 resamples of the imputed cells.

References:
    Efron, B. & Tibshirani, R.J. (1993).
    An Introduction to the Bootstrap.  Chapman & Hall.

    Saltelli, A. et al. (2008).
    Global Sensitivity Analysis.  Wiley.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr


def sensitivity_analysis(
    results_by_rate: Dict[float, Dict[str, float]],
    metric: str = "mean_RMSE",
) -> dict:
    """Assess rank stability of methods across missing rates.

    For each pair of missing rates, compute the Spearman rank correlation
    of method performance values.  High correlation → stable rankings.

    Args:
        results_by_rate: Mapping {missing_rate → {method_name → metric_value}}.
                         Example: {0.10: {"MissForest": 2.1, "KNN": 3.4}, ...}
        metric:          Name of the metric (for logging only).

    Returns:
        dict with:
          ``"rates"``: sorted list of rates.
          ``"rank_correlations"``: list of dicts {rate_a, rate_b, rho, p}.
          ``"mean_rho"``: mean Spearman ρ across all pairs.
    """
    rates = sorted(results_by_rate.keys())
    if len(rates) < 2:
        return {"rates": rates, "rank_correlations": [], "mean_rho": np.nan}

    correlations = []
    for i in range(len(rates)):
        for j in range(i + 1, len(rates)):
            ra, rb = rates[i], rates[j]
            methods = sorted(
                set(results_by_rate[ra].keys()) & set(results_by_rate[rb].keys())
            )
            if len(methods) < 2:
                continue
            vals_a = [results_by_rate[ra][m] for m in methods]
            vals_b = [results_by_rate[rb][m] for m in methods]
            rho, p = spearmanr(vals_a, vals_b)
            correlations.append({
                "rate_a": ra, "rate_b": rb,
                "rho": float(rho), "p_value": float(p),
                "n_methods": len(methods),
            })

    mean_rho = float(np.mean([c["rho"] for c in correlations])) if correlations else np.nan
    return {
        "metric": metric,
        "rates": rates,
        "rank_correlations": correlations,
        "mean_rho": mean_rho,
    }


def bootstrap_ci_rmse(
    X_true: np.ndarray,
    X_imp: np.ndarray,
    col_mask: np.ndarray,
    var_idx: int,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap 95 % CI for RMSE on imputed cells of one variable.

    Resamples the imputed cells with replacement ``n_bootstrap`` times,
    computing RMSE on each resample.  Returns the percentile interval.

    Args:
        X_true:      Complete reference matrix ``(N, P)``.
        X_imp:       Imputed matrix ``(N, P)``.
        col_mask:    Boolean ``(N,)`` — True where cell was imputed.
        var_idx:     Column index.
        n_bootstrap: Number of bootstrap resamples (default 1000).
        alpha:       Significance level (default 0.05 → 95 % CI).
        seed:        Random seed.

    Returns:
        dict with ``"rmse_observed"``, ``"ci_lower"``, ``"ci_upper"``.
    """
    rng = np.random.default_rng(seed)
    true_vals = X_true[col_mask, var_idx]
    imp_vals = X_imp[col_mask, var_idx]
    n = len(true_vals)

    if n < 2:
        return {"rmse_observed": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    rmse_obs = float(np.sqrt(np.mean((imp_vals - true_vals) ** 2)))
    boot_rmse = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_rmse[b] = np.sqrt(np.mean((imp_vals[idx] - true_vals[idx]) ** 2))

    lo = float(np.percentile(boot_rmse, 100 * alpha / 2))
    hi = float(np.percentile(boot_rmse, 100 * (1 - alpha / 2)))
    return {"rmse_observed": rmse_obs, "ci_lower": lo, "ci_upper": hi}


def robustness_summary(
    all_results: List[dict],
    method_col: str = "method",
    rmse_col: str = "RMSE",
) -> dict:
    """Aggregate mean ± SD of RMSE per method across all MC iterations.

    Provides a high-level robustness summary: a method with low SD is
    stable across Monte Carlo repetitions; high SD indicates sensitivity
    to the random seed.

    Args:
        all_results: List of per-run result dicts (each containing at least
                     ``method_col`` and ``rmse_col`` fields).
        method_col:  Key for method name in each record.
        rmse_col:    Key for the RMSE value in each record.

    Returns:
        dict mapping method name → {"mean_RMSE", "sd_RMSE", "n_runs"}.
    """
    from collections import defaultdict
    grouped: Dict[str, List[float]] = defaultdict(list)
    for r in all_results:
        method = r.get(method_col)
        rmse = r.get(rmse_col)
        if method is not None and rmse is not None and not np.isnan(float(rmse)):
            grouped[method].append(float(rmse))

    summary = {}
    for method, vals in grouped.items():
        arr = np.array(vals)
        summary[method] = {
            "mean_RMSE": float(np.mean(arr)),
            "sd_RMSE": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n_runs": len(arr),
        }
    return summary
