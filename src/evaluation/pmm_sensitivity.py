# -*- coding: utf-8 -*-
"""
pmm_sensitivity.py — PMM donor sensitivity analysis
====================================================

This module provides sensitivity analysis for Predictive Mean Matching (PMM)
with respect to the number of donors (k parameter).

van Buuren (2018, §3.4) recommends:
- k=3:  for small samples (N < 100)
- k=5:  for medium samples (100 ≤ N ≤ 500) — most common
- k=10: for large samples (N > 500)

This analysis compares imputation quality across different k values to assess
the robustness of PMM-based MICE to this hyperparameter.

References:
    van Buuren, S. (2018).
    Flexible Imputation of Missing Data (2nd ed.).
    CRC Press. Chapter 3: Univariate missing data, §3.4 Predictive mean matching.
    
    Morris, T.P., White, I.R., & Royston, P. (2014).
    Tuning multiple imputation by predictive mean matching and local residual
    draws. BMC Medical Research Methodology 14:75.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd

from ..imputers.mice_imputer import impute_mice
from ..evaluation.continuous_metrics import rmse_on_imputed_cells, mae_on_imputed_cells
from ..evaluation.variance_correlation import variance_ratio


def pmm_sensitivity_analysis(
    X_miss: np.ndarray,
    X_true: np.ndarray,
    continuous_idx: List[int],
    binary_idx: List[int],
    ordinal_idx: List[int],
    k_values: List[int] = [3, 5, 10],
    max_iter: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare MICE-PMM performance across different numbers of donors.

    For each k value in k_values, performs MICE imputation with PMM using
    k donors and evaluates reconstruction quality on continuous variables.

    Args:
        X_miss:         Incomplete data matrix ``(N, P)`` with NaN.
        X_true:         Complete reference matrix ``(N, P)`` (ground truth).
        continuous_idx: Indices of continuous variables (PMM applied).
        binary_idx:     Indices of binary variables (logistic regression).
        ordinal_idx:    Indices of ordinal variables (multinomial logistic).
        k_values:       List of k donor values to test (default [3, 5, 10]).
        max_iter:       Number of MICE cycles (default 10).
        random_state:   Integer seed for reproducibility.

    Returns:
        pandas DataFrame with columns:
            - k_donors: number of PMM donors
            - rmse: root mean squared error
            - mae: mean absolute error
            - r_v_cell: cell-level variance ratio
            - r_v_global: global variance ratio
            - execution_time_s: computation time in seconds

    Example:
        >>> results = pmm_sensitivity_analysis(
        ...     X_miss=X_incomplete,
        ...     X_true=X_complete,
        ...     continuous_idx=[0, 1, 2, 3],
        ...     binary_idx=[4],
        ...     ordinal_idx=[5],
        ...     k_values=[3, 5, 10],
        ... )
        >>> print(results)
           k_donors   rmse    mae  r_v_cell  r_v_global  execution_time_s
        0         3  25.12  18.34      0.98        0.95              3.21
        1         5  24.87  18.12      0.99        0.96              3.45
        2        10  24.92  18.19      0.97        0.96              3.78
    """
    results = []
    mask = np.isnan(X_miss)

    for k in k_values:
        print(f"  Testing PMM with k={k} donors...")
        
        # Impute with MICE-PMM
        t_start = time.perf_counter()
        X_imp = impute_mice(
            X_miss=X_miss,
            continuous_idx=continuous_idx,
            binary_idx=binary_idx,
            ordinal_idx=ordinal_idx,
            max_iter=max_iter,
            k_donors=k,
            random_state=random_state,
        )
        t_elapsed = time.perf_counter() - t_start

        # Evaluate reconstruction quality
        rmse_result = rmse_on_imputed_cells(X_true, X_imp, mask, continuous_idx)
        mae_result = mae_on_imputed_cells(X_true, X_imp, mask, continuous_idx)
        var_result = variance_ratio(X_true, X_imp, mask, continuous_idx)

        results.append({
            "k_donors": k,
            "rmse": round(rmse_result.get("mean_rmse", np.nan), 6),
            "mae": round(mae_result.get("mean_mae", np.nan), 6),
            "r_v_cell": round(var_result.get("mean_R_V_cell", np.nan), 6),
            "r_v_global": round(var_result.get("mean_R_V_global", np.nan), 6),
            "execution_time_s": round(t_elapsed, 2),
        })

    return pd.DataFrame(results)


def interpret_pmm_sensitivity(results: pd.DataFrame) -> str:
    """Generate interpretation of PMM sensitivity analysis results.

    Args:
        results: DataFrame from pmm_sensitivity_analysis().

    Returns:
        Markdown-formatted interpretation string.

    Example:
        >>> interpretation = interpret_pmm_sensitivity(results)
        >>> print(interpretation)
    """
    if results.empty:
        return "No results to interpret."

    best_rmse_idx = results["rmse"].idxmin()
    best_k = int(results.loc[best_rmse_idx, "k_donors"])
    best_rmse = float(results.loc[best_rmse_idx, "rmse"])
    
    rmse_range = float(results["rmse"].max() - results["rmse"].min())
    rmse_cv = float(results["rmse"].std() / results["rmse"].mean() * 100)

    interpretation = f"""
## PMM Donor Sensitivity Analysis

**Summary:**
- Best k donors: {best_k} (RMSE = {best_rmse:.2f})
- RMSE range: {rmse_range:.2f} ({rmse_cv:.1f}% CV)
- Variance ratio (cell-level): {results['r_v_cell'].mean():.3f} ± {results['r_v_cell'].std():.3f}
- Variance ratio (global): {results['r_v_global'].mean():.3f} ± {results['r_v_global'].std():.3f}

**Interpretation:**
"""

    if rmse_cv < 5.0:
        interpretation += f"""
PMM is **highly robust** to the choice of k donors (CV < 5%). 
The impact of k on reconstruction quality is negligible.
Any value in the tested range [{results['k_donors'].min()}, {results['k_donors'].max()}] 
is appropriate for this dataset (N={results.shape[0]}).
"""
    elif rmse_cv < 10.0:
        interpretation += f"""
PMM is **moderately robust** to the choice of k donors (5% ≤ CV < 10%).
Small differences in reconstruction quality are observed, but all tested 
values produce acceptable results. k={best_k} provides the best balance.
"""
    else:
        interpretation += f"""
PMM is **sensitive** to the choice of k donors (CV ≥ 10%).
The number of donors significantly impacts reconstruction quality.
**Recommendation:** Use k={best_k} for this dataset.
"""

    # Execution time analysis
    time_increase = float((results["execution_time_s"].max() / results["execution_time_s"].min() - 1) * 100)
    interpretation += f"""

**Computational cost:**
- Execution time increases by {time_increase:.1f}% from k={results['k_donors'].min()} 
  to k={results['k_donors'].max()}.
- This is expected as larger k requires searching more donor candidates.
"""

    return interpretation


# Helper function for integration in run.py
def run_pmm_sensitivity_from_config(
    X_miss: np.ndarray,
    X_true: np.ndarray,
    config: dict,
    output_path: str,
) -> None:
    """Run PMM sensitivity analysis and save results.

    Args:
        X_miss:      Incomplete data matrix.
        X_true:      Complete reference matrix.
        config:      Configuration dict with keys:
                     - continuous_idx
                     - binary_idx
                     - ordinal_idx
                     - k_values (optional, default [3, 5, 10])
                     - random_state
        output_path: Path to save results CSV.

    Example:
        >>> run_pmm_sensitivity_from_config(
        ...     X_miss, X_true, config,
        ...     output_path="outputs/pmm_sensitivity.csv"
        ... )
    """
    results = pmm_sensitivity_analysis(
        X_miss=X_miss,
        X_true=X_true,
        continuous_idx=config["continuous_idx"],
        binary_idx=config.get("binary_idx", []),
        ordinal_idx=config.get("ordinal_idx", []),
        k_values=config.get("k_values", [3, 5, 10]),
        max_iter=config.get("max_iter", 10),
        random_state=config.get("random_state", 42),
    )
    
    results.to_csv(output_path, index=False)
    print(f"✓ PMM sensitivity analysis saved to {output_path}")
    
    # Generate and print interpretation
    interp = interpret_pmm_sensitivity(results)
    print(interp)
