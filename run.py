# -*- coding: utf-8 -*-
"""
run.py — Clinical Imputation Study v4 — Main Pipeline Orchestrator
===================================================================
Executes the full simulation:
  1. Setup (logging, seed, output directories)
  2. Data ingestion and audit
  3. Monte Carlo simulation loop (mechanisms × rates × M iterations × 4 methods)
  4. Statistical analysis (Friedman, Nemenyi, Holm, Rubin, effect sizes)
  5. Visualisations (600 DPI, publication-ready)
  6. Reporting (CSV, JSON, LaTeX/Markdown tables)

Usage::

    python run.py                  # Full pipeline (default config)
    python run.py --mc 10          # 10 Monte Carlo iterations
    python run.py --skip-mnar      # Exclude MNAR mechanism
    python run.py --seed 123       # Custom random seed

IMPORTANT: uses the corrected scientific implementations only.
No legacy code from v1–v3 is called here.
"""

import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# ── Add src/ to path ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import (
    setup_logging, set_seed, create_output_dirs, timed,
    load_and_classify, audit_data, compute_descriptive_stats,
)
from amputation import AMPUTATION_REGISTRY
from imputers import impute_median_mode, impute_knn, impute_mice, impute_missforest
from evaluation import (
    evaluate_all_continuous, evaluate_all_categorical,
    variance_ratio, correlation_differences,
    clinical_model_impact, robustness_summary,
)
from stats_utils import (
    friedman_test, nemenyi_posthoc, nemenyi_cd, compute_rankings,
    rubin_rules, cohens_d, cliffs_delta, eta_squared_friedman,
)
from visualization import (
    fig_missing_pattern, fig_rmse_curves,
    fig_per_variable_rmse_heatmap, fig_rmse_ratio,
    fig_cd_diagram, fig_variance_ratio_curves, fig_boxplot_rmse,
    fig_roc_curves, fig_calibration, fig_violin_errors,
)
from reporting import (
    save_all_metrics, save_summary_table,
    save_audit_report, save_robustness_report,
    make_summary_latex, make_summary_markdown,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA = str(Path(__file__).parent / "data" / "raw" / "dataset.csv")
DEFAULT_MC = 5
DEFAULT_RATES = [0.10, 0.20, 0.40]
DEFAULT_MECHANISMS = ["MCAR", "MAR", "MNAR"]
DEFAULT_SEED = 42

METHOD_NAMES = ["Median/Mode", "KNN (k=5)", "MICE (PMM)", "MissForest"]

logger = logging.getLogger("CISv4")


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clinical Imputation Study v4")
    p.add_argument("--data", default=DEFAULT_DATA, help="Path to dataset CSV")
    p.add_argument("--mc", type=int, default=DEFAULT_MC, help="MC repetitions")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--skip-mnar", action="store_true")
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--mice-iter", type=int, default=10)
    p.add_argument("--mf-trees", type=int, default=100)
    p.add_argument("--mf-iter", type=int, default=10)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
@timed
def main() -> None:
    args = _parse_args()
    dirs = create_output_dirs(str(Path(__file__).parent))
    logger = setup_logging(str(dirs["logs"]))
    set_seed(args.seed)
    logger.info("Clinical Imputation Study v4 — START")

    # ── STEP 1: Data ─────────────────────────────────────────────────────────
    logger.info("[1/6] Loading dataset: %s", args.data)
    dset = load_and_classify(args.data)
    logger.info("  N=%d subjects, P=%d variables  (cont=%d, bin=%d, ord=%d)",
                dset.X.shape[0], dset.X.shape[1],
                len(dset.continuous_idx), len(dset.binary_idx), len(dset.ordinal_idx))

    audit = audit_data(dset.X, dset.col_names, dset.continuous_idx, dset.categorical_idx)
    logger.info("  Audit: %.1f%% non-Gaussian, %d quasi-constant variable(s)",
                audit["pct_non_gaussian"], audit["n_quasi_constant"])
    save_audit_report(audit, str(dirs["outputs"]))

    desc_stats = compute_descriptive_stats(dset.X, dset.col_names, dset.continuous_idx)
    desc_stats.to_csv(dirs["outputs"] / "descriptive_statistics.csv", index=False)

    # Identify hypertension column for clinical impact (column "HBP" or first binary)
    target_col = _find_target_col(dset.col_names, dset.binary_idx)

    # ── STEP 2: Monte Carlo simulation ────────────────────────────────────────
    logger.info("[2/6] Starting Monte Carlo simulation (M=%d × %d scenarios)",
                args.mc, len(DEFAULT_MECHANISMS) * len(DEFAULT_RATES))

    mechanisms = [m for m in DEFAULT_MECHANISMS if not (args.skip_mnar and m == "MNAR")]
    all_cont_records: List[dict] = []
    all_cat_records: List[dict] = []
    all_vr_records: List[dict] = []

    # For Rubin's rules: collect M imputed datasets per scenario × method
    rubin_bank: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list))

    # For clinical impact: collect one scenario (MCAR 20%)
    clinical_bank: Dict[str, dict] = {}
    roc_bank: Dict[str, dict] = {}

    for mechanism in mechanisms:
        ampute_fn = AMPUTATION_REGISTRY[mechanism]
        for rate in DEFAULT_RATES:
            scenario_key = f"{mechanism}_{int(rate*100)}pct"
            logger.info("  Scenario: %s  rate=%.0f%%", mechanism, rate * 100)

            for mc_iter in range(args.mc):
                seed_i = args.seed + mc_iter * 97  # deterministic per-iteration seed
                X_miss, mask = ampute_fn(dset.X, rate, seed=seed_i)

                for method_name, X_imp in _run_all_imputers(
                    X_miss, dset, args, seed_i
                ).items():
                    # Continuous metrics
                    cont = evaluate_all_continuous(
                        dset.X, X_imp, mask, dset.continuous_idx, dset.col_names)
                    for r in cont:
                        r.update({"mechanism": mechanism, "rate": rate,
                                  "method": method_name, "mc_iter": mc_iter})
                    all_cont_records.extend(cont)

                    # Categorical metrics
                    cat = evaluate_all_categorical(
                        dset.X, X_imp, mask, dset.categorical_idx, dset.col_names)
                    for r in cat:
                        r.update({"mechanism": mechanism, "rate": rate,
                                  "method": method_name, "mc_iter": mc_iter})
                    all_cat_records.extend(cat)

                    # Variance ratio
                    vr = variance_ratio(dset.X, X_imp, mask, dset.continuous_idx)
                    for entry in vr["per_variable"]:
                        entry.update({
                            "mechanism": mechanism, "rate": rate,
                            "method": method_name, "mc_iter": mc_iter,
                            "variable": dset.col_names[entry["var_idx"]],
                        })
                        all_vr_records.append(entry)

                    # Accumulate for Rubin's rules (M datasets same scenario)
                    rubin_bank[scenario_key][method_name].append(X_imp)

                    # Clinical impact for MCAR 20% (first MC iteration only)
                    if (mechanism == "MCAR" and rate == 0.20 and
                            mc_iter == 0 and target_col is not None):
                        predictor_cols = [j for j in dset.continuous_idx
                                          if j != target_col]
                        ci = clinical_model_impact(
                            dset.X, X_imp, target_col, predictor_cols,
                            col_names=dset.col_names, random_state=seed_i)
                        clinical_bank[method_name] = ci
                        # Build ROC data
                        y = dset.X[:, target_col].astype(int)
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.model_selection import cross_val_predict
                        lr = LogisticRegression(max_iter=1000, random_state=0)
                        Xp = X_imp[:, predictor_cols]
                        try:
                            prob = cross_val_predict(lr, Xp, y, cv=5,
                                                     method="predict_proba")[:, 1]
                            fpr, tpr, _ = roc_curve(y, prob)
                            from sklearn.metrics import roc_auc_score
                            roc_bank[method_name] = {
                                "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                                "auc": float(roc_auc_score(y, prob)),
                            }
                        except Exception:
                            pass

    # ── STEP 3: Rubin's rules ─────────────────────────────────────────────────
    logger.info("[3/6] Applying Rubin's combination rules")
    rubin_records: List[dict] = []
    for scenario_key, method_dict in rubin_bank.items():
        for method_name, datasets in method_dict.items():
            if len(datasets) < 2:
                continue
            rows = rubin_rules(datasets, dset.continuous_idx, dset.col_names)
            for r in rows:
                r["scenario"] = scenario_key
                r["method"] = method_name
            rubin_records.extend(rows)

    # ── STEP 4: Statistical analysis ──────────────────────────────────────────
    logger.info("[4/6] Statistical analysis")
    results_df = pd.DataFrame(all_cont_records)

    friedman_res, nemenyi_res, cd_val = {}, [], np.nan
    if not results_df.empty:
        # Build rank matrix: rows=variables, cols=methods
        pivot_rmse = (
            results_df
            .groupby(["variable", "method"])["RMSE"]
            .mean()
            .unstack()
        )[METHOD_NAMES].dropna()

        if len(pivot_rmse) >= 3:
            rank_matrix = compute_rankings(pivot_rmse.values, lower_is_better=True)
            friedman_res = friedman_test(rank_matrix)
            friedman_res["eta_sq"] = eta_squared_friedman(
                friedman_res["chi2"], friedman_res["N"], friedman_res["k"])
            avg_ranks = np.array(friedman_res["avg_ranks"])
            cd_val = nemenyi_cd(len(METHOD_NAMES), len(pivot_rmse))
            nemenyi_res = nemenyi_posthoc(avg_ranks, METHOD_NAMES, len(pivot_rmse))
            logger.info("  Friedman χ²=%.2f  p=%.2e  η²=%.3f",
                        friedman_res["chi2"], friedman_res["p_value"],
                        friedman_res["eta_sq"])

    robustness = robustness_summary(all_cont_records, "method", "RMSE")

    # ── STEP 5: Visualisations ────────────────────────────────────────────────
    logger.info("[5/6] Generating figures")
    fig_dir = str(dirs["figures"])
    supp_dir = str(dirs["figures_supp"])

    if not results_df.empty:
        fig_rmse_curves(results_df, fig_dir)
        fig_per_variable_rmse_heatmap(results_df, fig_dir)
        fig_rmse_ratio(results_df, fig_dir)

    vr_df = pd.DataFrame(all_vr_records)
    if not vr_df.empty:
        # Aggregate per-variable variance ratios into scenario-level means
        # Use R_V_global as the main metric (backward compatible with v2/v3)
        vr_agg = vr_df.groupby(["mechanism", "rate", "method", "mc_iter"]).agg({
            "R_V_global": "mean"
        }).reset_index()
        vr_agg = vr_agg.rename(columns={"R_V_global": "mean_R_V"})
        # Average across MC iterations
        vr_final = vr_agg.groupby(["mechanism", "rate", "method"]).agg({
            "mean_R_V": "mean"
        }).reset_index()
        fig_variance_ratio_curves(vr_final, supp_dir)

    if not results_df.empty:
        fig_boxplot_rmse(results_df, supp_dir)

    if roc_bank:
        fig_roc_curves(roc_bank, supp_dir)

    if clinical_bank:
        calib_data = {m: v.get("calibration", [])
                      for m, v in clinical_bank.items() if "calibration" in v}
        if calib_data:
            fig_calibration(calib_data, supp_dir)

    if friedman_res and nemenyi_res:
        avg_ranks_dict = dict(zip(METHOD_NAMES, friedman_res["avg_ranks"]))
        fig_cd_diagram(avg_ranks_dict, cd_val,
                       friedman_res["N"], 0.05, nemenyi_res, supp_dir)

    # ── STEP 6: Reporting ─────────────────────────────────────────────────────
    logger.info("[6/6] Saving outputs")
    out_dir = str(dirs["outputs"])

    save_all_metrics(
        all_cont_records, all_cat_records, all_vr_records, rubin_records, out_dir)
    save_summary_table(results_df, out_dir)
    save_robustness_report(robustness, friedman_res, out_dir)

    make_summary_latex(results_df, out_dir=out_dir)
    make_summary_markdown(results_df, out_dir=out_dir)

    logger.info("Clinical Imputation Study v4 — COMPLETE")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_all_imputers(
    X_miss: np.ndarray,
    dset,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Run all four imputation methods and return a dict {method_name: X_imp}."""
    results = {}

    results["Median/Mode"] = impute_median_mode(
        X_miss, dset.continuous_idx, dset.categorical_idx)

    results["KNN (k=5)"] = impute_knn(X_miss, k=args.knn_k)

    results["MICE (PMM)"] = impute_mice(
        X_miss,
        continuous_idx=dset.continuous_idx,
        binary_idx=dset.binary_idx,
        ordinal_idx=dset.ordinal_idx,
        max_iter=args.mice_iter,
        random_state=seed,
    )

    results["MissForest"] = impute_missforest(
        X_miss,
        continuous_idx=dset.continuous_idx,
        categorical_idx=dset.categorical_idx,
        n_estimators=args.mf_trees,
        max_iter=args.mf_iter,
        random_state=seed,
    )

    return results


def _find_target_col(col_names: List[str], binary_idx: List[int]) -> int | None:
    """Return the index of the hypertension (HBP/HTA) column, or first binary."""
    for keyword in ("HBP", "HTA", "hypertension", "Hypertension"):
        for j in binary_idx:
            if keyword.lower() in col_names[j].lower():
                return j
    return binary_idx[0] if binary_idx else None


if __name__ == "__main__":
    main()
