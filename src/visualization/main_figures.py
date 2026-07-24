# -*- coding: utf-8 -*-
"""
main_figures.py — Publication-ready main article figures (Fig 1–6)
===================================================================
All figures are saved at 600 DPI with tight layout.

Figures:
  Fig 1 — Missing data pattern heatmap (MCAR 20%, first 50 patients)
  Fig 2 — Study flowchart (methodology overview)
  Fig 3 — Mean RMSE by mechanism and missing rate (3 panels)
  Fig 4 — Per-variable RMSE heatmap (MCAR 40%, log scale)
  Fig 5 — RMSE ratio barplot (all methods vs MissForest, MCAR 40%)
  Fig 6 — Evidence-based decision algorithm flowchart
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .style import COLORS, METHODS, DPI, setup_style, save_figure


def fig_missing_pattern(
    X_miss: np.ndarray,
    col_names: List[str],
    out_dir: str,
    n_patients: int = 50,
    mechanism: str = "MCAR 20%",
    filename: str = "Fig1_Missing_Pattern.png",
) -> None:
    """Fig 1: Heatmap of missing data pattern for the first N patients.

    Args:
        X_miss:     Incomplete matrix ``(N, P)`` with NaN at missing cells.
        col_names:  Variable names.
        out_dir:    Output directory.
        n_patients: Number of patients to display (first n).
        mechanism:  Mechanism label for the title.
        filename:   Output filename.
    """
    setup_style()
    sub = X_miss[:n_patients, :]
    miss_matrix = np.isnan(sub).astype(float)  # 1 = missing, 0 = observed

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(
        miss_matrix.T,
        ax=ax,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        vmin=0, vmax=1,
        cbar_kws={"label": "Missing / Observed", "shrink": 0.6},
        linewidths=0.3, linecolor="white",
        xticklabels=5,
        yticklabels=col_names,
    )
    ax.set_title(f"Missing Data Pattern — {mechanism} (first {n_patients} patients)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(f"Patient index (first {n_patients})", fontsize=11)
    ax.set_ylabel("Variable", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    save_figure(fig, filename, out_dir)


def fig_rmse_curves(
    results_df: pd.DataFrame,
    out_dir: str,
    filename: str = "Fig3_RMSE_Curves.png",
) -> None:
    """Fig 3: Mean RMSE by mechanism and missing rate (3-panel figure).

    Args:
        results_df: DataFrame with columns:
                    ``mechanism``, ``rate``, ``method``, ``RMSE``.
                    Only continuous variables should be included.
        out_dir:    Output directory.
        filename:   Output filename.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax_i, mech in enumerate(["MCAR", "MAR", "MNAR"]):
        ax = axes[ax_i]
        sub = results_df[results_df["mechanism"] == mech]
        pivot = sub.groupby(["rate", "method"])["RMSE"].mean().reset_index()

        for method in METHODS:
            md = pivot[pivot["method"] == method]
            if md.empty:
                continue
            md_sorted = md.sort_values("rate")
            # Annotate extreme MNAR values
            y_vals = md_sorted["RMSE"].values
            if mech == "MNAR" and method == "MICE (PMM)" and y_vals.max() > 150:
                ax.annotate(
                    f"MICE: {y_vals.max():.1f}",
                    xy=(md_sorted["rate"].values[-1], min(y_vals.max(), 150)),
                    fontsize=8, color=COLORS[method], fontweight="bold",
                    ha="center",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                y_vals = np.clip(y_vals, 0, 150)

            ax.plot(
                md_sorted["rate"].values, y_vals,
                "o-", lw=2.5, ms=8,
                color=COLORS[method], label=method,
            )

        ax.set_title(mech, fontsize=13, fontweight="bold")
        ax.set_xlabel("Missing Rate", fontsize=11)
        ax.set_ylabel("Mean RMSE", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Mean RMSE by Missingness Mechanism and Missing Rate",
        fontsize=15, fontweight="bold",
    )
    save_figure(fig, filename, out_dir)


def fig_per_variable_rmse_heatmap(
    results_df: pd.DataFrame,
    out_dir: str,
    mechanism: str = "MCAR",
    rate: float = 0.40,
    filename: str = "Fig4_PerVariable_RMSE_Heatmap.png",
) -> None:
    """Fig 4: Per-variable RMSE heatmap (log scale) for one scenario.

    Args:
        results_df: DataFrame with columns ``mechanism``, ``rate``,
                    ``method``, ``variable``, ``RMSE``.
        out_dir:    Output directory.
        mechanism:  Mechanism to plot (default "MCAR").
        rate:       Missing rate to plot (default 0.40).
        filename:   Output filename.
    """
    setup_style()
    sub = results_df[
        (results_df["mechanism"] == mechanism) &
        (results_df["rate"] == rate)
    ]
    pivot = sub.groupby(["variable", "method"])["RMSE"].mean().unstack()
    pivot = pivot[METHODS].dropna()

    # Sort by MissForest RMSE descending
    if "MissForest" in pivot.columns:
        pivot = pivot.sort_values("MissForest", ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.35)))

    # Log-scale heatmap — add small offset to avoid log(0)
    log_data = np.log10(pivot.values + 1e-3)
    im = ax.imshow(log_data, aspect="auto", cmap="YlOrRd")

    # Annotations
    for i in range(log_data.shape[0]):
        for j in range(log_data.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax.set_xlabel("Imputation Method", fontsize=11)
    ax.set_ylabel("Variable", fontsize=11)
    ax.set_title(
        f"Per-Variable RMSE by Imputation Method — {mechanism} {int(rate*100)}%",
        fontsize=13, fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("RMSE (log scale)", fontsize=9)
    save_figure(fig, filename, out_dir)


def fig_rmse_ratio(
    results_df: pd.DataFrame,
    out_dir: str,
    reference_method: str = "MissForest",
    mechanism: str = "MCAR",
    rate: float = 0.40,
    filename: str = "Fig5_RMSE_Ratio.png",
) -> None:
    """Fig 5: RMSE ratio relative to reference method (horizontal barplot).

    Args:
        results_df:       Full results DataFrame.
        out_dir:          Output directory.
        reference_method: Method used as denominator (default MissForest).
        mechanism:        Scenario mechanism.
        rate:             Scenario missing rate.
        filename:         Output filename.
    """
    setup_style()
    sub = results_df[
        (results_df["mechanism"] == mechanism) &
        (results_df["rate"] == rate)
    ]
    pivot = sub.groupby(["variable", "method"])["RMSE"].mean().unstack()
    pivot = pivot[METHODS].dropna()

    if reference_method not in pivot.columns:
        return

    ref_col = pivot[reference_method]
    ratio_df = pivot.copy()
    for m in METHODS:
        ratio_df[m] = pivot[m] / ref_col.replace(0, np.nan)

    ratio_df = ratio_df.sort_values(reference_method, ascending=True)
    other_methods = [m for m in METHODS if m != reference_method]

    fig, ax = plt.subplots(figsize=(12, max(8, len(ratio_df) * 0.4)))
    y = np.arange(len(ratio_df))
    hatches = ["///", "...", "xxx"]
    bar_h = 0.22

    for i, (method, hatch) in enumerate(zip(other_methods, hatches)):
        ax.barh(
            y + i * bar_h,
            ratio_df[method].values,
            bar_h,
            color=COLORS[method], alpha=0.8,
            edgecolor="black", lw=0.5,
            hatch=hatch, label=method,
        )

    ax.axvline(1.0, color=COLORS[reference_method], lw=2.5, ls="--",
               label=f"{reference_method} (reference)")
    ax.set_yticks(y + bar_h)
    ax.set_yticklabels(ratio_df.index.tolist(), fontsize=8)
    ax.set_xlabel(f"RMSE Ratio (RMSE / RMSE {reference_method})", fontsize=11)
    ax.set_ylabel("Variable", fontsize=11)
    ax.set_title(
        f"Imputation Performance Relative to {reference_method} — "
        f"{mechanism} {int(rate*100)}%",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    save_figure(fig, filename, out_dir)
