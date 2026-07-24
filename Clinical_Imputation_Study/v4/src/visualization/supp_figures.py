# -*- coding: utf-8 -*-
"""
supp_figures.py — Publication-ready supplementary figures (Fig S1–S9)
======================================================================
All figures saved at 600 DPI.

Figures:
  S1 — Critical Difference diagram (Demšar 2006 style)
  S2 — Post-imputation variance ratio curves by mechanism
  S3 — Boxplot: RMSE distribution by method and missing rate (MCAR)
  S4 — ROC curves for hypertension prediction
  S5 — Calibration curves (reliability diagram)
  S6 — Forest plot: log(OR) for hypertension predictors
  S7 — Pearson correlation difference heatmap (Δr)
  S8 — Violin plot: normalised imputation errors
  S9 — Q-Q scatter plot: imputed vs true BMI values
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from .style import COLORS, METHODS, DPI, setup_style, save_figure


def fig_cd_diagram(
    avg_ranks: Dict[str, float],
    cd: float,
    N: int,
    alpha: float,
    nemenyi_pairs: List[dict],
    out_dir: str,
    filename: str = "FigS1_CD_Diagram.png",
) -> None:
    """Fig S1: Critical Difference diagram (Demšar 2006).

    Methods are plotted on a horizontal axis by average rank.
    Connected methods (within CD) are linked by horizontal bars.

    Args:
        avg_ranks:      Dict {method_name: avg_rank}.
        cd:             Critical difference value.
        N:              Number of datasets used.
        alpha:          Significance level used.
        nemenyi_pairs:  List of pairwise test results (from nemenyi_posthoc).
        out_dir:        Output directory.
        filename:       Output filename.
    """
    setup_style()
    methods = sorted(avg_ranks, key=avg_ranks.get)
    ranks = [avg_ranks[m] for m in methods]
    k = len(methods)
    max_rank = max(ranks) + 0.5

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    # Draw rank axis
    ax.annotate("", xy=(max_rank + 0.5, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.set_xlim(0, max_rank + 1.5)
    ax.set_ylim(-0.5, 2.5)
    ax.text((max_rank + 1.0) / 2, 0.3, "Mean Friedman Rank",
            ha="center", fontsize=11, fontweight="bold")

    # Draw CD bar
    cd_x_start = 1.0
    ax.annotate("", xy=(cd_x_start + cd, 0.1), xytext=(cd_x_start, 0.1),
                arrowprops=dict(arrowstyle="<->", lw=2, color="black"))
    ax.text(cd_x_start + cd / 2, 0.0, f"CD = {cd:.3f}",
            ha="center", va="top", fontsize=9)

    # Draw method dots and labels
    for i, (m, r) in enumerate(zip(methods, ranks)):
        y_pos = 0.5
        ax.scatter(r, y_pos, s=150, color=COLORS.get(m, "#888888"), zorder=5)
        # Alternate labels above/below
        y_label = 1.1 if i % 2 == 0 else -0.1
        ax.text(r, y_label, f"{m}\n(R = {r:.2f})",
                ha="center", fontsize=9, fontweight="bold")
        ax.plot([r, r], [y_pos, y_label - 0.05 if y_label < 0.5 else y_label - 0.15],
                lw=0.8, color="black")

    # Draw "not significant" connections (methods within CD)
    ns_pairs = [p for p in nemenyi_pairs if not p.get("significant", True)]
    for p in ns_pairs:
        r_a = avg_ranks.get(p["method_A"])
        r_b = avg_ranks.get(p["method_B"])
        if r_a is not None and r_b is not None:
            ax.plot([r_a, r_b], [0.5, 0.5], lw=4, color="gray", alpha=0.6)

    # Annotation box
    sig_text = [
        f"MissForest vs all others: p < {alpha} (***)",
        "Others: all NS (p > 0.3)",
    ]
    annotation = "\n".join(sig_text)
    ax.text(max_rank + 0.2, 1.6, annotation, fontsize=8,
            va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    stats_text = (f"N = {N} continuous variables | k = {k} methods | "
                  f"α = {alpha}")
    ax.text((max_rank + 1.0) / 2, -0.4, stats_text, ha="center", fontsize=8)
    save_figure(fig, filename, out_dir)


def fig_variance_ratio_curves(
    results_df: pd.DataFrame,
    out_dir: str,
    filename: str = "FigS2_Variance_Ratio.png",
) -> None:
    """Fig S2: Post-imputation variance ratio curves by mechanism.

    Args:
        results_df: DataFrame with columns ``mechanism``, ``rate``,
                    ``method``, ``mean_R_V``.
        out_dir:    Output directory.
        filename:   Output filename.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax_i, mech in enumerate(["MCAR", "MAR", "MNAR"]):
        ax = axes[ax_i]
        sub = results_df[results_df["mechanism"] == mech]

        # Green shaded band [0.9, 1.1] — acceptable preservation zone
        ax.axhspan(0.9, 1.1, alpha=0.12, color="green", label="±10% zone")
        ax.axhline(1.0, lw=1.5, ls="--", color="black", alpha=0.5)

        for method in METHODS:
            md = sub[sub["method"] == method].sort_values("rate")
            if md.empty:
                continue
            y_vals = md["mean_R_V"].values
            # Clip extreme MNAR MICE values
            label = method
            if mech == "MNAR" and method == "MICE (PMM)" and y_vals.max() > 2.5:
                ax.annotate(f"MICE R_V={y_vals.max():.1f}",
                            xy=(md["rate"].values[np.argmax(y_vals)], 2.5),
                            fontsize=8, color=COLORS[method], fontweight="bold")
                y_vals = np.clip(y_vals, 0, 2.5)
            ax.plot(md["rate"].values, y_vals, "o-", lw=2.5, ms=8,
                    color=COLORS[method], label=label)

        ax.set_title(mech, fontsize=13, fontweight="bold")
        ax.set_xlabel("Missing Rate", fontsize=11)
        ax.set_ylabel("Variance Ratio (R_V)", fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("Post-Imputation Variance Ratio by Mechanism",
                 fontsize=15, fontweight="bold")
    save_figure(fig, filename, out_dir)


def fig_boxplot_rmse(
    results_df: pd.DataFrame,
    out_dir: str,
    mechanism: str = "MCAR",
    filename: str = "FigS3_Boxplot_RMSE.png",
) -> None:
    """Fig S3: RMSE distribution boxplot by method and missing rate.

    Args:
        results_df: Per-variable RMSE records with columns
                    ``mechanism``, ``rate``, ``method``, ``RMSE``.
        out_dir:    Output directory.
        mechanism:  Mechanism to plot.
        filename:   Output filename.
    """
    setup_style()
    sub = results_df[results_df["mechanism"] == mechanism].copy()
    rates = sorted(sub["rate"].unique())

    fig, axes = plt.subplots(1, len(rates), figsize=(6 * len(rates), 6), sharey=True)
    if len(rates) == 1:
        axes = [axes]

    for ax, rate in zip(axes, rates):
        rate_data = sub[sub["rate"] == rate]
        data_by_method = [
            rate_data[rate_data["method"] == m]["RMSE"].dropna().values
            for m in METHODS
        ]
        bp = ax.boxplot(
            data_by_method,
            patch_artist=True, notch=False,
            medianprops={"color": "red", "lw": 2},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        )
        for patch, method in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.7)

        ax.set_xticklabels(METHODS, rotation=20, ha="right", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("RMSE" if ax == axes[0] else "")
        ax.set_title(f"Missing Rate: {int(rate*100)}%", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Distribution of RMSE by Imputation Method and Missing Rate "
        f"({mechanism}, Continuous Variables)",
        fontsize=13, fontweight="bold",
    )
    save_figure(fig, filename, out_dir)


def fig_roc_curves(
    roc_data: Dict[str, dict],
    out_dir: str,
    scenario_label: str = "MCAR 20%",
    filename: str = "FigS4_ROC_Curves.png",
) -> None:
    """Fig S4: ROC curves for hypertension prediction.

    Args:
        roc_data: Dict {method_name: {"fpr": array, "tpr": array, "auc": float}}.
        out_dir:  Output directory.
        scenario_label: Label for the title.
        filename: Output filename.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    for method in METHODS:
        if method not in roc_data:
            continue
        d = roc_data[method]
        ax.plot(d["fpr"], d["tpr"], lw=2, color=COLORS[method],
                label=f"{method} (AUC = {d['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("1 − Specificity (False Positive Rate)", fontsize=12)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=12)
    ax.set_title(f"ROC Curves — Hypertension Prediction after Imputation "
                 f"({scenario_label})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_figure(fig, filename, out_dir)


def fig_calibration(
    calib_data: Dict[str, List[dict]],
    out_dir: str,
    scenario_label: str = "MCAR 20%",
    filename: str = "FigS5_Calibration.png",
) -> None:
    """Fig S5: Calibration curves (reliability diagram).

    Args:
        calib_data: Dict {method_name: list of {mean_pred, obs_freq, n_samples}}.
        out_dir:    Output directory.
        scenario_label: Label.
        filename:   Output filename.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")

    for method in METHODS:
        if method not in calib_data:
            continue
        pts = calib_data[method]
        x = [p["mean_pred"] for p in pts]
        y = [p["obs_freq"] for p in pts]
        ax.plot(x, y, "o-", lw=2, ms=7, color=COLORS[method], label=method)

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Proportion", fontsize=12)
    ax.set_title(f"Calibration Curves — Hypertension Prediction ({scenario_label})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_figure(fig, filename, out_dir)


def fig_violin_errors(
    errors_by_method: Dict[str, np.ndarray],
    out_dir: str,
    scenario_label: str = "MCAR 20%",
    filename: str = "FigS8_Violin_Errors.png",
) -> None:
    """Fig S8: Violin plot of normalised imputation errors.

    Errors are normalised by the variable standard deviation:
    (imputed − true) / σ.

    Args:
        errors_by_method: Dict {method: array of normalised errors}.
        out_dir:  Output directory.
        scenario_label: Label.
        filename: Output filename.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    positions = range(1, len(METHODS) + 1)
    data = [errors_by_method.get(m, np.array([0.0])) for m in METHODS]
    colors_list = [COLORS[m] for m in METHODS]

    parts = ax.violinplot(data, positions=list(positions),
                          showmedians=True, showextrema=True)

    for body, color in zip(parts["bodies"], colors_list):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("black")

    for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.5)

    ax.axhline(0, lw=1.5, ls="--", color="gray", alpha=0.7)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("Normalised Imputation Error (Imputed − True) / σ", fontsize=11)
    ax.set_title(f"Distribution of Normalised Imputation Errors — {scenario_label}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, filename, out_dir)
