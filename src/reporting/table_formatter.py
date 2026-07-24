# -*- coding: utf-8 -*-
"""
table_formatter.py — Publication-ready table generation (LaTeX / Markdown)
===========================================================================
Generates formatted tables suitable for direct inclusion in a scientific
manuscript (LaTeX) or a GitHub README / Jupyter notebook (Markdown).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def format_mean_sd(mean: float, sd: float, decimals: int = 2) -> str:
    """Format a mean ± SD pair as a string.

    Args:
        mean:     Mean value.
        sd:       Standard deviation.
        decimals: Number of decimal places.

    Returns:
        String ``"mean ± sd"`` or ``"—"`` for NaN values.
    """
    if np.isnan(mean) or np.isnan(sd):
        return "—"
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(sd)}"


def make_summary_latex(
    results_df: pd.DataFrame,
    caption: str = "Mean RMSE (±SD) by imputation method, mechanism, and missing rate.",
    label: str = "tab:rmse_summary",
    out_dir: Optional[str] = None,
    filename: str = "Table1_RMSE_Summary.tex",
) -> str:
    """Generate a LaTeX ``booktabs`` table of mean RMSE ± SD.

    Args:
        results_df: DataFrame with columns ``mechanism``, ``rate``,
                    ``method``, ``RMSE``.
        caption:    LaTeX table caption.
        label:      LaTeX cross-reference label.
        out_dir:    If provided, write the ``.tex`` file there.
        filename:   Output filename.

    Returns:
        LaTeX table as a string.
    """
    summary = (
        results_df
        .groupby(["mechanism", "rate", "method"])["RMSE"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["cell"] = summary.apply(
        lambda r: format_mean_sd(r["mean"], r["std"]), axis=1
    )
    pivot = summary.pivot_table(
        index=["mechanism", "rate"], columns="method", values="cell", aggfunc="first"
    )

    # Ensure column order matches METHODS
    from visualization.style import METHODS
    cols_present = [m for m in METHODS if m in pivot.columns]
    pivot = pivot[cols_present]

    n_cols = len(cols_present)
    col_spec = "llll" + "r" * n_cols

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Mechanism & Rate & " + " & ".join(cols_present) + " \\\\",
        "\\midrule",
    ]

    current_mech = None
    for (mech, rate), row in pivot.iterrows():
        if mech != current_mech:
            if current_mech is not None:
                lines.append("\\midrule")
            current_mech = mech
        rate_str = f"{int(rate * 100)}\\%"
        cells = " & ".join(str(row.get(m, "—")) for m in cols_present)
        lines.append(f"{mech} & {rate_str} & {cells} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    tex = "\n".join(lines)

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / filename).write_text(tex, encoding="utf-8")

    return tex


def make_summary_markdown(
    results_df: pd.DataFrame,
    out_dir: Optional[str] = None,
    filename: str = "Table1_RMSE_Summary.md",
) -> str:
    """Generate a Markdown table of mean RMSE ± SD.

    Args:
        results_df: DataFrame with columns ``mechanism``, ``rate``,
                    ``method``, ``RMSE``.
        out_dir:    If provided, write the ``.md`` file there.
        filename:   Output filename.

    Returns:
        Markdown table as a string.
    """
    from visualization.style import METHODS

    summary = (
        results_df
        .groupby(["mechanism", "rate", "method"])["RMSE"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["cell"] = summary.apply(
        lambda r: format_mean_sd(r["mean"], r["std"]), axis=1
    )
    pivot = summary.pivot_table(
        index=["mechanism", "rate"], columns="method", values="cell", aggfunc="first"
    )
    cols_present = [m for m in METHODS if m in pivot.columns]
    pivot = pivot[cols_present]

    header = "| Mechanism | Rate | " + " | ".join(cols_present) + " |"
    sep = "|---|---|" + "|---|" * len(cols_present)
    rows = [header, sep]

    for (mech, rate), row in pivot.iterrows():
        rate_str = f"{int(rate * 100)}%"
        cells = " | ".join(str(row.get(m, "—")) for m in cols_present)
        rows.append(f"| {mech} | {rate_str} | {cells} |")

    md = "\n".join(rows)

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / filename).write_text(md, encoding="utf-8")

    return md
