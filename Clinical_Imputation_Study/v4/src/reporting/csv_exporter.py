# -*- coding: utf-8 -*-
"""
csv_exporter.py — Automatic CSV export of all simulation results
=================================================================
All outputs are written to the configured ``outputs/`` directory with
consistent naming conventions and ISO-8601 timestamps.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger("CISv4")


def save_results_csv(
    records: List[Dict[str, Any]],
    filename: str,
    out_dir: str,
    index: bool = False,
) -> Path:
    """Save a list of result records to a CSV file.

    Args:
        records:  List of dicts (each row of the output table).
        filename: Output filename (e.g. ``"continuous_metrics.csv"``).
        out_dir:  Output directory.
        index:    Whether to write the DataFrame index.

    Returns:
        Path to the written file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    out_path = Path(out_dir) / filename
    df.to_csv(out_path, index=index, encoding="utf-8")
    logger.info("  [export] %s  (%d rows × %d cols)", out_path.name,
                len(df), len(df.columns))
    return out_path


def save_all_metrics(
    continuous_records: List[dict],
    categorical_records: List[dict],
    variance_records: List[dict],
    rubin_records: List[dict],
    out_dir: str,
    timestamp: bool = True,
) -> Dict[str, Path]:
    """Save all metric tables to CSV in a single call.

    Args:
        continuous_records:  From evaluate_all_continuous.
        categorical_records: From evaluate_all_categorical.
        variance_records:    From variance_ratio (per_variable list).
        rubin_records:       From rubin_rules.
        out_dir:             Output directory.
        timestamp:           If True, prefix filenames with YYYYMMDD.

    Returns:
        dict mapping logical name → file path.
    """
    prefix = datetime.now().strftime("%Y%m%d_") if timestamp else ""
    written: Dict[str, Path] = {}

    if continuous_records:
        written["continuous"] = save_results_csv(
            continuous_records, f"{prefix}continuous_metrics.csv", out_dir)

    if categorical_records:
        written["categorical"] = save_results_csv(
            categorical_records, f"{prefix}categorical_metrics.csv", out_dir)

    if variance_records:
        written["variance"] = save_results_csv(
            variance_records, f"{prefix}variance_ratio.csv", out_dir)

    if rubin_records:
        written["rubin"] = save_results_csv(
            rubin_records, f"{prefix}rubin_rules.csv", out_dir)

    return written


def save_summary_table(
    results_df: pd.DataFrame,
    out_dir: str,
    filename: str = "summary_table.csv",
) -> Path:
    """Save the pivoted mean-RMSE summary table.

    Args:
        results_df: Full results DataFrame with columns
                    ``mechanism``, ``rate``, ``method``, ``RMSE``.
        out_dir:    Output directory.
        filename:   Output filename.

    Returns:
        Path to written file.
    """
    summary = (
        results_df
        .groupby(["mechanism", "rate", "method"])["RMSE"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "Mean_RMSE", "std": "SD_RMSE", "count": "N"})
    )
    summary = summary.round(4)
    return save_results_csv(summary.to_dict("records"), filename, out_dir)
