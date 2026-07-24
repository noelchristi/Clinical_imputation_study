# -*- coding: utf-8 -*-
"""
json_exporter.py — JSON export of audit reports and robustness results
======================================================================
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("CISv4")


class _NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle numpy scalar types in JSON serialisation."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def save_json(data: Any, filename: str, out_dir: str, indent: int = 2) -> Path:
    """Serialise ``data`` to a JSON file.

    Args:
        data:     Any JSON-serialisable object (dict, list, etc.).
        filename: Output filename.
        out_dir:  Output directory (created if absent).
        indent:   JSON indentation width.

    Returns:
        Path to written file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=_NumpyEncoder, ensure_ascii=False)
    logger.info("  [export] %s", out_path.name)
    return out_path


def save_audit_report(audit: dict, out_dir: str) -> Path:
    """Save the pre-simulation data quality audit to JSON.

    Args:
        audit:   Dict returned by :func:`audit_data`.
        out_dir: Output directory.

    Returns:
        Path to written file.
    """
    audit_with_meta = {
        "generated_at": datetime.now().isoformat(),
        "audit": audit,
    }
    return save_json(audit_with_meta, "audit_report.json", out_dir)


def save_robustness_report(robustness: dict, friedman: dict, out_dir: str) -> Path:
    """Save robustness summary and Friedman test results to JSON.

    Args:
        robustness: Dict returned by :func:`robustness_summary`.
        friedman:   Dict returned by :func:`friedman_test`.
        out_dir:    Output directory.

    Returns:
        Path to written file.
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "robustness_summary": robustness,
        "friedman_test": friedman,
    }
    return save_json(report, "robustness.json", out_dir)
