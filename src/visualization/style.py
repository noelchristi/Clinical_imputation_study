# -*- coding: utf-8 -*-
"""
style.py — Shared visual style configuration
=============================================
Defines the colour palette, method ordering, DPI, and utility
functions shared by all figure modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Constants ─────────────────────────────────────────────────────────────────
DPI: int = 600   # publication requirement

METHODS: list = [
    "Median/Mode",
    "KNN (k=5)",
    "MICE (PMM)",
    "MissForest",
]

COLORS: dict = {
    "Median/Mode": "#4C72B0",
    "KNN (k=5)": "#55A868",
    "MICE (PMM)": "#C44E52",
    "MissForest": "#DD8452",
}

_RCPARAMS: dict = {
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,        # screen preview; actual save is at DPI=600
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
}


def setup_style() -> None:
    """Apply publication-ready matplotlib style globally."""
    sns.set_style("whitegrid")
    plt.rcParams.update(_RCPARAMS)


def save_figure(fig: plt.Figure, filename: str, directory: str) -> None:
    """Save a figure to ``directory/filename`` at 600 DPI with tight layout.

    Creates the directory if it does not exist.

    Args:
        fig:       Matplotlib Figure object.
        filename:  Output filename (e.g. ``"Fig1_Missing_Pattern.png"``).
        directory: Target directory path.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(directory) / filename, dpi=DPI)
    plt.close(fig)
