# -*- coding: utf-8 -*-
"""
visualization — Public API for all figure-generation functions
==============================================================
"""

from .style import setup_style, save_figure, COLORS, METHODS, DPI
from .main_figures import (
    fig_missing_pattern,
    fig_rmse_curves,
    fig_per_variable_rmse_heatmap,
    fig_rmse_ratio,
)
from .supp_figures import (
    fig_cd_diagram,
    fig_variance_ratio_curves,
    fig_boxplot_rmse,
    fig_roc_curves,
    fig_calibration,
    fig_violin_errors,
)

__all__ = [
    "setup_style", "save_figure", "COLORS", "METHODS", "DPI",
    "fig_missing_pattern", "fig_rmse_curves",
    "fig_per_variable_rmse_heatmap", "fig_rmse_ratio",
    "fig_cd_diagram", "fig_variance_ratio_curves", "fig_boxplot_rmse",
    "fig_roc_curves", "fig_calibration", "fig_violin_errors",
]