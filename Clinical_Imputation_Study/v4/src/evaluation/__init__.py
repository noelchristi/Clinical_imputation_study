# -*- coding: utf-8 -*-
"""
evaluation — Public API for all evaluation metrics
===================================================
"""

from .continuous_metrics import evaluate_continuous_variable, evaluate_all_continuous
from .categorical_metrics import evaluate_categorical_variable, evaluate_all_categorical
from .variance_correlation import (
    variance_ratio,
    correlation_differences,
    mean_absolute_correlation_error,
)
from .clinical_impact import clinical_model_impact
from .sensitivity import (
    sensitivity_analysis,
    bootstrap_ci_rmse,
    robustness_summary,
)

__all__ = [
    "evaluate_continuous_variable", "evaluate_all_continuous",
    "evaluate_categorical_variable", "evaluate_all_categorical",
    "variance_ratio", "correlation_differences", "mean_absolute_correlation_error",
    "clinical_model_impact",
    "sensitivity_analysis", "bootstrap_ci_rmse", "robustness_summary",
]