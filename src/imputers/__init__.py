# -*- coding: utf-8 -*-
"""
imputers — Public API for imputation methods
=============================================
Re-exports all four imputation functions and the method registry.

IMPORTANT methodological notes:
  - ``impute_mice`` uses Predictive Mean Matching (PMM) for continuous
    variables and logistic regression for categorical, matching the R
    ``mice`` package specification (van Buuren & Groothuis-Oudshoorn 2011).
  - ``impute_missforest`` is the true Stekhoven & Buehlmann (2012) algorithm
    with separate RandomForestClassifier / RandomForestRegressor per variable
    type.  It is NOT IterativeImputer(RandomForestRegressor).
"""

from .baseline import impute_median_mode
from .knn_imputer import impute_knn
from .mice_imputer import impute_mice
from .missforest_imputer import impute_missforest

IMPUTER_REGISTRY: dict = {
    "Median/Mode": impute_median_mode,
    "KNN": impute_knn,
    "MICE (PMM)": impute_mice,
    "MissForest": impute_missforest,
}

__all__ = [
    "impute_median_mode",
    "impute_knn",
    "impute_mice",
    "impute_missforest",
    "IMPUTER_REGISTRY",
]