# -*- coding: utf-8 -*-
"""
stats_utils — Public API for statistical comparison utilities
=============================================================
"""

from .nonparametric import (
    friedman_test,
    nemenyi_posthoc,
    nemenyi_cd,
    compute_rankings,
)
from .rubin import rubin_rules
from .effect_sizes import (
    cohens_d,
    cliffs_delta,
    cliffs_delta_magnitude,
    eta_squared_kruskal,
    eta_squared_friedman,
)

__all__ = [
    "friedman_test", "nemenyi_posthoc", "nemenyi_cd", "compute_rankings",
    "rubin_rules",
    "cohens_d", "cliffs_delta", "cliffs_delta_magnitude",
    "eta_squared_kruskal", "eta_squared_friedman",
]