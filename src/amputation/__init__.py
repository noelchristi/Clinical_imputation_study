# -*- coding: utf-8 -*-
"""
amputation — Public API for missing-data generation
=====================================================
Re-exports the three amputation functions and the mechanism registry.
"""

from .mcar import ampute_mcar
from .mar import ampute_mar
from .mnar import ampute_mnar

AMPUTATION_REGISTRY: dict = {
    "MCAR": ampute_mcar,
    "MAR": ampute_mar,
    "MNAR": ampute_mnar,
}

__all__ = ["ampute_mcar", "ampute_mar", "ampute_mnar", "AMPUTATION_REGISTRY"]