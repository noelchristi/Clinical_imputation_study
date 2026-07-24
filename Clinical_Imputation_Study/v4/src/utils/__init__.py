# -*- coding: utf-8 -*-
"""
utils — Public API for the utilities package
=============================================
Re-exports all public symbols from submodules so that callers can write::

    from utils import setup_logging, load_and_classify, audit_data
"""

from .pipeline_helpers import setup_logging, set_seed, create_output_dirs, timed
from .data_loader import load_and_classify, DatasetInfo, VarMeta
from .preprocessing import (
    compute_descriptive_stats,
    detect_quasi_constant,
    audit_data,
)

__all__ = [
    "setup_logging", "set_seed", "create_output_dirs", "timed",
    "load_and_classify", "DatasetInfo", "VarMeta",
    "compute_descriptive_stats", "detect_quasi_constant", "audit_data",
]