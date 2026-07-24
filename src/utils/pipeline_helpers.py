# -*- coding: utf-8 -*-
"""
pipeline_helpers.py — Setup utilities for Clinical Imputation Study v4
=======================================================================
Provides logging configuration, reproducible seed management,
output directory creation, and a timing decorator.

No scientific content here; purely operational infrastructure.
"""

import logging
import sys
import time
import functools
from pathlib import Path
from typing import Callable, Any

import numpy as np


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure a named logger writing to both file and stdout.

    Args:
        log_dir: Directory where ``run.log`` will be created.
        level:   Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger` instance named ``"CISv4"``.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("CISv4")
    logger.setLevel(level)

    # Avoid duplicate handlers when function is called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # File handler
    fh = logging.FileHandler(Path(log_dir) / "run.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for full reproducibility.

    Sets numpy random state. Called once at pipeline start.

    Args:
        seed: Non-negative integer seed value.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"Random seed must be non-negative; got {seed}.")
    np.random.seed(seed)


# ── Output directories ────────────────────────────────────────────────────────

def create_output_dirs(base: str = ".") -> dict:
    """Create all required output directories if they do not exist.

    Args:
        base: Root directory relative to which sub-dirs are created.

    Returns:
        dict mapping logical name to resolved :class:`pathlib.Path`.
    """
    dirs = {
        "outputs": Path(base) / "outputs",
        "figures": Path(base) / "figures",
        "figures_supp": Path(base) / "figures" / "supplementary",
        "logs": Path(base) / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── Timing decorator ─────────────────────────────────────────────────────────

def timed(func: Callable) -> Callable:
    """Decorator that logs wall-clock execution time of any function.

    Uses ``logging.getLogger("CISv4")`` so logs appear in the main file.

    Args:
        func: Any callable to wrap.

    Returns:
        Wrapped callable that logs elapsed time on completion.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger("CISv4")
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("  [timer] %s completed in %.2f s", func.__name__, elapsed)
        return result
    return wrapper
