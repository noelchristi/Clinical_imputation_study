# -*- coding: utf-8 -*-
"""
data_loader.py — Dataset ingestion and variable classification
==============================================================
Handles loading of the clinical CSV, sanitisation, and automatic
variable-type classification.  Classification follows the convention
used in the methodological literature on mixed-type imputation:

  * Continuous  : numeric with > CONTINUOUS_THRESHOLD unique values
  * Binary      : exactly 2 distinct non-NaN integer values
  * Ordinal     : 3–ORDINAL_MAX distinct non-NaN integer values
  * Nominal     : > ORDINAL_MAX categories (excluded from imputation by default)

Reference:
    Stekhoven & Bühlmann (2012) Bioinformatics 28(1):112–118 — variable
    classification for MissForest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ── Thresholds for variable classification ────────────────────────────────────
CONTINUOUS_THRESHOLD: int = 10   # > this  → continuous
ORDINAL_MAX: int = 10            # <= this  → ordinal (if integer)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class VarMeta:
    """Metadata record for a single dataset variable.

    Attributes:
        name:        Column name.
        var_type:    One of ``"continuous"``, ``"binary"``, ``"ordinal"``, ``"nominal"``.
        n_unique:    Number of distinct non-NaN values.
        n_missing:   Count of NaN cells (always 0 for complete reference data).
        pct_missing: Percentage of NaN cells.
        col_idx:     Integer position in the numpy array.
    """
    name: str
    var_type: str
    n_unique: int
    n_missing: int
    pct_missing: float
    col_idx: int


@dataclass
class DatasetInfo:
    """Container returned by :func:`load_and_classify`.

    Attributes:
        df:               Raw :class:`pandas.DataFrame` (complete, original scale).
        X:                Numpy array ``(N, P)`` with categorical columns integer-encoded.
        var_meta:         Ordered list of :class:`VarMeta`, one per column.
        continuous_idx:   Column indices of continuous variables.
        binary_idx:       Column indices of binary variables.
        ordinal_idx:      Column indices of ordinal variables.
        categorical_idx:  Union of binary and ordinal indices.
        col_names:        List of column names (same order as X).
    """
    df: pd.DataFrame
    X: np.ndarray
    var_meta: List[VarMeta]
    continuous_idx: List[int]
    binary_idx: List[int]
    ordinal_idx: List[int]
    categorical_idx: List[int] = field(init=False)
    col_names: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.categorical_idx = sorted(self.binary_idx + self.ordinal_idx)
        self.col_names = [m.name for m in self.var_meta]


# ── Public API ─────────────────────────────────────────────────────────────────

def load_and_classify(
    path: str,
    sep: str = ";",
    decimal: str = ",",
) -> DatasetInfo:
    """Load a clinical CSV and classify every variable automatically.

    Args:
        path:    Path to the CSV file.
        sep:     Column separator (default ``";"``).
        decimal: Decimal character (default ``","``).

    Returns:
        Fully populated :class:`DatasetInfo` instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError:        If the dataset is empty after loading.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p.resolve()}")

    df = pd.read_csv(p, sep=sep, decimal=decimal)
    df = df.replace([np.inf, -np.inf], np.nan)

    if df.empty:
        raise ValueError(f"Dataset loaded from {path} is empty.")

    # Encode categoricals to integers; keep continuous as float
    df_enc, encoders = _encode_categoricals(df)

    X = df_enc.to_numpy(dtype=float, copy=True)
    # Clean residual NaN (from inf→NaN conversion in ratio columns)
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            valid = col[~nan_mask]
            if len(valid) > 0:
                X[nan_mask, j] = np.nanmedian(valid)
            else:
                X[nan_mask, j] = 0.0
    meta: List[VarMeta] = []
    cont_idx, bin_idx, ord_idx = [], [], []

    for j, col in enumerate(df_enc.columns):
        series = df[col].dropna()
        n_unique = int(series.nunique())
        n_miss = int(df[col].isna().sum())
        pct_miss = round(100.0 * n_miss / len(df), 2)

        vtype = _classify_column(series, n_unique)

        meta.append(VarMeta(
            name=col, var_type=vtype,
            n_unique=n_unique, n_missing=n_miss,
            pct_missing=pct_miss, col_idx=j,
        ))

        if vtype == "continuous":
            cont_idx.append(j)
        elif vtype == "binary":
            bin_idx.append(j)
        elif vtype == "ordinal":
            ord_idx.append(j)

    return DatasetInfo(
        df=df, X=X, var_meta=meta,
        continuous_idx=cont_idx,
        binary_idx=bin_idx,
        ordinal_idx=ord_idx,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _classify_column(series: pd.Series, n_unique: int) -> str:
    """Return the variable type string for one column.

    Logic (in order):
      1. If all values are numeric and n_unique > CONTINUOUS_THRESHOLD → ``"continuous"``
      2. If all values are integer-like (no fractional part) and n_unique == 2 → ``"binary"``
      3. If all values are integer-like and 3 <= n_unique <= ORDINAL_MAX → ``"ordinal"``
      4. Otherwise → ``"nominal"``

    Args:
        series:   Non-null values of the column.
        n_unique: Number of distinct values.

    Returns:
        Variable type string.
    """
    if n_unique > CONTINUOUS_THRESHOLD:
        return "continuous"

    # Check integer-like (covers 0/1, 1/2/3, …)
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    is_int_like = (numeric % 1 == 0).all() if len(numeric) > 0 else False

    if is_int_like:
        if n_unique == 2:
            return "binary"
        if 3 <= n_unique <= ORDINAL_MAX:
            return "ordinal"

    return "nominal"


def _encode_categoricals(df: pd.DataFrame) -> tuple:
    """Integer-encode object/categorical columns in place.

    Numeric columns are left untouched.  Returns the encoded DataFrame
    and a dict mapping column → encoding array for inversion if needed.

    Args:
        df: Original DataFrame.

    Returns:
        Tuple (df_encoded, encoders_dict).
    """
    df_enc = df.copy()
    encoders: Dict[str, np.ndarray] = {}

    for col in df_enc.columns:
        if df_enc[col].dtype == object or str(df_enc[col].dtype) == "category":
            cats = df_enc[col].dropna().unique()
            mapping = {v: i for i, v in enumerate(sorted(cats))}
            encoders[col] = cats
            df_enc[col] = df_enc[col].map(mapping)

    return df_enc, encoders
