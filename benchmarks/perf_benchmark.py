# -*- coding: utf-8 -*-
"""Reproducible performance benchmark for v4 imputation/evaluation kernels.

This benchmark is methodology-neutral: it only measures runtime and memory
for existing APIs on a fixed synthetic dataset.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import time
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amputation import ampute_mcar
from evaluation import evaluate_all_categorical, evaluate_all_continuous, variance_ratio
from imputers import impute_knn, impute_median_mode, impute_mice, impute_missforest


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_mb() -> float:
    """Return process working-set memory in MB (Windows)."""
    psapi = ctypes.WinDLL("Psapi.dll")
    kernel32 = ctypes.WinDLL("Kernel32.dll")
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ok = get_process_memory_info(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if ok == 0:
        return float("nan")
    return counters.WorkingSetSize / (1024.0**2)


def build_synthetic_data(seed: int = 123) -> tuple[np.ndarray, list[int], list[int], list[int], list[int], list[str]]:
    """Generate complete mixed-type matrix for deterministic benchmarking."""
    rng = np.random.default_rng(seed)
    n_rows = 500
    n_cont, n_bin, n_ord = 24, 8, 8

    x_cont = rng.normal(loc=0.0, scale=1.5, size=(n_rows, n_cont))
    x_bin = rng.binomial(1, 0.35, size=(n_rows, n_bin)).astype(float)
    x_ord = rng.integers(0, 4, size=(n_rows, n_ord)).astype(float)

    x_complete = np.hstack([x_cont, x_bin, x_ord]).astype(float)

    continuous_idx = list(range(n_cont))
    binary_idx = list(range(n_cont, n_cont + n_bin))
    ordinal_idx = list(range(n_cont + n_bin, n_cont + n_bin + n_ord))
    categorical_idx = binary_idx + ordinal_idx
    col_names = [f"v{i}" for i in range(x_complete.shape[1])]

    return x_complete, continuous_idx, binary_idx, ordinal_idx, categorical_idx, col_names


def measure_block(fn, repeats: int) -> tuple[float, float, float, float, object]:
    """Measure mean/std runtime and RAM deltas over repeated runs."""
    times = []
    ram_deltas = []
    out = None

    for _ in range(repeats):
        gc.collect()
        before = working_set_mb()
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        after = working_set_mb()

        times.append(dt)
        ram_deltas.append(after - before)

    return (
        float(np.mean(times)),
        float(np.std(times)),
        float(np.mean(ram_deltas)),
        float(np.std(ram_deltas)),
        out,
    )


def run_benchmark() -> dict:
    """Execute benchmark for all kernels and return metrics as dict."""
    (
        x_complete,
        continuous_idx,
        binary_idx,
        ordinal_idx,
        categorical_idx,
        col_names,
    ) = build_synthetic_data()

    x_miss, mask = ampute_mcar(x_complete, 0.20, seed=42)

    results = {}
    cache = {}

    blocks = {
        "impute_median_mode": (
            lambda: impute_median_mode(x_miss, continuous_idx, categorical_idx),
            3,
        ),
        "impute_knn": (lambda: impute_knn(x_miss, k=5), 3),
        "impute_mice": (
            lambda: impute_mice(
                x_miss,
                continuous_idx,
                binary_idx,
                ordinal_idx,
                max_iter=5,
                random_state=42,
            ),
            3,
        ),
        "impute_missforest": (
            lambda: impute_missforest(
                x_miss,
                continuous_idx,
                categorical_idx,
                n_estimators=30,
                max_iter=4,
                random_state=42,
            ),
            3,
        ),
    }

    for name, (fn, repeats) in blocks.items():
        t_mean, t_std, m_mean, m_std, out = measure_block(fn, repeats)
        cache[name] = out
        results[name] = {
            "time_s_mean": t_mean,
            "time_s_std": t_std,
            "ram_delta_mb_mean": m_mean,
            "ram_delta_mb_std": m_std,
            "checksum_sum": float(np.sum(out)),
        }

    x_mf = cache["impute_missforest"]
    eval_blocks = {
        "eval_continuous": (
            lambda: evaluate_all_continuous(
                x_complete, x_mf, mask, continuous_idx, col_names
            ),
            10,
        ),
        "eval_categorical": (
            lambda: evaluate_all_categorical(
                x_complete, x_mf, mask, categorical_idx, col_names
            ),
            10,
        ),
        "variance_ratio": (
            lambda: variance_ratio(x_complete, x_mf, mask, continuous_idx),
            10,
        ),
    }

    for name, (fn, repeats) in eval_blocks.items():
        t_mean, t_std, m_mean, m_std, out = measure_block(fn, repeats)
        out_size = (
            int(len(out))
            if isinstance(out, list)
            else int(len(out.get("per_variable", [])))
        )
        results[name] = {
            "time_s_mean": t_mean,
            "time_s_std": t_std,
            "ram_delta_mb_mean": m_mean,
            "ram_delta_mb_std": m_std,
            "out_size": out_size,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run performance benchmark")
    parser.add_argument(
        "--output", required=True, help="Output JSON file path for benchmark metrics"
    )
    args = parser.parse_args()

    metrics = run_benchmark()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
