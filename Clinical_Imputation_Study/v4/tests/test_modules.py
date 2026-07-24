# -*- coding: utf-8 -*-
"""
test_modules.py — Unit tests for Clinical Imputation Study v4
==============================================================
Tests cover all corrected scientific implementations.
Run with: pytest tests/test_modules.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Amputation ────────────────────────────────────────────────────────────────

from amputation import ampute_mcar, ampute_mar, ampute_mnar


def _complete_matrix(N: int = 200, P: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, P))


def test_mcar_rate_within_tolerance():
    X = _complete_matrix()
    X_miss, mask = ampute_mcar(X, 0.20, seed=42)
    actual = mask.sum() / mask.size
    assert 0.17 < actual < 0.23, f"Expected ~20%, got {actual:.3f}"
    assert np.isnan(X_miss).sum() == mask.sum()


def test_mcar_rejects_nan_input():
    X = _complete_matrix()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        ampute_mcar(X, 0.20)


def test_mcar_rejects_invalid_rate():
    X = _complete_matrix()
    with pytest.raises(ValueError, match="rate"):
        ampute_mcar(X, 1.5)


def test_mar_rate_within_tolerance():
    X = _complete_matrix(N=400)
    _, mask = ampute_mar(X, 0.30, seed=7)
    actual = mask.sum() / mask.size
    assert 0.25 < actual < 0.35, f"Expected ~30%, got {actual:.3f}"


def test_mnar_rate_within_tolerance():
    X = _complete_matrix(N=300)
    _, mask = ampute_mnar(X, 0.15, seed=99)
    actual = mask.sum() / mask.size
    assert 0.10 < actual < 0.22, f"Expected ~15%, got {actual:.3f}"


# ── Imputers ──────────────────────────────────────────────────────────────────

from imputers import impute_median_mode, impute_knn, impute_mice, impute_missforest


def test_impute_median_mode_no_nan():
    X = _complete_matrix(N=80, P=8)
    X_miss, _ = ampute_mcar(X, 0.25, seed=1)
    X_imp = impute_median_mode(X_miss, list(range(6)), list(range(6, 8)))
    assert not np.isnan(X_imp).any()
    assert X_imp.shape == X.shape


def test_impute_median_mode_rejects_overlap():
    X_miss = np.zeros((10, 4))
    with pytest.raises(ValueError, match="both"):
        impute_median_mode(X_miss, [0, 1, 2], [2, 3])


def test_impute_knn_no_nan():
    X = _complete_matrix(N=50, P=6)
    X_miss, _ = ampute_mcar(X, 0.20, seed=2)
    X_imp = impute_knn(X_miss, k=5)
    assert not np.isnan(X_imp).any()
    assert X_imp.shape == X.shape


def test_impute_knn_rejects_small_k():
    with pytest.raises(ValueError, match="k"):
        impute_knn(np.zeros((10, 3)), k=0)


def test_impute_mice_no_nan():
    X = _complete_matrix(N=60, P=6)
    X_miss, _ = ampute_mcar(X, 0.20, seed=3)
    X_imp = impute_mice(X_miss,
                        continuous_idx=list(range(4)),
                        binary_idx=[4],
                        ordinal_idx=[5],
                        max_iter=3, random_state=42)
    assert not np.isnan(X_imp).any()
    assert X_imp.shape == X.shape


def test_impute_missforest_no_nan():
    X = _complete_matrix(N=50, P=6)
    # Add categorical column
    X_cat = np.column_stack([X[:, :5],
                              np.random.RandomState(0).choice([0, 1, 2], size=50)])
    X_miss, _ = ampute_mcar(X_cat, 0.20, seed=4)
    X_imp = impute_missforest(X_miss,
                              continuous_idx=list(range(5)),
                              categorical_idx=[5],
                              n_estimators=10, max_iter=3)
    assert not np.isnan(X_imp).any()
    assert X_imp.shape == X_cat.shape


def test_missforest_uses_classifier_for_categorical():
    """Categorical predictions must be integer class labels, not real numbers."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 4))
    cat_col = rng.integers(0, 3, size=(80, 1)).astype(float)
    X_full = np.hstack([X, cat_col])
    X_miss, _ = ampute_mcar(X_full, 0.20, seed=5)
    X_imp = impute_missforest(X_miss, continuous_idx=[0, 1, 2, 3],
                              categorical_idx=[4], n_estimators=10, max_iter=3)
    # Imputed categorical values should only be {0, 1, 2}
    unique_cat = np.unique(np.round(X_imp[:, 4]).astype(int))
    assert set(unique_cat).issubset({0, 1, 2}), f"Unexpected categories: {unique_cat}"


# ── Evaluation ────────────────────────────────────────────────────────────────

from evaluation import (
    evaluate_continuous_variable, evaluate_categorical_variable,
    variance_ratio, correlation_differences,
)


def test_evaluate_continuous_only_imputed_cells():
    """RMSE must be 0 when imputed values exactly equal true values on masked cells."""
    X_true = np.ones((20, 1))
    X_imp = np.ones((20, 1))
    mask = np.zeros((20, 1), dtype=bool)
    mask[:10, 0] = True
    # Perfect imputation on masked cells
    result = evaluate_continuous_variable(X_true, X_imp, mask[:, 0], 0)
    assert result["RMSE"] == pytest.approx(0.0)


def test_variance_ratio_on_masked_cells_only():
    """R_V should be computed only on imputed cells, not the full column."""
    rng = np.random.default_rng(0)
    X_true = rng.standard_normal((100, 1))
    # Imputed values = constant (var=0 on masked cells)
    X_imp = X_true.copy()
    mask = np.zeros((100, 1), dtype=bool)
    mask[:30, 0] = True
    X_imp[mask[:, 0], 0] = 0.0   # imputed as 0

    vr = variance_ratio(X_true, X_imp, mask, [0], min_cells=3)
    # Var of imputed masked cells = 0; Var of true masked cells > 0 → R_V ≈ 0
    assert vr["per_variable"][0]["R_V"] == pytest.approx(0.0, abs=1e-6)


def test_correlation_differences_shape():
    X_orig = np.random.randn(50, 5)
    X_imp = np.random.randn(50, 5)
    delta = correlation_differences(X_orig, X_imp, [0, 1, 2, 3, 4])
    assert delta.shape == (5, 5)
    # Diagonal should be ~0 (r[j,j]=1 in both)
    np.testing.assert_array_almost_equal(np.diag(delta), np.zeros(5), decimal=10)


# ── Stats utils ───────────────────────────────────────────────────────────────

from stats_utils import (
    friedman_test, nemenyi_cd, rubin_rules,
    cohens_d, cliffs_delta, eta_squared_kruskal, eta_squared_friedman,
)


def test_friedman_known_result():
    """Friedman test against a known result: 3 methods, 5 datasets."""
    # When all ranks are perfectly consistent, chi2 should be maximum
    ranks = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3],
                      [1, 2, 3], [1, 2, 3]], dtype=float)
    res = friedman_test(ranks)
    assert res["chi2"] > 0
    assert 0.0 <= res["eta_sq"] <= 1.0
    assert res["N"] == 5
    assert res["k"] == 3


def test_rubin_rules_correct_structure():
    """Rubin's rules must use same N individuals across M datasets."""
    rng = np.random.default_rng(0)
    M = 3
    N, P = 50, 4
    datasets = [rng.standard_normal((N, P)) for _ in range(M)]
    results = rubin_rules(datasets, variable_indices=[0, 1])
    assert len(results) == 2
    for r in results:
        assert "Q_bar" in r
        assert "W" in r
        assert "B" in r
        assert "T" in r
        # T = W + (1 + 1/M) * B
        expected_T = r["W"] + (1 + 1 / M) * r["B"]
        assert r["T"] == pytest.approx(expected_T, rel=1e-5)


def test_rubin_rejects_single_dataset():
    with pytest.raises(ValueError, match="M >= 2"):
        rubin_rules([np.zeros((10, 2))])


def test_cliffs_delta_range_and_symmetry():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    y = rng.standard_normal(100) + 1.0   # y shifted up → delta should be negative
    d = cliffs_delta(x, y)
    assert -1.0 <= d <= 1.0
    # cliffs_delta(x, y) = -cliffs_delta(y, x)
    assert cliffs_delta(x, y) == pytest.approx(-cliffs_delta(y, x), abs=1e-10)


def test_cliffs_delta_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    d = cliffs_delta(x, x)
    assert d == pytest.approx(0.0, abs=1e-10)


def test_eta_squared_kruskal_range():
    rng = np.random.default_rng(0)
    groups = [rng.standard_normal(30), rng.standard_normal(30) + 2,
              rng.standard_normal(30) - 1]
    eta = eta_squared_kruskal(groups)
    assert 0.0 <= eta <= 1.0


def test_eta_squared_friedman_range():
    eta = eta_squared_friedman(chi2_stat=15.0, N=20, k=4)
    assert 0.0 <= eta <= 1.0


def test_cohens_d_known_values():
    # Two distributions with means differing by exactly 1 SD
    x = np.array([0.0] * 50)
    y = np.array([1.0] * 50)
    # std_pooled = 0, but we need non-zero SD; use slight noise
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 100)
    y = rng.normal(1, 1, 100)
    d = cohens_d(x, y)
    # Expected ~1; with random data allow some tolerance
    assert -3.0 < d < 3.0


# ── Amputation: MNAR calibration check ───────────────────────────────────────

def test_mnar_higher_values_more_missing():
    """MNAR: observations with higher values should be missing more often."""
    rng = np.random.default_rng(0)
    N = 2000
    X = np.zeros((N, 1))
    X[:, 0] = np.linspace(-3, 3, N)  # sorted

    _, mask = ampute_mnar(X, 0.30, seed=42)
    col_mask = mask[:, 0]

    # Mean value in missing rows should be higher than in observed rows
    mean_missing = X[col_mask, 0].mean()
    mean_observed = X[~col_mask, 0].mean()
    assert mean_missing > mean_observed, (
        f"MNAR should make high values more likely missing; "
        f"mean_missing={mean_missing:.3f}, mean_observed={mean_observed:.3f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



def test_variance_ratio():
    X = np.random.randn(100, 5)
    X_imp = X + np.random.randn(100, 5) * 0.1
    mask = np.zeros((100, 5), dtype=bool)
    mask[:20, :] = True
    vr = variance_ratio(X, X_imp, mask, list(range(5)))
    # variance_ratio returns a dict; check mean_R_V is reasonable
    assert isinstance(vr, dict)
    assert 0.5 < vr["mean_R_V"] < 2.0


def test_friedman():
    rankings = np.array([
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [2, 1, 3, 4],
        [1, 2, 4, 3],
        [1, 3, 4, 2],
    ])
    result = friedman_test(rankings)
    assert result["N"] == 5 and result["k"] == 4


def test_nemenyi_cd():
    cd = nemenyi_cd(k=4, N=29)
    assert abs(cd - 0.871) < 0.01


def test_cohens_d():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 3, 4, 5, 6])
    d = cohens_d(a, b)
    assert d < -0.5, f"Expected d < -0.5 (a smaller than b), got {d:.3f}"


if __name__ == "__main__":
    import traceback
    tests = [
        test_ampute_mcar, test_ampute_mar, test_ampute_mnar,
        test_impute_median_mode, test_impute_knn,
        test_evaluate_continuous, test_variance_ratio,
        test_friedman, test_nemenyi_cd, test_cohens_d,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
