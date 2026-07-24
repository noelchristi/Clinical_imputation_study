# Clinical Imputation Study

**A Monte Carlo Simulation Study Comparing Missing Data Imputation Methods in Cardiometabolic Clinical Research**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## Overview

Complete, reproducible pipeline for evaluating missing data imputation methods on real-world cardiometabolic clinical data. The study compares **4 imputation methods** under **3 missingness mechanisms** (MCAR, MAR, MNAR) at **3 missing rates** (10%, 20%, 40%) using **Monte Carlo simulation** (M = 5 repetitions).

**Key finding**: MissForest consistently and significantly outperforms Median/Mode, KNN, and MICE across all variable types and scenarios (Friedman χ² = 37.25, p < 10⁻⁷, η² = 0.52).

---

## Project Structure

```
Clinical_Imputation_Study/
│
├── data/raw/                          # Clinical dataset (N = 312, P = 33)
│
├── v1/                                # Baseline: single-run MCAR simulation
├── v2/                                # Monte Carlo: MCAR + MAR, 4 methods
├── v3/                                # Supplements: MNAR, Rubin's rules, calibration
│
├── v4/                                # Production pipeline (current)
│   ├── src/
│   │   ├── amputation/                # MCAR, MAR, MNAR generators
│   │   ├── imputers/                  # Median/Mode, KNN, MICE (PMM), MissForest
│   │   ├── evaluation/                # Continuous & categorical metrics, clinical impact
│   │   ├── stats_utils/               # Friedman, Nemenyi, Holm, Rubin's rules, Cohen's d
│   │   ├── visualization/             # Publication-ready figures (300 DPI)
│   │   ├── reporting/                 # CSV/JSON export, LaTeX/Markdown tables
│   │   └── utils/                     # Data loading, logging, preprocessing
│   ├── config/                        # Pipeline configuration
│   ├── outputs/                       # Generated results
│   ├── figures/                       # Generated figures
│   ├── tests/                         # Unit tests (10/10 passing)
│   ├── docs/                          # Documentation & reports
│   ├── article/                       # Manuscript (Markdown + DOCX)
│   ├── run.py                         # Main entry point
│   ├── requirements.txt
│   └── environment.yml
│
└── README.md
```

---

## Methods Compared

| # | Method | Algorithm | Reference |
|---|--------|-----------|-----------|
| 1 | **Median/Mode** | Univariate median (continuous) / mode (categorical) | Baseline |
| 2 | **KNN (k=5)** | k-Nearest Neighbours, partial Euclidean distance | Troyanskaya et al. (2001) |
| 3 | **MICE (PMM)** | Multiple Imputation by Chained Equations with Predictive Mean Matching | van Buuren (2018), Rubin (1987) |
| 4 | **MissForest** | Iterative Random Forest imputation | Stekhoven & Bühlmann (2012) |

---

## Missingness Mechanisms

- **MCAR** — Missing Completely At Random: deletion independent of all values
- **MAR** — Missing At Random: deletion depends on observed variables (logistic transform)
- **MNAR** — Missing Not At Random: deletion depends on the missing value itself (distance from median)

---

## Evaluation Metrics

**Continuous variables**: RMSE, NRMSE, MAE, Bias, KS test, Wilcoxon test, variance ratio (R_V), correlation preservation (Pearson/Spearman/Kendall)

**Categorical variables**: F1-score (weighted), Accuracy, Cohen's κ, Matthews Correlation Coefficient (MCC), Chi-squared test

**Clinical impact**: Logistic regression (AUC, Brier score, Hosmer-Lemeshow, OR bias, significance change rate), Linear regression (R², residual diagnostics)

**Multiple imputation**: Rubin's rules (Within/Between/Total variance, RIV, FMI, Relative Efficiency)

**Statistical inference**: Friedman test with Nemenyi post-hoc and Holm correction, Cohen's d, Cliff's δ, η²

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/clinical-imputation-study.git
cd clinical-imputation-study

# Install dependencies
pip install -r v4/requirements.txt

# Run full pipeline (MCAR + MAR, M = 5 repetitions, ~2 hours)
cd v4
python run.py --mc 5 --skip-mnar --seed 42

# Run quick demo (MCAR only, 1 repetition, ~5 minutes)
python run.py --mc 1 --skip-mnar --skip-mar
```

---

## Key Results (v4, MCAR 20%)

| Method | RMSE (Cont.) | F1 (Cat.) | R_V | Friedman Rank |
|--------|-------------|-----------|-----|---------------|
| **MissForest** | **13.23** | **0.86** | **0.80** | **1.13** |
| KNN (k=5) | 27.07 | 0.56 | 0.27 | 2.88 |
| Median/Mode | 27.30 | 0.73 | 0.00* | 2.83 |
| MICE (PMM) | 32.69 | 0.81 | 0.99 | 3.17 |

*Bug detected in R_V computation for Median/Mode — see v4/docs/RAPPORT_V4.md.

**Friedman test**: χ² = 37.25, p = 4.1 × 10⁻⁸, η² = 0.52  
**MissForest is significantly superior to all other methods** (Holm-corrected p < 0.01).

---

## Manuscript

The complete scientific manuscript is available in `v4/article/`:
- `manuscrit_final_corrige.md` — Full article (5 sections, 17 figures, 6 tables)
- `manuscrit_final_corrige_v3.docx` — DOCX with all figures embedded

**Abstract**: Monte Carlo simulation study (N = 312 subjects, 33 cardiometabolic variables) comparing 4 imputation methods under MCAR, MAR, and MNAR at 10%, 20%, and 40% missing rates. MissForest is the recommended first-line method for clinical data with skewed distributions and nonlinear relationships.

---

## Requirements

- Python ≥ 3.12
- numpy ≥ 1.24, pandas ≥ 2.0, scipy ≥ 1.10
- scikit-learn ≥ 1.3, matplotlib ≥ 3.7, seaborn ≥ 0.12
- statsmodels, scikit-posthocs

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@misc{clinical-imputation-study-2026,
  title  = {A Monte Carlo Simulation Study Comparing Missing Data Imputation Methods
            in Cardiometabolic Clinical Research},
  author = {},
  year   = {2026},
  note   = {Complete reproducible pipeline with manuscript},
  url    = {https://github.com/your-org/clinical-imputation-study}
}
```

---

## License

MIT License. See `LICENSE` file for details.

---

## References

1. Rubin DB. *Multiple Imputation for Nonresponse in Surveys*. Wiley, 1987.
2. Stekhoven DJ, Bühlmann P. MissForest. *Bioinformatics*, 2012;28(1):112-118.
3. van Buuren S. *Flexible Imputation of Missing Data*. 2nd ed. CRC Press, 2018.
4. Little RJA, Rubin DB. *Statistical Analysis with Missing Data*. 3rd ed. Wiley, 2019.
5. Waljee AK et al. Comparison of imputation methods. *BMJ Open*, 2013;3(8):e002847.
6. Demšar J. Statistical comparisons of classifiers over multiple data sets. *JMLR*, 2006;7:1-30.
7. Jakobsen JC et al. When and how should multiple imputation be used. *BMC Med Res Methodol*, 2017;17(1):162.
