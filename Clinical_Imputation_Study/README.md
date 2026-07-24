# Clinical Imputation Study

**A Monte Carlo Simulation Study Comparing Missing Data Imputation Methods in Cardiometabolic Clinical Research**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## Overview

Complete, reproducible pipeline for evaluating missing data imputation methods on real-world cardiometabolic clinical data (N = 312 subjects, 33 variables). The study compares **4 imputation methods** under **3 missingness mechanisms** (MCAR, MAR, MNAR) at **3 missing rates** (10%, 20%, 40%) using **Monte Carlo simulation**.

**Key finding**: MissForest consistently and significantly outperforms Median/Mode, KNN, and MICE across all variable types and scenarios (Friedman χ² = 37.25, p < 10⁻⁷, η² = 0.52, RMSE 1.8–2.5× lower).

---

## Project Structure

```
├── src/
│   ├── amputation/                # MCAR, MAR, MNAR generators
│   ├── imputers/                  # Median/Mode, KNN, MICE (PMM), MissForest
│   ├── evaluation/                # Continuous & categorical metrics, clinical impact, model preservation
│   ├── stats_utils/               # Friedman, Nemenyi, Holm, Rubin's rules, Cohen's d
│   ├── visualization/             # Publication-ready figures (300 DPI)
│   ├── reporting/                 # CSV/JSON export, LaTeX/Markdown tables
│   └── utils/                     # Data loading, logging, preprocessing
├── config/                        # Pipeline configuration (YAML)
├── article/                       # Manuscript (Markdown + DOCX, 17 figures)
├── tests/                         # Unit tests (10/10 passing)
├── docs/                          # Documentation & detailed report
├── benchmarks/                    # Performance benchmarks
├── run.py                         # Main entry point
├── requirements.txt
└── README.md
```

---

## Methods

| # | Method | Algorithm | Reference |
|---|--------|-----------|-----------|
| 1 | **Median/Mode** | Univariate median (continuous) / mode (categorical) | Baseline |
| 2 | **KNN (k=5)** | k-Nearest Neighbours, partial Euclidean distance | Troyanskaya et al. (2001) |
| 3 | **MICE (PMM)** | Multiple Imputation by Chained Equations with Predictive Mean Matching | van Buuren (2018), Rubin (1987) |
| 4 | **MissForest** | Iterative Random Forest imputation | Stekhoven & Bühlmann (2012) |

### Missingness Mechanisms
- **MCAR** — Missing Completely At Random: deletion independent of all values
- **MAR** — Missing At Random: deletion depends on observed variables (logistic transform)
- **MNAR** — Missing Not At Random: deletion depends on the missing value itself

### Evaluation Metrics
- **Continuous**: RMSE, NRMSE, MAE, Bias, KS test, Wilcoxon test, correlation preservation (Pearson/Spearman/Kendall)
- **Categorical**: F1-score (weighted), Accuracy, Cohen's κ, Matthews Correlation Coefficient, Chi-squared test
- **Clinical impact**: Logistic regression (AUC, Brier score, Hosmer-Lemeshow, OR bias, significance change rate)
- **Multiple imputation**: Rubin's rules (Within/Between/Total variance, RIV, FMI, Relative Efficiency)
- **Statistical inference**: Friedman test with Nemenyi post-hoc and Holm correction, Cohen's d, Cliff's δ, η²

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/clinical-imputation-study.git
cd clinical-imputation-study

# Install dependencies
pip install -r requirements.txt

# Quick demo (MCAR only, 1 repetition, ~5 minutes)
python run.py --mc 1 --skip-mar --skip-mnar --seed 42

# Full pipeline (MCAR + MAR, M = 5, ~2 hours)
python run.py --mc 5 --skip-mnar --seed 42

# Complete pipeline (all 3 mechanisms, M = 100 for publication)
python run.py --mc 100 --seed 42
```

---

## Key Results (MCAR 20%)

| Method | RMSE (Cont.) | F1 (Cat.) | Friedman Rank |
|--------|-------------|-----------|---------------|
| **MissForest** | **13.23** | **0.86** | **1.13** |
| KNN (k=5) | 27.07 | 0.56 | 2.88 |
| Median/Mode | 27.30 | 0.73 | 2.83 |
| MICE (PMM) | 32.69 | 0.81 | 3.17 |

**Friedman test**: χ² = 37.25, p = 4.1 × 10⁻⁸, η² = 0.52.  
**MissForest is significantly superior to all other methods** (post-hoc Holm-corrected p < 0.01).

---

## Manuscript

Complete publication-ready manuscript in `article/`:
- `manuscrit_final_corrige.md` — Full article (5 sections, 17 figures, 6 tables)
- `manuscrit_final_corrige_v3.docx` — DOCX with all figures embedded

Detailed run report: `docs/RAPPORT_V4.md`.

---

## Requirements

- Python ≥ 3.12
- numpy ≥ 1.24, pandas ≥ 2.0, scipy ≥ 1.10
- scikit-learn ≥ 1.3, matplotlib ≥ 3.7, seaborn ≥ 0.12

---

## Citation

```bibtex
@misc{clinical-imputation-study-2026,
  title  = {A Monte Carlo Simulation Study Comparing Missing Data Imputation
            Methods in Cardiometabolic Clinical Research},
  year   = {2026},
  note   = {Reproducible pipeline with complete manuscript},
  url    = {https://github.com/your-org/clinical-imputation-study}
}
```

---

## References

1. Rubin DB. *Multiple Imputation for Nonresponse in Surveys*. Wiley, 1987.
2. Stekhoven DJ, Bühlmann P. MissForest. *Bioinformatics*, 2012;28(1):112-118.
3. van Buuren S. *Flexible Imputation of Missing Data*. 2nd ed. CRC Press, 2018.
4. Little RJA, Rubin DB. *Statistical Analysis with Missing Data*. 3rd ed. Wiley, 2019.
5. Waljee AK et al. Comparison of imputation methods. *BMJ Open*, 2013;3(8):e002847.
6. Jakobsen JC et al. When and how should multiple imputation be used. *BMC Med Res Methodol*, 2017;17(1):162.
