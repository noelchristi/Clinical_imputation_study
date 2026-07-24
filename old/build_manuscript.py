"""
Build the final integrated manuscript (manuscrit_final_corrige.md)
from the existing English manuscript by inserting figure image links
and captions at the correct locations.
"""

import re

SRC  = r'c:\Users\LENOVO\Desktop\THESE\article2\manuscrit_final.md'
DEST = r'c:\Users\LENOVO\Desktop\THESE\article2\manuscrit_final_corrige.md'
FIG_DIR = 'figures_final'   # relative path inside the document

with open(SRC, encoding='utf-8') as f:
    text = f.read()

# ── Figure captions ──────────────────────────────────────────────────────────
FIG1_BLOCK = """
---

![Figure 1 – Study Design and Analytical Framework]({FIG_DIR}/Fig1_StudyDesign_Flowchart.png)

**Figure 1.** Study design and analytical framework. A complete cardiometabolic clinical database (N = 312 subjects, P = 33 variables, 0% native missing data) served as ground truth. Nine experimental scenarios were defined by crossing three missingness mechanisms (MCAR, MAR, MNAR) with three missing rates (τ ∈ {{10%, 20%, 40%}}). Four imputation methods were applied across M = 5 Monte Carlo iterations each, yielding 180 independent imputation runs. Performance was evaluated through reconstruction accuracy, distribution preservation, correlation and variance conservation, clinical model fidelity, and Rubin's multiple imputation rules. Statistical comparisons employed the Friedman test with Nemenyi post-hoc analysis and Holm correction.

---
""".format(FIG_DIR=FIG_DIR)

FIG2_BLOCK = """

---

![Figure 2 – Missing Data Pattern Matrix (MCAR 20%)]({FIG_DIR}/Fig2_MissingDataPattern_MCAR20.png)

**Figure 2.** Missing data pattern matrix for the MCAR 20% scenario (first 50 patients, all 33 variables). Yellow cells indicate missing values; blue cells indicate observed values. Under MCAR, the deletion pattern is completely random, producing no systematic structure in the matrix. The 20% missing rate yields approximately 6–7 missing values per patient on average.

---
""".format(FIG_DIR=FIG_DIR)

FIG4_BLOCK = """

---

![Figure 4 – Per-Variable RMSE by Imputation Method (MCAR 40%)]({FIG_DIR}/Fig4_RMSE_Heatmap_Variables_MCAR40.png)

**Figure 4.** Per-variable RMSE by imputation method under MCAR 40% (continuous variables, N = 29, excluding zero-variance variables). Colour intensity reflects RMSE magnitude on a log scale. Variables are sorted by MissForest RMSE in descending order. MissForest (rightmost column) shows systematically lower RMSE across all variable types, with the most pronounced advantage for highly skewed variables (HOMA-IR, HOMAS%, HMW-Adiponectin) where the median-based baseline severely underestimates reconstruction error.

---
""".format(FIG_DIR=FIG_DIR)

FIG8_BLOCK = """

---

![Figure 8 – Distribution of RMSE by Method and Missing Rate (MCAR)]({FIG_DIR}/Fig8_Boxplot_RMSE_MCAR.png)

**Figure 8.** Distribution of variable-level RMSE across 29 continuous variables by imputation method and missing rate under MCAR. Boxes represent the interquartile range; horizontal red lines indicate medians; whiskers extend to 1.5×IQR; circles denote outlier variables (primarily pharmacological variables with extreme distributions). MissForest (red) exhibits the lowest median RMSE at all three missing rates, with the tightest IQR indicating consistency across variable types. MICE (BayesianRidge, green) shows the widest IQR and most extreme outliers, reflecting sensitivity to distributional violations.

---
""".format(FIG_DIR=FIG_DIR)

FIG3_BLOCK = """

---

![Figure 3 – Mean RMSE by Missingness Mechanism and Missing Rate]({FIG_DIR}/Fig3_RMSE_Curves_by_Mechanism.png)

**Figure 3.** Mean RMSE of imputation methods across 29 continuous variables as a function of missing rate (τ), stratified by missingness mechanism. **Left panel (MCAR):** MissForest consistently achieves the lowest RMSE at all missing rates. **Centre panel (MAR):** Performance of all methods degrades relative to MCAR; at 40% MAR, Median/Mode marginally surpasses MissForest due to structural dependency loss. **Right panel (MNAR):** All methods converge toward uniformly poor performance; the RMSE axis is truncated for readability. Error bars (where visible) represent ±1 SD across M = 5 Monte Carlo iterations.

---
""".format(FIG_DIR=FIG_DIR)

FIG5_BLOCK = """

---

![Figure 5 – Multi-Metric Performance Radar (MCAR)]({FIG_DIR}/Fig5_RadarPlot_MCAR.png)

**Figure 5.** Multi-metric performance profiles of the four imputation methods under MCAR at 10%, 20%, and 40% missing rates. Axes represent five complementary performance dimensions: F1-score and Cohen's κ for ordinal variables; accuracy for binary variables; 1−NRMSE for continuous reconstruction accuracy; and variance ratio (R_V) for distributional fidelity. MissForest (red) dominates across all metrics at all missing rates. The progressive shrinkage of all profiles with increasing missing rate reflects the universal degradation in imputation quality.

---
""".format(FIG_DIR=FIG_DIR)

FIG6_BLOCK = """

---

![Figure 6 – Post-Imputation Variance Ratio]({FIG_DIR}/Fig6_VarianceRatio.png)

**Figure 6.** Post-imputation variance ratio (R_V = Var_imputed / Var_original) as a function of missing rate under MCAR (solid lines) and MAR (dashed lines). R_V = 1.0 (dotted horizontal reference) indicates perfect variance conservation; R_V < 1 indicates variance underestimation (type I error inflation risk); R_V > 1 indicates variance inflation. The shaded green band [0.80, 1.20] represents the acceptable zone. Median/Mode (blue) collapses to R_V = 0.62 at 40% MCAR, crossing the critical threshold between 10% and 20% missing. MissForest (red) remains within the acceptable zone at all evaluated rates under both mechanisms. MICE (BayesianRidge, green) inflates variance above 1.0 at all rates, particularly under MAR (R_V = 1.23 at 40% MAR).

---
""".format(FIG_DIR=FIG_DIR)

FIG7_BLOCK = """

---

![Figure 7 – Evidence-Based Decision Algorithm for Clinical Imputation]({FIG_DIR}/Fig7_DecisionAlgorithm.png)

**Figure 7.** Evidence-based decision algorithm for selecting an imputation strategy in clinical research. Branching criteria are based on the proportion of missing data per variable and variable type. Recommendation strength is colour-coded: green (low-risk, permissive strategies), yellow/orange (caution, multiple imputation required), red (alert zone, heightened uncertainty). The mechanism guide (bottom-left) summarises recommendations by missingness mechanism; the computational trade-off panel (bottom-right) provides runtime benchmarks for N ≈ 300. MNAR = Missing Not At Random; M = number of imputations.

---
""".format(FIG_DIR=FIG_DIR)

# ── Insert points ─────────────────────────────────────────────────────────────

# Figure 1: before "## 1. INTRODUCTION"
text = text.replace(
    '\n## 1. INTRODUCTION\n',
    FIG1_BLOCK + '\n## 1. INTRODUCTION\n',
    1
)

# Figure 2: after the MNAR mathematical equation block (end of section 2.3)
# Insert after "### 2.4 Monte Carlo Protocol"
text = text.replace(
    '\n### 2.4 Monte Carlo Protocol\n',
    FIG2_BLOCK + '\n### 2.4 Monte Carlo Protocol\n',
    1
)

# Figure 3: after "and Table 3." in section 3.2
text = text.replace(
    'The impact of the missingness mechanism on MissForest performance at 40% missing rate is shown in Figure 3 and Table 3.',
    'The impact of the missingness mechanism on MissForest performance at 40% missing rate is shown in Figure 3 and Table 3.\n' + FIG3_BLOCK,
    1
)

# Figure 4 and Figure 8: after Table 2 block (after the end of Table 2 note)
# Insert after the "Abbreviations:" line that closes Table 2
text = text.replace(
    '*Abbreviations: RMSE = Root Mean Squared Error; MAE = Mean Absolute Error; KS = Kolmogorov-Smirnov test; F1 = weighted F1-score; Acc. = Accuracy; κ = Cohen\'s kappa; R_V = variance ratio.*',
    '*Abbreviations: RMSE = Root Mean Squared Error; MAE = Mean Absolute Error; KS = Kolmogorov-Smirnov test; F1 = weighted F1-score; Acc. = Accuracy; κ = Cohen\'s kappa; R_V = variance ratio.*'
    + FIG4_BLOCK + FIG8_BLOCK,
    1
)

# Figure 5: after Friedman / Nemenyi section 3.3, after "CD = 0.871, p < 0.0125 for all comparisons"
text = text.replace(
    'All three comparisons involving MissForest were significant after Holm correction (p < 0.0125). The Critical Difference Diagram (Figure S1) visually confirms this separation: MissForest\'s mean rank of 1.41 lies more than one CD (0.871) below all other methods.',
    'All three comparisons involving MissForest were significant after Holm correction (p < 0.0125). The Critical Difference Diagram (Figure S1) visually confirms this separation: MissForest\'s mean rank of 1.41 lies more than one CD (0.871) below all other methods.'
    + FIG5_BLOCK,
    1
)

# Figure 6: after "illustrated in Figure 6."
text = text.replace(
    'The variance ratio trajectories across mechanisms and rates are illustrated in Figure 6.',
    'The variance ratio trajectories across mechanisms and rates are illustrated in Figure 6.'
    + FIG6_BLOCK,
    1
)

# Figure 7: in section 4.6, after the last bullet of the decision algorithm list
text = text.replace(
    '- **Large-scale data** (N > 10⁵): Consider KNN or fast MICE variants (computational trade-off acceptable at low missing rates).',
    '- **Large-scale data** (N > 10⁵): Consider KNN or fast MICE variants (computational trade-off acceptable at low missing rates).'
    + FIG7_BLOCK,
    1
)

# ── Update figure count in header ──────────────────────────────────────────
text = text.replace('**Figures**: 8 |', '**Figures**: 8 (integrated) |', 1)

with open(DEST, 'w', encoding='utf-8') as f:
    f.write(text)

print(f'Manuscript with integrated figures saved to:\n  {DEST}')
print(f'File size: {len(text):,} characters')
