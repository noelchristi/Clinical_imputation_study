# A Monte Carlo Simulation Study Comparing Missing Data Imputation Methods in Cardiometabolic Clinical Research: MCAR, MAR and MNAR Mechanisms at 10%, 20% and 40% Missing Rates

---

**Running title**: Imputation Methods for Missing Clinical Data

**Word count**: ~6,800 (main text)
**Figures**: 8 | **Tables**: 6 | **Supplementary**: 13 figures, 12 tables

---

## ABSTRACT

**Background** — Missing data are ubiquitous in clinical research and can introduce substantial bias if handled inappropriately. While multiple imputation methods exist, their comparative performance under realistic cardiometabolic data structures — characterised by highly skewed distributions and nonlinear inter-variable relationships — remains insufficiently characterised across missingness mechanisms.

**Objective** — To systematically evaluate and compare four imputation methods (Median/Mode, KNN, MICE/BayesianRidge, MissForest) under MCAR, MAR, and MNAR mechanisms at 10%, 20%, and 40% missing rates, using Monte Carlo simulation on a real-world cardiometabolic clinical database.

**Methods** — A complete clinical database of 312 subjects and 33 cardiometabolic variables served as ground truth. Missing data were generated under three mechanisms (MCAR, MAR, MNAR) at three rates each. Each scenario was repeated over M = 5 Monte Carlo iterations. Imputation performance was assessed via RMSE, NRMSE, MAE, bias, F1-score, Cohen's κ, MCC, KS and Wilcoxon tests for continuous variables, and Chi-squared tests for categorical variables. Rubin's rules (RIV, FMI, RE) quantified multiple imputation efficiency. Model preservation was evaluated through logistic regression (HTA prediction: AUC, Brier score, OR bias, β coefficient bias, significance change rate) and linear regression (IMC, R²). The Friedman test with Nemenyi post-hoc and Holm correction provided formal statistical comparison. Sensitivity analyses varied KNN k ∈ {3, 5, 10, 15} and MissForest hyperparameters (n_estimators ∈ {50, 100}, max_depth ∈ {5, 10, ∞}). Robustness was tested against outliers and reduced sample size (N = 100).

**Results** — MissForest consistently and significantly outperformed all other methods across all variable types, missingness mechanisms, and missing rates. For continuous variables under MCAR 20%, MissForest achieved RMSE = 32.53 versus 60.47 (Median/Mode), 55.09 (KNN), and 70.28 (MICE). For ordinal variables, MissForest attained Cohen's κ = 0.731 at 20% MCAR versus κ = 0.000 for Median/Mode. The Friedman test was highly significant across all scenarios (p < 10⁻⁵, η² = 0.95), and Nemenyi post-hoc tests with Holm correction confirmed MissForest as significantly superior (CD = 0.871, p < 0.0125 for all comparisons). MissForest preserved correlations best (mean absolute Pearson Δr = 0.022, 2.1× better than Median/Mode), produced the best-calibrated clinical model (AUC = 0.941, Brier score = 0.087, OR relative bias = 15.3%, significance change rate = 22.2%), and maintained the most stable variance ratio (R_V = 0.824 at 40% MCAR versus 0.620 for Median/Mode). Rubin's rules confirmed M = 5 imputations as sufficient for 20% missing data (RE = 0.937, RIV < 0.01). Under MNAR, all methods degraded substantially: MissForest RMSE increased by 74% from MCAR to MNAR at 20%, and MICE exhibited catastrophic failure at 40% MNAR (R_V = 1,309.5). Sensitivity analyses confirmed MissForest robustness to hyperparameters (RMSE variation < 4%). Median/Mode showed paradoxical improvement on reduced samples (RMSE decreasing from 60.5 to 34.9 at N = 100) while MissForest degraded by 26%.

**Conclusion** — MissForest is the recommended first-line imputation method for cardiometabolic clinical data characterised by skewed distributions and nonlinear multivariate relationships, for missing rates up to 30%. Median/Mode imputation should be abandoned in inferential clinical research due to severe variance underestimation (R_V = 0.62 at 40% MCAR) and massive distributional distortion (89.7% of variables with significant KS test). Under MNAR, no standard imputation method is reliable, and selection models with sensitivity analyses are mandatory. An evidence-based decision tree is provided to guide clinical researchers in selecting appropriate imputation strategies.

**Keywords**: missing data, MCAR, MAR, MNAR, multiple imputation, MissForest, MICE, KNN imputation, Monte Carlo simulation, Friedman test, Rubin's rules, variance ratio, calibration, cardiometabolic disease, clinical research methodology.

---

## 1. INTRODUCTION

Missing data represent one of the most pervasive methodological challenges in clinical research, affecting virtually every observational study and randomised controlled trial [1,2]. In cardiometabolic research — where datasets routinely combine anthropometric measurements, haemodynamic parameters, lipid profiles, glycaemic biomarkers, and inflammatory markers — the proportion of incomplete records can range from 10% to over 40% depending on the variable and clinical setting [3].

The consequences of inappropriate missing data handling are well documented. Complete-case analysis, still the default approach in many clinical publications, discards potentially informative observations, reduces statistical power, and — critically — can produce biased estimates when data are not missing completely at random (MCAR) [1,4]. At the opposite extreme, simplistic imputation methods such as mean or median substitution artificially reduce variance, compress distributions toward the centre, and inflate type I error rates in subsequent inferential analyses [5,6].

The methodological armamentarium for missing data has expanded considerably over the past three decades. Multiple Imputation by Chained Equations (MICE), introduced by van Buuren and colleagues and formalised within Rubin's framework of repeated imputation and combination rules, has become the de facto standard in clinical epidemiology [2,7,8]. Non-parametric alternatives such as k-Nearest Neighbours (KNN) imputation and Random Forest-based imputation (MissForest) have gained traction, particularly for datasets characterised by nonlinear relationships and non-Gaussian distributions [9,10]. However, the comparative performance of these methods under clinically realistic data structures — specifically, highly skewed biomarker distributions, mixed continuous-categorical variable types, and varying missingness mechanisms — remains incompletely characterised.

Three fundamental concerns motivate the present study. First, the majority of simulation studies evaluating imputation methods employ synthetic data drawn from multivariate normal distributions, a scenario that poorly reflects the skewness, heavy tails, and nonlinear associations prevalent in cardiometabolic biomarkers [11]. Second, while MCAR is the most commonly simulated mechanism, the more clinically plausible MAR (Missing At Random) and particularly MNAR (Missing Not At Random) scenarios are rarely examined within a unified comparative framework [12]. Third, existing comparisons often emphasise reconstruction accuracy (RMSE, MAE) while neglecting the downstream impact on clinical prediction models — the very analyses for which imputation is performed in practice [13].

We designed a comprehensive Monte Carlo simulation study grounded in a real, complete cardiometabolic clinical database (N = 312, P = 33 variables). Our objectives were: (1) to quantify the comparative performance of four imputation methods (Median/Mode baseline, KNN, MICE/BayesianRidge, MissForest) across three missingness mechanisms (MCAR, MAR, MNAR) at three missing rates (10%, 20%, 40%); (2) to evaluate the downstream impact of imputation on clinical prediction models (logistic regression for hypertension, linear regression for BMI); (3) to assess multiple imputation properties via Rubin's rules; (4) to conduct formal statistical comparisons using Friedman tests with Nemenyi post-hoc analyses; and (5) to derive an evidence-based decision algorithm guiding clinical researchers in selecting appropriate imputation strategies.

---

## 2. MATERIAL AND METHODS

### 2.1 Study Database

The study utilised a complete, anonymised clinical database from a single-centre cardiometabolic cohort of 312 adult subjects (mean age 57.2 ± 11.7 years; 52.9% female). The database contained no native missing values, enabling its use as an uncontaminated ground truth for controlled amputation experiments.

The 33 variables spanned six clinical domains:

| Domain | Variables | n |
|--------|-----------|----|
| Anthropometry | Age, Sex, BMI, Waist circumference | 4 |
| Haemodynamics | Systolic BP, Diastolic BP, Hypertension status, HT subtype | 4 |
| Lipid metabolism | Protein, HDL-c, Total-c, LDL-c, Triglycerides, Non-HDL-c, 4 lipid ratios | 10 |
| Glucose metabolism | Glucose, Insulin, HOMAB%, HOMA-IR, HOMAS%, Glycaemic profile | 6 |
| Inflammation | PCSK9, High Molecular Weight Adiponectin | 2 |
| Pharmacology | Glibenclamide, Insulin, Lasilix, Lexomil, Metformin (doses) | 5 |
| Derived | Insulin (pmol/L), Glucose (mmol/L) | 2 |

### 2.2 Data Quality Audit and Variable Classification

An automated quality audit was performed prior to simulation. Variables with zero variance (Lasilix 40 mg, Lexomil 6 mg: all subjects at 0 mg) were identified and excluded from multivariate analyses. Four variable pairs exhibited extreme correlations (r > 0.95), attributable to unit conversion redundancies (Glu ↔ lu(mmol/l): r = 1.000; Ins ↔ Insu pmol/L: r = 1.000). No outliers requiring exclusion were detected. Variance Inflation Factors (VIF) were < 10 for all predictors, indicating acceptable collinearity.

Each variable was algorithmically classified based on distributional properties (Table 1). Normality was assessed via the Shapiro-Wilk test (α = 0.05) and skewness quantification.

**Table 1 — Variable classification (N = 312 subjects, 33 variables)**

| Class | n | % | Examples | Skewness range |
|-------|---|---|----------|---------------|
| Continuous, Gaussian | 2 | 6.1 | Age, Total cholesterol (Shapiro-Wilk p > 0.05) | −0.04 to +0.24 |
| Continuous, skewed | 23 | 69.7 | BMI, SBP, HDL-c, LDL-c, Triglycerides, PCSK9 | +0.10 to +2.74 |
| Continuous, highly skewed (skew > 3) | 6 | 18.2 | HOMA-IR (+8.81), HMW Adiponectin (+7.55), Insulin (+3.64), HOMAB% (+2.64), HOMAS% (+5.79) | +2.64 to +8.81 |
| Continuous, quasi-constant | 2 | 6.1 | Lasilix, Lexomil | — |
| Binary | 2 | 6.1 | Sex, Hypertension | — |
| Ordinal | 2 | 6.1 | HT subtype (3 levels), Glycaemic profile (4 levels) | — |

A total of 87.9% (29/33) of continuous variables were non-Gaussian, establishing distributional non-normality as the central methodological constraint of this study.

### 2.3 Missing Data Simulation

Three missingness mechanisms were implemented at three missing rates (τ ∈ {0.10, 0.20, 0.40}), yielding nine experimental scenarios.

**MCAR (Missing Completely At Random).** Each cell (i, j) of the N × P data matrix X was independently deleted with probability τ:

$$P(R_{ij} = 0 \mid X_{\text{obs}}, X_{\text{miss}}) = \tau$$

**MAR (Missing At Random).** The probability of missingness for variable j depended on the remaining observed variables X_{−j} through a logistic transformation:

$$P(R_{ij} = 0) = \tau + (1 - \tau) \cdot \sigma\left( \sum_{k \neq j} w_k X_{ik}^{\text{obs}} \right)$$

where σ(·) denotes the logistic function and weights w_k were drawn from Uniform(−1, 1).

**MNAR (Missing Not At Random).** Missingness probability was proportional to the deviation of the value from the variable median:

$$P(R_{ij} = 0) \propto \tau + (1 - \tau) \cdot \sigma\left( \frac{|X_{ij} - \text{med}(X_j)|}{\text{MAD}(X_j)} \right)$$

where MAD denotes the median absolute deviation.

### 2.4 Monte Carlo Protocol

For each {mechanism × rate} combination, amputation was repeated over M = 5 independent Monte Carlo iterations (configurable to M = 100) with distinct random seeds (seed = 42 + 100·rep + 100·τ). Performance metrics are reported as mean ± standard deviation across iterations. The full experimental design comprised 3 mechanisms × 3 rates × 5 repetitions × 4 methods = 180 independent imputation runs.

### 2.5 Imputation Methods

**Median/Mode (Baseline).** Univariate non-parametric imputation: missing continuous values replaced by the empirical median; missing categorical values replaced by the mode. Implemented via `SimpleImputer` (scikit-learn 1.7.0). This method serves as the minimal benchmark: any advanced method must significantly outperform it to justify additional complexity.

**KNN Imputer (k = 5).** Missing values estimated as the unweighted mean of the k = 5 nearest neighbours in the space of complete variables, using partial Euclidean distance. Implemented via `KNNImputer` (scikit-learn).

$$\hat{X}_{ij} = \frac{1}{k} \sum_{m \in \mathcal{N}_k(i)} X_{mj}$$

**MICE — Multiple Imputation by Chained Equations.** Iterative chained equations with BayesianRidge estimator. Each variable X_j is regressed on X_{−j}, with parameters sampled from their Bayesian posterior distribution (`sample_posterior = True`). Implemented via `IterativeImputer` (scikit-learn, 10 iterations). For quantifying multiple imputation properties, M = 5 imputed datasets were generated via bootstrap resampling, and Rubin's combination rules were applied [1,7]:

$$\bar{Q} = \frac{1}{M} \sum_{m=1}^{M} \hat{Q}_m; \quad T = \bar{W} + \left(1 + \frac{1}{M}\right) B$$

where \bar{W} is the within-imputation variance and B the between-imputation variance. The Relative Increase in Variance (RIV), Fraction of Missing Information (FMI), and Relative Efficiency (RE) were computed for eight representative continuous variables.

**MissForest.** Iterative non-parametric algorithm based on Random Forests. For each variable, a Random Forest regressor (100 trees, max_depth = 10) is trained on observed values and used to predict missing entries. Implemented via `IterativeImputer` with `RandomForestRegressor` (scikit-learn, 5 iterations), following the methodology of Stekhoven and Bühlmann [9].

### 2.6 Evaluation Metrics

**Continuous variables.** Reconstruction accuracy: RMSE, NRMSE (RMSE normalised by variable range), MAE, and bias (mean signed error). Distribution preservation: two-sample Kolmogorov-Smirnov test p-value and paired Wilcoxon signed-rank test p-value. Effect sizes: Cohen's d (standardised mean difference in RMSE versus Median/Mode baseline) and Cliff's δ (non-parametric effect size for KS test).

**Categorical/ordinal variables.** Classification metrics: weighted F1-score, accuracy, Cohen's κ, and Matthews Correlation Coefficient (MCC). Distribution preservation: Chi-squared test of independence between original and imputed class distributions.

**Correlation preservation.** Mean absolute difference in pairwise correlations (Pearson r, Spearman ρ, Kendall τ) between original and imputed data:

$$\Delta r = \frac{1}{P(P-1)/2} \sum_{i<j} \left| r_{ij}^{\text{orig}} - r_{ij}^{\text{imp}} \right|$$

**Variance stability.** Ratio of post-imputation to original variance, R_V = Var(X_imp) / Var(X_orig). R_V < 1 indicates variance underestimation (type I error inflation risk); R_V > 1 indicates variance inflation (type II error risk).

**Multiple imputation properties.** Within-imputation variance (\bar{W}), between-imputation variance (B), total variance (T), RIV, FMI, and RE [1].

**Clinical model preservation.** A logistic regression model predicting hypertension (HTA) from 25 continuous predictors was fitted after each imputation. Metrics: accuracy, AUC, Brier score, Hosmer-Lemeshow calibration test, relative OR bias, maximum absolute β coefficient bias, standard error ratio (imputed/reference), and significance change rate (proportion of predictors whose p < 0.05 status changed). A linear regression model predicting BMI was also fitted, with R² reported.

### 2.7 Statistical Comparisons

The Friedman test (non-parametric repeated-measures ANOVA) was applied to the rankings of methods by RMSE across variables. Effect size was quantified via η². Following significant Friedman tests, Nemenyi post-hoc tests identified significantly different method pairs using the Critical Difference:

$$\text{CD} = q_{\alpha, k} \sqrt{\frac{k(k+1)}{6N}}$$

with q_{0.05,4} = 2.569, k = 4 methods, N = 29 variables. Holm's sequentially rejective procedure controlled the family-wise error rate across all pairwise comparisons.

### 2.8 Sensitivity and Robustness Analyses

**Hyperparameter sensitivity.** KNN was evaluated at k ∈ {3, 5, 10, 15}. MissForest was tested over n_estimators ∈ {50, 100} and max_depth ∈ {5, 10, ∞}.

**Robustness.** (i) Outlier robustness: ratio of MAE for observations in the top 5% of BMI versus the remaining 95%. (ii) Small-sample robustness: comparison of RMSE between full dataset (N = 312) and a random subsample (N = 100). (iii) Multicollinearity: VIF computation for all continuous predictors.

### 2.9 Software and Reproducibility

All analyses were performed in Python 3.13.5 using pandas 2.3.1, numpy 2.3.1, scikit-learn 1.7.0, scipy 1.16.0, and matplotlib 3.10. The random seed was fixed at 42 for all stochastic components. Complete source code, aggregated results, and all figures (300 DPI) are available in the accompanying repository (`simulation_v2.py`, `supplement_v3.py`).

---

## 3. RESULTS

### 3.1 Overall Imputation Performance

Table 2 presents the aggregate performance of the four imputation methods for continuous, binary, and ordinal variables under the MCAR mechanism at all three missing rates.

**Table 2 — Comparative imputation performance under MCAR (mean ± SD across M = 5 Monte Carlo iterations)**

| Rate | Method | Continuous (n = 29) | | | Binary (n = 2) | | | Ordinal (n = 2) | | | Variance |
|------|--------|------|------|------|------|------|------|------|------|------|------|
| | | RMSE | MAE | KS p<0.05 (%) | F1 | Acc. | κ | F1 | Acc. | κ | R_V |
| **10%** | Median/Mode | 56.20 | 31.46 | 0.0 | 0.471 | 0.615 | 0.000 | 0.360 | 0.521 | 0.000 | 0.907 |
| | KNN (k=5) | 50.78 | 36.62 | 0.0 | 0.626 | 0.641 | 0.231 | 0.401 | 0.355 | 0.170 | 0.922 |
| | MICE (BayesRidge) | 67.36 | 51.22 | 0.0 | 0.632 | 0.638 | 0.237 | 0.441 | 0.380 | 0.148 | 1.007 |
| | **MissForest** | **26.27** | **15.20** | 0.0 | **0.734** | **0.744** | **0.452** | **0.865** | **0.846** | **0.771** | **0.959** |
| **20%** | Median/Mode | 60.47 | 30.97 | 17.2 | 0.506 | 0.643 | 0.000 | 0.397 | 0.555 | 0.000 | 0.812 |
| | KNN (k=5) | 55.09 | 34.89 | 0.0 | 0.657 | 0.672 | 0.272 | 0.363 | 0.317 | 0.127 | 0.836 |
| | MICE (BayesRidge) | 70.28 | 52.47 | 3.4 | 0.636 | 0.635 | 0.230 | 0.453 | 0.392 | 0.145 | 1.031 |
| | **MissForest** | **32.53** | **17.17** | 6.9 | **0.777** | **0.791** | **0.534** | **0.842** | **0.819** | **0.731** | **0.922** |
| **40%** | Median/Mode | 59.77 | 31.55 | **89.7** | 0.505 | 0.644 | 0.000 | 0.394 | 0.551 | 0.000 | **0.620** |
| | KNN (k=5) | 55.04 | 37.00 | 20.7 | 0.612 | 0.639 | 0.157 | 0.243 | 0.222 | 0.064 | 0.672 |
| | MICE (BayesRidge) | 74.55 | 57.19 | 37.9 | 0.590 | 0.585 | 0.122 | 0.364 | 0.314 | 0.028 | 1.121 |
| | **MissForest** | **39.92** | **21.45** | 17.2 | **0.740** | **0.745** | **0.438** | **0.798** | **0.778** | **0.655** | **0.824** |

*Abbreviations: RMSE = Root Mean Squared Error; MAE = Mean Absolute Error; KS = Kolmogorov-Smirnov test; F1 = weighted F1-score; Acc. = Accuracy; κ = Cohen's kappa; R_V = variance ratio.*

Three principal observations emerge from Table 2. First, **MissForest dominates systematically** across all variable types, missing rates, and metrics. For continuous variables at 20% MCAR, MissForest achieves an RMSE of 32.53, representing a 46.2% reduction compared with Median/Mode (60.47), a 40.9% reduction versus KNN (55.09), and a 53.7% reduction versus MICE (70.28). The superiority is most pronounced for ordinal variables, where MissForest attains κ = 0.771 at 10% missing versus κ = 0.000 for Median/Mode — the latter producing agreement no better than chance.

Second, **Median/Mode exhibits dangerously deceptive stability**. Its RMSE increases only 6.3% from 10% to 40% missing (56.20 → 59.77), suggesting robustness — yet this masks a catastrophic distributional collapse: 89.7% of continuous variables show statistically significant distributional distortion (KS p < 0.05) at 40%, and variance is underestimated by 38% (R_V = 0.620).

Third, **MICE with BayesianRidge is the worst-performing method for continuous variables** (RMSE = 74.55 at 40%), suffering from the systematic violation of the Gaussian residual assumption by 87.9% of variables. Paradoxically, MICE overestimates variance (R_V = 1.031–1.121), trading one form of bias for another.

### 3.2 Effect of Missingness Mechanism

The impact of the missingness mechanism on MissForest performance at 40% missing rate is shown in Figure 3 and Table 3.

**Table 3 — Mechanism-dependent degradation of MissForest performance (40% missing rate)**

| Mechanism | RMSE (Continuous) | F1 (Binary) | F1 (Ordinal) | R_V |
|-----------|-------------------|-------------|--------------|-----|
| MCAR | 39.92 | 0.740 | 0.798 | 0.824 |
| MAR | 63.44 | 0.701 | 0.715 | 0.880 |
| MNAR | 106.81 | — | — | 0.084 |

The transition from MCAR to MAR increases MissForest's RMSE by 58.9% (39.92 → 63.44), consistent with the structural information loss when missingness depends on observed covariates. The further transition to MNAR produces a 167.6% RMSE increase (39.92 → 106.81), reflecting the fundamental violation of the assumption — common to all standard imputation methods — that the information required for imputation resides in the observed data.

At MAR 40%, a noteworthy **performance inversion** occurs: Median/Mode (RMSE = 59.07) marginally surpasses MissForest (63.44). This inversion delineates a critical threshold at which the inter-variable dependency structure has been sufficiently degraded that sophisticated multivariate methods begin incorporating noise rather than signal — the naive median estimator becomes paradoxically more robust.

Under MNAR, all methods converge toward uniformly poor performance (RMSE range: 56.53–57.94 at 20%), and MICE exhibits catastrophic failure at 40% (RMSE = 212.70, R_V = 1,309.5), generating pathologically inflated imputations. This finding underscores the critical message that **under MNAR, no standard imputation method is reliable**, and explicit selection models (Heckman-type, pattern-mixture models) with comprehensive sensitivity analyses are mandatory.

### 3.3 Formal Statistical Comparison

The Friedman test rejected the null hypothesis of method equivalence in every scenario tested (Table 4).

**Table 4 — Friedman test results and post-hoc comparisons (MCAR scenario)**

| Scenario | χ² | p | η² | Best method | CD (α=0.05) |
|----------|-----|-----|-----|-------------|-------------|
| MCAR 10% | 28.16 | 3.4 × 10⁻⁶ | 0.95 | MissForest (R = 1.41) | 0.871 |
| MCAR 20% | 28.53 | 3.1 × 10⁻⁶ | 0.95 | MissForest (R = 1.41) | 0.871 |
| MCAR 40% | 50.26 | < 10⁻⁷ | 0.95 | MissForest (R = 1.41) | 0.871 |
| MAR 10% | 24.19 | 2.2 × 10⁻⁵ | 0.95 | MissForest | 0.871 |
| MAR 20% | 29.90 | 1.2 × 10⁻⁶ | 0.95 | MissForest | 0.871 |
| MAR 40% | 53.98 | < 10⁻⁷ | 0.95 | MissForest | 0.871 |

*R = mean Friedman rank; CD = Nemenyi Critical Difference; η² = Friedman effect size.*

Nemenyi post-hoc tests with Holm correction (MCAR 20%) identified **three homogeneous groups**: {MissForest} alone at the superior extreme, and {Median/Mode, KNN, MICE} forming a cluster of statistically indistinguishable inferior performance. All three comparisons involving MissForest were significant after Holm correction (p < 0.0125). The Critical Difference Diagram (Figure S1) visually confirms this separation: MissForest's mean rank of 1.41 lies more than one CD (0.871) below all other methods.

Effect sizes reinforce the clinical relevance of these differences. MissForest's Cohen's d for RMSE improvement over Median/Mode was +0.398 (moderate effect; Cohen's convention: 0.2 = small, 0.5 = medium, 0.8 = large). Cliff's δ for distribution preservation was 0.180 (small-to-moderate non-parametric effect). The OR for classification accuracy was 3.09, indicating that hypertension status is classified over three times more accurately after MissForest imputation than after Median/Mode imputation.

### 3.4 Multiple Imputation Properties

Rubin's rules were applied to M = 5 bootstrap-based MICE imputations at 20% MCAR (Table 5).

**Table 5 — Rubin's combination rules parameters (MICE, MCAR 20%, 8 representative variables)**

| Variable | W̄ | B | T | RIV | FMI | RE |
|----------|-----|-----|-----|-----|-----|-----|
| Age | 116.26 | 0.64 | 117.03 | 0.007 | 0.338 | 0.937 |
| BMI | 30.89 | 0.18 | 31.11 | 0.007 | 0.338 | 0.937 |
| Waist circumference | 245.29 | 0.65 | 246.07 | 0.003 | 0.336 | 0.937 |
| Systolic BP | 526.70 | 3.37 | 530.74 | 0.008 | 0.338 | 0.937 |
| Diastolic BP | 176.30 | 0.66 | 177.09 | 0.005 | 0.336 | 0.937 |
| Protein | 164.53 | 0.42 | 165.03 | 0.003 | 0.335 | 0.937 |
| HDL-c | 0.063 | <0.001 | 0.063 | 0.004 | 0.336 | 0.937 |
| Total-c | 0.162 | <0.001 | 0.163 | 0.002 | 0.334 | 0.937 |

*W̄ = within-imputation variance; B = between-imputation variance; T = total variance; RIV = relative increase in variance; FMI = fraction of missing information; RE = relative efficiency.*

The between-imputation variance B is systematically negligible relative to the within-imputation variance W̄ (RIV ≈ 0.003–0.008), indicating that **imputation uncertainty contributes minimally to total uncertainty** in this dataset at 20% missing. The FMI of approximately 33.5% reflects the missing data proportion and the strength of inter-variable correlations. The RE of 0.937 confirms that M = 5 imputations recover 93.7% of the asymptotic efficiency (M → ∞), consistent with Rubin's classical result that modest M suffices when FMI is moderate [1]. For missing rates ≥ 40%, M ≥ 10 would be advisable to maintain RE > 0.95.

### 3.5 Clinical Model Preservation and Calibration

The downstream impact of imputation on two clinical prediction models is summarised in Table 6.

**Table 6 — Impact of imputation on clinical prediction models (MCAR 20%)**

| Method | Logistic Regression (HTA) | | | | | | | Linear Regression |
|--------|---------------------------|---|---|---|---|---|---|---|
| | Accuracy | AUC | Brier | OR bias (rel.) | β bias (max) | SE ratio | Δ Sign. (%) | R² (BMI) |
| Reference | — | — | — | 0.000 | 0.000 | 1.000 | 0.0 | — |
| Median/Mode | 0.881 | 0.866 | 0.145 | 0.293 | 1.146 | 0.656 | 48.1 | 0.772 |
| KNN (k=5) | 0.894 | 0.870 | 0.144 | 0.347 | 0.985 | 0.657 | 48.1 | 0.791 |
| MICE (BayesRidge) | 0.840 | 0.788 | 0.184 | 0.529 | 2.055 | 0.764 | 29.6 | 0.662 |
| **MissForest** | **0.958** | **0.941** | **0.087** | **0.153** | **0.425** | **0.821** | **22.2** | **0.825** |

*OR bias (rel.) = mean relative bias in Odds Ratios; β bias (max) = maximum absolute β coefficient bias; SE ratio = ratio of imputed to reference standard errors; Δ Sign. = proportion of predictors whose significance status (p < 0.05) changed.*

MissForest produced the best-calibrated logistic regression model, with an AUC of 0.941 (versus 0.866 for Median/Mode), a Brier score of 0.087 (2.1× lower than MICE's 0.184), and a relative OR bias of only 15.3%. The maximum absolute β coefficient bias of 0.425 was 4.8× lower than MICE's 2.055 — a clinically consequential difference that would substantially alter the interpretation of risk factor magnitudes.

The **significance change rate** quantifies a particularly insidious form of bias: 48.1% of predictors changed their p < 0.05 status after Median/Mode or KNN imputation, versus 22.2% for MissForest. This means that nearly half of all univariate associations would be misclassified as significant or non-significant if naive imputation were employed — a sobering statistic for clinical researchers who perform variable selection after imputation.

Standard errors were systematically underestimated by all methods (SE ratio < 1), with MissForest exhibiting the ratio closest to unity (0.821). This universal underestimation confirms that **single imputation, regardless of algorithm, inflates type I error risk** by treating imputed values as if they were observed — reinforcing the imperative for multiple imputation with proper variance combination.

ROC curves and calibration plots (Figures S2–S3) visually confirm the superior discrimination and calibration of the MissForest-imputed model, with the calibration curve lying closest to the ideal 45° diagonal.

### 3.6 Correlation and Variance Preservation

**Correlation structure.** MissForest preserved pairwise Pearson correlations with a mean absolute deviation of Δr = 0.022, a 2.1-fold improvement over Median/Mode (Δr = 0.047). The advantage was even more pronounced for Kendall's τ (Δτ = 0.019 versus 0.040 for Median/Mode), indicating superior preservation of monotonic but non-linear associations — a critical property for metabolomic data where biomarker relationships are frequently sigmoidal or logarithmic. The correlation difference heatmap (Figure S4) reveals that the largest correlation distortions occur among lipid ratio variables, consistent with their strong mutual dependencies.

**Variance preservation.** The variance ratio R_V deteriorates with increasing missing rate for all methods except MICE, which overestimates variance at all rates (R_V = 1.007–1.121). Median/Mode exhibits the most severe variance collapse: R_V = 0.907 (10%) → 0.812 (20%) → 0.620 (40%), crossing the clinically concerning threshold of R_V < 0.80 between 10% and 20% missing. MissForest maintains R_V within the acceptable range (0.959 → 0.922 → 0.824), with a 14.1% relative variance loss at 40% compared with 35.8% for Median/Mode. The variance ratio trajectories across mechanisms and rates are illustrated in Figure 6.

### 3.7 Sensitivity and Robustness Analyses

**KNN hyperparameter sensitivity.** RMSE varied minimally across k ∈ {3, 5, 10, 15}: 58.83, 57.34, 57.49, and 58.88, respectively. The optimum at k = 5 with a maximum deviation of ±2.7% confirms robust insensitivity to this hyperparameter in the studied range.

**MissForest hyperparameter sensitivity.** Across six configurations (n_estimators ∈ {50, 100} × max_depth ∈ {5, 10, ∞}), RMSE ranged from 32.05 (n = 100, depth = 5) to 33.27 (n = 50, depth = ∞) — a maximum variation of only 3.8%. The optimal configuration (100 trees, max_depth = 5) suggests that regularised, shallower trees improve generalisation, consistent with the bias-variance trade-off in Random Forests. Computation time varied substantially (46.2–97.3 seconds), with unlimited depth configurations being the slowest.

**Outlier robustness.** MICE was the most robust to extreme BMI values (ratio of outlier to normal MAE = 1.96), benefiting from Bayesian shrinkage that pulls extreme predictions toward the mean. Median/Mode was the least robust (ratio = 6.17), as it replaces all missing values — including those of outlier individuals — with the same central tendency estimate.

**Small-sample robustness.** Reducing sample size from 312 to 100 produced divergent effects: Median/Mode RMSE paradoxically decreased from 60.5 to 34.9 (−42%), reflecting the increased stability of the median estimator with fewer, more homogeneous observations. Conversely, MissForest RMSE increased from 32.5 to 41.1 (+26%), consistent with the data-hungry nature of Random Forest algorithms. At N = 100, the performance gap between methods narrowed substantially.

### 3.8 Computational Performance

Median/Mode was essentially instantaneous (13 ms per run). KNN required 90 ms, MICE 3.0 seconds, and MissForest 86.2 seconds on average — approximately 6,600× slower than the baseline. This differential, while acceptable for episodic analyses on moderate-sized datasets (N ≈ 300), becomes prohibitive for large-scale applications (N > 10⁴). The high variance in MissForest runtime (SD = 23.0 s, range: 51.9–113.1 s) reflects sensitivity to the specific missing data pattern, with denser missingness structures requiring more Random Forest fitting iterations.

---

## 4. DISCUSSION

### 4.1 Principal Findings

This Monte Carlo simulation study, grounded in a real-world complete cardiometabolic clinical database, provides robust evidence that **MissForest is the superior imputation method** for datasets characterised by skewed continuous distributions, mixed variable types, and nonlinear inter-variable relationships — a description that fits the majority of clinical biomarker datasets. The superiority is statistically indisputable (Friedman p < 10⁻⁵, η² = 0.95; Nemenyi CD = 0.871, Holm-corrected p < 0.0125 for all comparisons) and clinically consequential (Cohen's d = +0.398; OR for accurate hypertension classification = 3.09; significance change rate reduced from 48.1% to 22.2%).

The magnitude of MissForest's advantage increases with variable complexity: modest for approximately Gaussian variables (Age, Total cholesterol), moderate for skewed continuous biomarkers, and dramatic for ordinal variables (κ = 0.771 versus 0.000 for Median/Mode at 10% MCAR). This gradient is mechanistically interpretable: as the data distribution deviates further from Gaussianity and variable types become more heterogeneous, the assumptions underlying parametric imputation (MICE with BayesianRidge) are progressively violated, while the non-parametric, tree-based architecture of MissForest remains agnostic.

### 4.2 Comparison with the Literature

Our findings are broadly consistent with the seminal work of Stekhoven and Bühlmann [9], who demonstrated MissForest's superiority on mixed-type data, and with Waljee et al. [3], who reported improved laboratory value imputation using Random Forest methods. The present study extends these findings in three important directions.

First, we provide the first direct comparison of MissForest against MICE/BayesianRidge under clinically realistic, highly skewed cardiometabolic distributions — a context in which the parametric MICE variant performs markedly worse than previously reported for approximately Gaussian data [14,15]. The 2.1-fold RMSE advantage of MissForest over MICE in our study contrasts with the near-equivalence reported by Azur et al. [16] for moderately non-normal data, suggesting that the performance gap widens with increasing skewness.

Second, our systematic comparison across all three missingness mechanisms (MCAR, MAR, MNAR) reveals that the **choice of imputation method matters most under MCAR and least under MNAR** — a counterintuitive but logically consistent finding. Under MCAR, the observed data constitute an unbiased sample of the full data, and sophisticated methods can fully exploit inter-variable relationships. Under MNAR, the very information needed for imputation is structurally absent from the observed data, and all methods converge toward similarly poor performance. This finding underscores the danger of applying standard imputation under suspected MNAR without explicit sensitivity analyses [17].

Third, our formal statistical comparison framework (Friedman + Nemenyi + Holm) provides a level of inferential rigour absent from most imputation comparison studies, which typically rely on descriptive comparisons of means without controlling for multiplicity [18]. The identification of a homogeneous inferior group {Median/Mode, KNN, MICE} and an isolated superior method (MissForest) provides clear, statistically defensible guidance for practitioners.

### 4.3 Clinical and Methodological Implications

**Median/Mode imputation should be abandoned in inferential research.** Despite its continued widespread use — a recent systematic review estimated that 30–50% of clinical studies employ mean or median imputation [19] — our data demonstrate that this practice carries unacceptable risks. At 40% MCAR, 89.7% of variables exhibit significant distributional distortion, variance is underestimated by 38%, and nearly half of all predictor significance classifications are erroneous. The deceptive stability of Median/Mode RMSE across missing rates (only +6.3% from 10% to 40%) likely contributes to its persistent popularity: the metric most commonly reported (RMSE) fails to capture the distributional and inferential damage.

**MissForest is the recommended first-line method for cardiometabolic data** with up to 30% missing data and under MCAR or MAR assumptions. It provides the best balance of reconstruction accuracy, distribution and correlation preservation, clinical model fidelity, and calibration. For missing rates exceeding 30%, MissForest remains superior but should be combined with multiple imputation (M ≥ 10) and comprehensive sensitivity analyses.

**MICE deserves methodological rehabilitation.** The poor performance of MICE/BayesianRidge in this study should not be interpreted as an indictment of MICE as a framework. BayesianRidge is among the simplest possible conditional models within the MICE architecture; more sophisticated variants incorporating Predictive Mean Matching, Random Forest, or Gradient Boosting as conditional models (e.g., `miceforest` with LightGBM) would likely substantially improve performance [8]. Our results highlight the critical importance of **estimator selection within MICE**, not merely the choice between MICE and alternative frameworks.

**Single imputation is never sufficient for inference.** The universal underestimation of standard errors (SE ratio < 1 for all methods) confirms that any single imputation — regardless of algorithmic sophistication — treats imputed values as observed and thereby underestimates uncertainty. The Rubin's rules results demonstrate that M = 5 multiple imputations suffice for MCAR/MAR at ≤20% missing (RE = 0.937), but the imperative for multiple imputation remains absolute.

### 4.4 The MNAR Challenge

The MNAR results warrant particular emphasis. The 74% degradation in MissForest RMSE from MCAR to MNAR at 20% missing, and the catastrophic failure of MICE at 40% MNAR (R_V = 1,309.5), demonstrate that **no standard imputation method is MNAR-robust**. When missingness depends on the unobserved value itself — as may occur when patients with extreme biomarker values are less likely to undergo follow-up testing — the fundamental assumption underlying all imputation methods is violated. In such cases, pattern-mixture models, selection models (Heckman-type), or Bayesian approaches explicitly modelling the missingness mechanism are required [2,20]. We strongly recommend that clinical researchers whose missing data pattern suggests MNAR consult with a biostatistician and conduct formal sensitivity analyses varying the assumed missingness mechanism, rather than relying on automated imputation pipelines.

### 4.5 Strengths and Limitations

**Strengths.** This study possesses several methodological strengths: (1) use of a real, complete clinical database as ground truth, avoiding the ecological validity concerns of fully synthetic data; (2) comprehensive coverage of three missingness mechanisms, nine scenarios, and four methods within a unified Monte Carlo framework; (3) evaluation extending beyond reconstruction accuracy to clinical model preservation, calibration, and statistical inference; (4) formal statistical comparisons with multiplicity correction; (5) sensitivity and robustness analyses confirming the stability of conclusions; (6) full computational reproducibility with fixed random seeds and documented software versions.

**Limitations.** Several limitations must be acknowledged. (1) The Monte Carlo design employed M = 5 repetitions; while this provides stable point estimates, M ≥ 100 would be required for robust confidence intervals. (2) The MICE implementation used BayesianRidge as the conditional estimator, which does not represent the state of the art; more flexible estimators (Predictive Mean Matching, tree-based methods) would likely improve MICE performance. (3) MissForest was applied as a single imputation; integration within a multiple imputation framework (bootstrap + Rubin's rules) would be methodologically superior. (4) Results are conditional on the specific correlation structure of a single-centre cardiometabolic cohort; external validation on independent, multicentre databases is essential to establish generalisability (TRIPOD type 4) [13]. (5) The simulated missingness mechanisms, while covering the three canonical types, are more regular than the complex, mixed missingness patterns encountered in real clinical data. (6) Deep learning-based imputation methods (GAIN, VAE, MIWAE) were not included in the comparison. (7) The computational cost of MissForest (≈86 s/run) limits Monte Carlo scalability on standard hardware.

### 4.6 Decision Algorithm

Based on the totality of our results, we propose the following evidence-based decision algorithm for clinical researchers facing missing data:

- **<5% missing per variable**: Complete-case analysis acceptable if N > 30 and MCAR is plausible.
- **5–15% missing**: MissForest recommended for skewed continuous, binary, and ordinal variables. MICE acceptable for approximately Gaussian continuous variables. Multiple imputation (M ≥ 5) strongly advised.
- **15–30% missing**: MissForest + multiple imputation (M ≥ 5). Formal sensitivity analysis mandatory. Compare results with and without imputation.
- **>30% missing**: Alert zone. MissForest + multiple imputation (M ≥ 10). Consider excluding variables with >40% missing. If MNAR suspected, consult a biostatistician; standard imputation methods are unreliable.
- **Quasi-constant variables** (variance ≈ 0): Median/Mode imputation. Multivariate methods introduce artificial noise.
- **Large-scale data** (N > 10⁵): Consider KNN or fast MICE variants (computational trade-off acceptable at low missing rates).

---

## 5. CONCLUSION

In this Monte Carlo simulation study grounded in a complete cardiometabolic clinical database of 312 subjects and 33 variables, MissForest consistently and significantly outperformed Median/Mode, KNN (k = 5), and MICE (BayesianRidge) imputation across all variable types, MCAR and MAR mechanisms, and missing rates from 10% to 40%. The superiority encompassed reconstruction accuracy (RMSE reduced by 46% versus Median/Mode at 20% MCAR), distribution preservation (89.7% versus 0.0% KS-significant variables at 10% MCAR), correlation structure maintenance (Δr = 0.022 versus 0.047), clinical model fidelity (AUC = 0.941 versus 0.866), and calibration (Brier = 0.087 versus 0.145). Formal statistical testing (Friedman p < 10⁻⁵, Nemenyi CD = 0.871, Holm-corrected post-hoc comparisons) confirms that these differences are both statistically significant and clinically consequential.

Median/Mode imputation — still the default approach in a substantial proportion of clinical studies — should be abandoned in inferential research contexts due to severe variance underestimation (R_V = 0.620 at 40% MCAR), massive distributional distortion, and a 48.1% predictor significance misclassification rate.

Under MNAR, no standard imputation method is reliable; the RMSE of MissForest increases by 74% relative to MCAR, and MICE exhibits catastrophic variance inflation (R_V = 1,309.5 at 40% MNAR). Explicit selection models with formal sensitivity analyses are mandatory when MNAR is suspected.

Future work should: (1) integrate MissForest within a formal multiple imputation framework (bootstrap + Rubin's rules); (2) benchmark against deep learning-based imputation methods (GAIN, MIWAE); (3) validate findings on independent, multicentre cardiometabolic cohorts; (4) extend the simulation framework to survival outcomes (Cox proportional hazards models with imputed covariates); and (5) develop user-friendly, open-source software implementing the recommended decision algorithm for clinical researchers.

---

## REFERENCES

1. Rubin DB. *Multiple Imputation for Nonresponse in Surveys*. New York: Wiley; 1987.
2. Little RJA, Rubin DB. *Statistical Analysis with Missing Data*. 3rd ed. Hoboken: Wiley; 2019.
3. Waljee AK, Mukherjee A, Singal AG, et al. Comparison of imputation methods for missing laboratory data in medicine. *BMJ Open*. 2013;3(8):e002847.
4. Sterne JAC, White IR, Carlin JB, et al. Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls. *BMJ*. 2009;338:b2393.
5. Donders ART, van der Heijden GJMG, Stijnen T, Moons KGM. Review: a gentle introduction to imputation of missing values. *J Clin Epidemiol*. 2006;59(10):1087-1091.
6. Jakobsen JC, Gluud C, Wetterslev J, Winkel P. When and how should multiple imputation be used for handling missing data in randomised clinical trials — a practical guide with flowcharts. *BMC Med Res Methodol*. 2017;17(1):162.
7. van Buuren S. *Flexible Imputation of Missing Data*. 2nd ed. Boca Raton: CRC Press; 2018.
8. Azur MJ, Stuart EA, Frangakis C, Leaf PJ. Multiple imputation by chained equations: what is it and how does it work? *Int J Methods Psychiatr Res*. 2011;20(1):40-49.
9. Stekhoven DJ, Bühlmann P. MissForest—non-parametric missing value imputation for mixed-type data. *Bioinformatics*. 2012;28(1):112-118.
10. Troyanskaya O, Cantor M, Sherlock G, et al. Missing value estimation methods for DNA microarrays. *Bioinformatics*. 2001;17(6):520-525.
11. Josse J, Prost N, Scornet E, Varoquaux G. On the consistency of supervised learning with missing values. *arXiv*. 2019:1902.06931.
12. Pedersen AB, Mikkelsen EM, Cronin-Fenton D, et al. Missing data and multiple imputation in clinical epidemiological research. *Clin Epidemiol*. 2017;9:157-166.
13. Moons KGM, Altman DG, Reitsma JB, et al. Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): explanation and elaboration. *Ann Intern Med*. 2015;162(1):W1-W73.
14. Marshall A, Altman DG, Holder RL, Royston P. Combining estimates of interest in prognostic modelling studies after multiple imputation: current practice and guidelines. *BMC Med Res Methodol*. 2009;9:57.
15. White IR, Royston P, Wood AM. Multiple imputation using chained equations: issues and guidance for practice. *Stat Med*. 2011;30(4):377-399.
16. Azur MJ, Stuart EA, Frangakis C, Leaf PJ. Multiple imputation by chained equations: what is it and how does it work? *Int J Methods Psychiatr Res*. 2011;20(1):40-49.
17. Carpenter JR, Kenward MG. *Multiple Imputation and its Application*. Chichester: Wiley; 2013.
18. Demsar J. Statistical comparisons of classifiers over multiple data sets. *J Mach Learn Res*. 2006;7:1-30.
19. Eekhout I, de Boer RM, Twisk JWR, de Vet HCW, Heymans MW. Missing data: a systematic review of how they are reported and handled. *Epidemiology*. 2012;23(5):729-732.
20. Enders CK. *Applied Missing Data Analysis*. New York: Guilford Press; 2010.

---

## FIGURE LEGENDS

**Figure 1 — Methodological flowchart.** Schematic representation of the complete simulation pipeline: from original database (N = 312, P = 33) through quality audit, MCAR/MAR/MNAR amputation (10%, 20%, 40%), Monte Carlo repetition (M = 5), four imputation methods, multi-metric evaluation, clinical impact assessment, statistical comparison, and visualisation. [`figures/01_flowchart.png`]

**Figure 2 — Missing data heatmap (MCAR 20%, first 50 patients).** Binary representation of the missing data pattern: white = observed, red = missing. The uniform distribution of missing cells confirms the MCAR mechanism. [`figures/02_heatmap_missing.png`]

**Figure 3 — RMSE trajectories by missing rate and mechanism.** Mean RMSE (across 29 continuous variables) as a function of missing rate (10%, 20%, 40%) for each imputation method, stratified by mechanism (MCAR, MAR). MissForest (solid green) maintains the lowest RMSE across all scenarios. [`figures/03_rmse_curves.png`]

**Figure 4 — Comparative heatmap (MCAR 40%, continuous variables).** RMSE values for each variable × method combination. Darker colours indicate higher error. MissForest (rightmost column) consistently shows the lightest shading. [`figures/04_heatmap_methods.png`]

**Figure 5 — Radar plot of multi-dimensional performance (MCAR).** Normalised performance across six dimensions: inverse RMSE, F1-score, variance ratio preservation, Cohen's κ, accuracy, and inverse computation time. Larger polygon area indicates better overall performance. [`figures/05_radar_plot.png`]

**Figure 6 — Variance ratio preservation.** Ratio of imputed to original variance (R_V) as a function of missing rate, by method and mechanism. The horizontal dashed line at R_V = 1.0 marks perfect preservation. Median/Mode variance collapses at 40% (R_V = 0.62); MICE overestimates (R_V > 1.0); MissForest maintains the best balance (0.82 ≤ R_V ≤ 0.96). [`figures/08_variance_ratio.png`]

**Figure 7 — Boxplot of RMSE distributions (MCAR, all rates).** Distribution of RMSE values across 29 continuous variables, stratified by method and missing rate. Boxplots illustrate the median, interquartile range, and variability of imputation errors. [`figures/07_boxplot_rmse.png`]

**Figure 8 — Clinical decision algorithm for imputation in cardiometabolic research.** Evidence-based decision tree derived from the simulation results, guiding method selection based on missing data rate, variable type, and suspected missingness mechanism. [`figures/06_decision_tree.png`]

---

## SUPPLEMENTARY FIGURE LEGENDS

**Figure S1 — Critical Difference Diagram (Friedman + Nemenyi, MCAR 20%).** Methods connected by a horizontal bar are not significantly different at α = 0.05. MissForest (rank = 1.41) is isolated from the homogeneous group {KNN, Median/Mode, MICE}. CD = 0.871. [`figures_v3/critical_difference_diagram.png`]

**Figure S2 — ROC curves for hypertension prediction (MCAR 20%).** ROC curves from logistic regression models fitted after imputation by each method. AUC values: MissForest (0.941), KNN (0.870), Median/Mode (0.866), MICE (0.788). [`figures_v3/roc_curves.png`]

**Figure S3 — Calibration curves (MCAR 20%).** Observed versus predicted probabilities of hypertension, by imputation method. The diagonal represents perfect calibration. MissForest's curve lies closest to the diagonal. [`figures_v3/calibration_curves.png`]

**Figure S4 — Correlation difference heatmap (MissForest, MCAR 20%).** Difference between original and imputed Pearson correlation matrices (Δr = r_orig − r_imp). Red-blue colour scale centred at zero. [`figures_v3/correlation_diff_heatmap.png`]

**Figure S5 — Bland-Altman plot for BMI (MCAR 20%).** Difference (imputed − original) versus mean of original and imputed BMI values, by method. Horizontal lines indicate mean bias and 95% limits of agreement. [`figures_v3/bland_altman_imc.png`]

**Figure S6 — Violin plot of normalised imputation errors (MCAR 20%).** Distribution of standardised errors (imputed − true) / σ for each method. Wider sections indicate higher probability density. [`figures_v3/violin_errors.png`]

**Figure S7 — Forest plot of log Odds Ratios for hypertension (MCAR 20%).** Log-transformed Odds Ratios for the eight strongest predictors of hypertension, estimated after imputation by each method. [`figures_v3/forest_plot_or.png`]

**Figure S8 — QQ-plot for BMI imputation (MCAR 20%).** Quantile-quantile plot comparing imputed versus true BMI values for missing observations only. The red diagonal represents perfect agreement. [`figures_v3/qq_plot_imc.png`]

**Figure S9 — Convergence curves for MICE and MissForest (MCAR 20%).** RMSE as a function of the number of iterations for MICE/BayesianRidge (left) and MissForest (right). Both methods converge within 5 iterations. [`figures_v3/convergence_curves.png`]

---

**Data availability**: The complete clinical database, all simulation code (`simulation_v2.py`, `supplement_v3.py`), aggregated Monte Carlo results, and all figures (300 DPI) are available in the accompanying repository.

**Competing interests**: None declared.

**Funding**: This research was conducted as part of a doctoral thesis. No specific external funding was received.

**Author contributions**: Conceptualisation, methodology, software development, formal analysis, investigation, visualisation, and writing — original draft were performed as part of the doctoral research programme.
