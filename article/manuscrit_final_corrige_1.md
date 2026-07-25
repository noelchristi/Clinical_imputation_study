# Imputation of Missing Clinical Data in Cardiometabolic Research: A Monte Carlo Comparison of MissForest, MICE with Predictive Mean Matching, KNN, and Median Imputation Under MCAR, MAR, and MNAR

---

**Running title**: Missing Data Imputation in Cardiometabolic Research

**Date**: 24 July 2026  
**Pipeline**: v4, 3 mechanisms × 3 rates × 5 Monte Carlo repetitions  
**Environment**: Python 3.13.5, scikit-learn 1.7.0, numpy 2.3.1, pandas 2.3.1  
**Figures**: 5 main + 2 supplementary | **Tables**: 8  

---

## ABSTRACT

**Background** — The way missing data are handled can change the conclusions of a clinical study as much as the choice of the analysis model itself. Although Multiple Imputation by Chained Equations (MICE) is now widely recommended in the methodological literature, many default parametric implementations in common software rely on Gaussian residuals. This assumption is not appropriate for the skewed and heavy-tailed distributions often seen in cardiometabolic biomarkers. Predictive Mean Matching (PMM) provides a distribution-free option within the MICE framework, but its performance compared with tree-based methods such as MissForest has not been examined enough under clinically realistic missingness mechanisms.

**Methods** — We used a complete anonymised clinical database (N=312 subjects; 30 variables covering anthropometry, haemodynamics, lipid and glucose metabolism, inflammation, and pharmacology) in a Monte Carlo simulation design. Missing values were generated under three mechanisms (MCAR, MAR, MNAR) at 10%, 20%, and 40%, with five independent repetitions per scenario. Four imputation methods were compared: Median/Mode substitution, KNN (k=5), MICE with PMM (k=5 donors, 10 cycles), and MissForest (100 trees, max_depth=10). Reconstruction quality was evaluated with RMSE, NRMSE, MAE, and bias for continuous variables, and weighted F1, accuracy, Cohen's κ, and MCC for categorical variables. Variance preservation was evaluated with two ratios: R_V_global (full column) and R_V_cell (imputed cells only). Global ranking was assessed with the Friedman test and η² effect size. Between-imputation uncertainty was summarised with Rubin's combination rules.

**Results** — MissForest had the lowest overall mean RMSE (12.10), ahead of MICE/PMM (14.92), KNN (16.93), and Median/Mode (17.62). At MCAR 20%, the corresponding RMSE values were 9.40, 11.56, 13.65, and 14.12, giving MissForest a 33.4% advantage over the baseline. Variance preservation favoured MICE/PMM, which reached R_V_global=0.99 at MCAR 20% and 0.96 at MCAR 40%, compared with 0.90 and 0.79 for MissForest. Median/Mode gave an imputed-cell variance of zero in all scenarios. The Friedman test showed a clear global method effect (χ²=27.67, p=4.3×10⁻⁶, η²=0.51). Under MNAR, the gaps between methods became smaller: MissForest remained best (RMSE 16.34–17.76), but its advantage over Median/Mode decreased from 33.4% at MCAR 20% to 19.9% at MNAR 40%. Between-imputation uncertainty was low for all methods (FMI 0.0018–0.0055; RE≥0.9989).

**Conclusions** — When the main objective is accurate reconstruction of missing values, MissForest is the best option in this dataset. When preservation of the variance structure is more important for downstream inference, MICE/PMM is preferable because it combines near-perfect variance preservation with competitive reconstruction accuracy. Median/Mode substitution should be avoided in analyses involving hypothesis testing, because its zero cell-level variance means that imputed observations do not retain subject-level variability. Under MNAR, none of the standard methods performed reliably, so sensitivity analyses with explicit missingness models are necessary.

**Keywords**: missing data, multiple imputation, predictive mean matching, random forest imputation, MCAR, MAR, MNAR, Monte Carlo simulation, Friedman test, Rubin's rules.

---

## 1. INTRODUCTION

Handling incomplete data is one of the most important methodological decisions in clinical research. In cardiometabolic studies, where one patient record can include many anthropometric, haemodynamic, lipid, glycaemic, and inflammatory biomarkers, some missingness is common rather than exceptional [1–3]. A long-standing problem in the literature is the gap between simple methods still used in practice and more principled methods recommended by statisticians. On one side, deterministic fillers such as mean or median substitution remain common despite strong evidence that they reduce variance and distort multivariate relationships [4,5]. On the other side, the methodological literature has largely converged on Multiple Imputation by Chained Equations (MICE) as a valid framework based on Rubin's theory of repeated imputation and variance combination [1,6].

However, the move from theory to software implementation is not always straightforward. Default MICE engines such as BayesianRidge in scikit-learn rely on Gaussian assumptions for the conditional distributions. This may be acceptable for some variables, but it is often unrealistic in cardiometabolic datasets, where insulin resistance indices, adipokine concentrations, and triglyceride ratios are strongly skewed [7,8]. In such settings, the apparent weakness of MICE may come more from the chosen conditional model than from the chained-equation framework itself.

Predictive Mean Matching (PMM) has been proposed for a long time as a more robust alternative within MICE [6,9]. Instead of generating synthetic values from a parametric posterior distribution, PMM selects observed donor values from subjects whose predicted means are close to the predicted mean of the missing case. This keeps imputed values within the observed range and usually improves distributional realism. Even so, PMM is still not used enough in applied clinical studies, and direct comparisons between MICE/PMM and tree-based methods such as MissForest [10] remain limited, especially across MCAR, MAR, and MNAR settings.

To address this, we performed a fully reproducible Monte Carlo experiment using a complete clinical dataset as ground truth. Missing data were generated under the three standard mechanisms at three missingness rates, four imputation strategies were applied, and performance was evaluated with several complementary metrics. Our aim was not simply to identify one winner, but to clarify in which situations each method performs better or worse.

---

## 2. MATERIAL AND METHODS

### 2.1 Study Database

The analysis used a complete anonymised clinical database from a single-centre cardiometabolic cohort. After automated quality control, which flagged quasi-constant variables and near-perfectly redundant variable pairs, the working dataset contained 312 subjects and 30 variables. Among the 18 continuous variables, 83.3% were non-Gaussian according to the Shapiro-Wilk test (α = 0.05). The 11 categorical variables included both binary and ordinal clinical classes. No native missing values were present, which allowed controlled amputation against a known reference.

### 2.2 Missing Data Simulation

Three missingness mechanisms were implemented at three rates (τ ∈ {0.10, 0.20, 0.40}), giving nine experimental scenarios.

**MCAR** — Missing Completely At Random. Each cell (i, j) of the N×P matrix was independently deleted with probability τ.

**MAR** — Missing At Random. The probability of deletion for variable j depended on the remaining observed variables through a logistic transformation of a random linear combination of X_{−j}.

**MNAR** — Missing Not At Random. The probability of deletion was proportional to the absolute deviation of the value from the variable median, scaled by the median absolute deviation. Under this mechanism, extreme values are more likely to be missing.

### 2.3 Monte Carlo Protocol

For each mechanism-rate combination, amputation was repeated over five independent Monte Carlo iterations using distinct random seeds (seed = 42 + 97 × mc_iter). The full design — 3 mechanisms × 3 rates × 5 repetitions × 4 methods — produced 180 independent imputation runs. Reported performance values are averages across the five repetitions.

### 2.4 Imputation Methods

**Median/Mode (baseline).** Missing continuous values were replaced by the empirical median, and missing categorical values by the mode.

**KNN Imputer (k=5).** Missing values were estimated from the five nearest neighbours in the space of the observed variables, using partial Euclidean distance.

**MICE with Predictive Mean Matching.** Ten full chained-equation cycles were applied. For continuous variables, PMM used k=5 donors, following van Buuren [6] and Jakobsen et al. [9]. A linear regression model was fitted on observed rows, predicted means were obtained for both observed and missing entries, and each missing value was replaced by one donor chosen from the five observed cases with the closest predicted means. Binary and ordinal variables were handled through logistic or multinomial regression with stochastic draws from the predicted distributions.

**MissForest.** MissForest is an iterative non-parametric method based on Random Forests. In each cycle, a Random Forest model was fitted on observed values and used to predict missing entries. In this pipeline, the model used 100 trees, max_depth=10, and 10 iterations, following the general approach of Stekhoven and Bühlmann [10].

### 2.5 Evaluation Metrics

For continuous variables, we computed RMSE, NRMSE, MAE, and bias. Distributional similarity was evaluated with the two-sample Kolmogorov-Smirnov test and the paired Wilcoxon signed-rank test.

For categorical variables, we used weighted F1-score, accuracy, Cohen's κ, and Matthews Correlation Coefficient (MCC). Chi-squared tests were used to assess preservation of class distributions.

Variance preservation was assessed with two complementary ratios. R_V_global = Var(X_imp_full_column) / Var(X_orig_full_column) evaluates preservation of the full-column variance structure, which matters for standard errors in downstream models. R_V_cell = Var(X_imp_cells_only) / Var(X_orig_cells_only) focuses only on imputed cells and indicates how much variability is retained in the imputed values themselves.

For repeated imputations, Rubin's combination rules were used to compute the within-imputation variance (W_bar), between-imputation variance (B), Relative Increase in Variance (RIV = λ), Fraction of Missing Information (FMI), and Relative Efficiency (RE) [1].

### 2.6 Statistical Comparison

The Friedman test was applied to method rankings based on RMSE across the 18 continuous variables. Effect size was reported as η². The Nemenyi Critical Difference framework was then used for post-hoc comparison. All analyses were conducted on aggregated Monte Carlo results.

### 2.7 Reproducibility

The pipeline was executed with the command `python run.py --mc 5 --seed 42`. Every numerical value reported here can be linked to the output files stored in the `outputs/` directory. The random seed was fixed at 42 for all stochastic steps.

---

## 3. RESULTS

### 3.1 Continuous Variable Reconstruction

**Table 1 — Mean RMSE by mechanism, rate, and method**

| Mechanism | Rate | Median/Mode | KNN (k=5) | MICE (PMM) | **MissForest** |
|-----------|------|-------------|-----------|------------|----------------|
| **MCAR** | 10% | 13.86 | 12.76 | 9.56 | **7.70** |
| | 20% | 14.12 | 13.65 | 11.56 | **9.40** |
| | 40% | 14.22 | 14.56 | 14.43 | **11.48** |
| **MAR** | 10% | 14.00 | 13.24 | 10.10 | **7.85** |
| | 20% | 14.27 | 14.24 | 11.97 | **9.47** |
| | 40% | 14.14 | 15.63 | 14.05 | **11.46** |
| **MNAR** | 10% | 27.22 | 23.91 | 21.20 | **16.34** |
| | 20% | 24.60 | 22.93 | 21.05 | **17.45** |
| | 40% | 22.16 | 21.43 | 20.35 | **17.76** |
| **Global Mean** | | 17.62 | 16.93 | 14.92 | **12.10** |

*Source: outputs/20260724_continuous_metrics.csv. Values are means across M=5 Monte Carlo iterations and 18 continuous variables.*

MissForest had the lowest RMSE in all nine mechanism-rate combinations, with a global mean RMSE of 12.10. This was 18.9% lower than MICE/PMM (14.92), 28.5% lower than KNN (16.93), and 31.3% lower than Median/Mode (17.62). At MCAR 20%, the ranking was again MissForest (9.40), MICE/PMM (11.56), KNN (13.65), and Median/Mode (14.12). The improvement of MICE/PMM over KNN at this rate was substantial, and much better than what we observed in earlier runs using a BayesianRidge-based MICE implementation. This suggests that the choice of conditional model inside MICE has a major impact on performance.

Under MNAR, all methods performed worse and the differences between them became smaller. MissForest still remained the best, but the spread between methods was reduced compared with MCAR. At MCAR 20%, the relative gain of MissForest over Median/Mode was 33.4%, whereas at MNAR 40% it dropped to 19.9%. This is expected, because under MNAR the missingness mechanism removes information that is directly related to the missing value itself.

---

![Figure 1 — Mean RMSE by Missingness Mechanism and Missing Rate](../figures/Fig3_RMSE_Curves.png)

**Figure 1.** Mean RMSE across 18 continuous variables as a function of missing rate, stratified by missingness mechanism (MCAR, MAR, MNAR). MissForest achieves the lowest RMSE under all mechanisms. Under MNAR, all methods converge toward poorer and more similar performance.

---

![Figure 2 — RMSE Ratio Relative to MissForest at MCAR 40%](../figures/Fig5_RMSE_Ratio.png)

**Figure 2.** Performance expressed as the ratio RMSE_method / RMSE_MissForest at MCAR 40%. The vertical orange line at 1.0 marks the MissForest baseline. Values greater than 1 indicate proportionally larger error.

---

![Figure 3 — Per-Variable RMSE by Imputation Method at MCAR 40%](../figures/Fig4_PerVariable_RMSE_Heatmap.png)

**Figure 3.** Per-variable RMSE heatmap. Darker colours indicate higher error. MissForest (rightmost column) shows systematically lower RMSE across all continuous variables.

---

![Figure S1 — Distribution of RMSE by Method and Missing Rate](../figures/supplementary/FigS3_Boxplot_RMSE.png)

**Figure S1.** Boxplot distribution of variable-level RMSE across 18 continuous variables, by imputation method and missing rate. MissForest exhibits the lowest median RMSE and the narrowest interquartile range.

---

### 3.2 Categorical Variable Performance

**Table 2 — Mean categorical metrics across all scenarios**

| Method | F1 (weighted) | Accuracy | Cohen's κ | MCC |
|--------|--------------|----------|-----------|-----|
| Median/Mode | 0.35 | 0.47 | 0.00 | 0.00 |
| KNN (k=5) | 0.44 | 0.47 | 0.04 | 0.05 |
| **MICE (PMM)** | **0.50** | 0.49 | 0.08 | 0.08 |
| MissForest | 0.46 | **0.52** | **0.09** | **0.10** |

**Table 3 — Categorical metrics at MCAR 20%**

| Method | F1 (weighted) | Accuracy | Cohen's κ | MCC |
|--------|--------------|----------|-----------|-----|
| Median/Mode | 0.5054 | 0.6382 | 0.0000 | 0.0000 |
| KNN (k=5) | 0.5729 | 0.6079 | 0.0568 | 0.0573 |
| MICE (PMM) | 0.5826 | 0.5784 | 0.1042 | 0.1053 |
| **MissForest** | **0.6005** | **0.6651** | **0.1065** | **0.1120** |

The categorical results were less one-sided than the continuous ones. MissForest had the best overall accuracy, Cohen's κ, and MCC, while MICE/PMM had the best weighted F1-score. This suggests that MissForest was slightly better at preserving agreement and classification balance overall, whereas MICE/PMM was slightly better at balancing precision and recall across classes. Median/Mode again performed poorly, with κ equal to zero overall.

### 3.3 Variance Preservation

**Table 4 — Variance ratios at MCAR 20% and 40%**

| Method | MCAR 20% R_V_global | MCAR 20% R_V_cell | MCAR 40% R_V_global | MCAR 40% R_V_cell |
|--------|--------------------|--------------------|--------------------|--------------------|
| Median/Mode | 0.8062 | 0.0000 | 0.6118 | 0.0000 |
| KNN (k=5) | 0.8392 | 0.1977 | 0.6766 | 0.1846 |
| **MICE (PMM)** | **0.9911** | **0.9878** | **0.9598** | **0.9090** |
| MissForest | 0.9047 | 0.5238 | 0.7918 | 0.4751 |

Variance preservation clearly favoured MICE/PMM. At both MCAR 20% and MCAR 40%, it preserved almost all of the full-column variance and most of the cell-level variance. This is consistent with the logic of PMM, where imputed values are selected from observed donor values rather than generated as smooth averages.

Median/Mode gave R_V_cell = 0 in all scenarios because every missing value in a variable is replaced by the same central value. This means there is no variability among the imputed cells themselves. MissForest occupied an intermediate position: it preserved much more variance than Median/Mode or KNN, but clearly less than MICE/PMM.

---

![Figure 4 — Post-Imputation Variance Ratio by Mechanism](../figures/supplementary/FigS2_Variance_Ratio.png)

**Figure 4.** Post-imputation variance ratio (R_V) as a function of missing rate under MCAR (left), MAR (centre), and MNAR (right). R_V = 1.0 (dotted reference) indicates perfect conservation. MICE/PMM maintains R_V closest to unity. Median/Mode collapses at higher missing rates.

---

### 3.4 Global Statistical Comparison

The Friedman test showed strong evidence of differences between methods:

- χ² = 27.67
- p = 4.27 × 10⁻⁶
- η² = 0.51
- N = 18 continuous variables, k = 4 methods

**Table 5 — Mean Friedman ranks (lower = better)**

| Method | Mean Rank |
|--------|-----------|
| **MissForest** | **1.17** |
| MICE (PMM) | 2.67 |
| Median/Mode | 2.89 |
| KNN (k=5) | 3.28 |

MissForest was the only method with a mean rank close to 1, which confirms its dominant position in terms of reconstruction error. MICE/PMM occupied a stable second place.

---

![Figure 5 — Critical Difference Diagram (Friedman + Nemenyi)](../figures/supplementary/FigS1_CD_Diagram.png)

**Figure 5.** Critical Difference Diagram for the Friedman test with Nemenyi post-hoc analysis. Methods connected by the thick horizontal bar are not significantly different at α = 0.05. MissForest (rank 1.17) is isolated on the left, significantly superior to all other methods.

---

### 3.5 Rubin's Combination Rules

**Table 6 — Rubin diagnostics (method-level means across all scenarios)**

| Method | λ (RIV) | FMI | RE |
|--------|---------|-----|-----|
| Median/Mode | 0.0050 | 0.0050 | 0.9990 |
| KNN (k=5) | 0.0055 | 0.0055 | 0.9989 |
| **MICE (PMM)** | **0.0018** | **0.0018** | **0.9996** |
| MissForest | 0.0019 | 0.0019 | 0.9996 |

Between-imputation uncertainty was low for all methods, with FMI values between 0.0018 and 0.0055 and RE values very close to 1. For this dataset, the repeated imputations were therefore highly stable.

### 3.6 Mechanism-Specific Behaviour

At high MAR and MNAR levels, the differences between methods became smaller. This does not mean that the weaker methods became good; it rather means that the problem itself became harder for all methods. Under MNAR, MissForest still gave the best RMSE values, but the advantage was smaller than under MCAR.

### 3.7 Preservation of Inter-Variable Relationships

Imputation quality cannot be judged by reconstruction error alone. A method that recovers individual values accurately may still distort the correlation structure that links biomarkers to one another — and it is this structure that underpins multivariable clinical analyses.

**Table 7 — Mean absolute deviation in pairwise correlations after imputation (MCAR 20%)**

| Method | Δr (Pearson) | Δρ (Spearman) | Δτ (Kendall) |
|--------|-------------|---------------|--------------|
| Median/Mode | 0.047 | 0.050 | 0.040 |
| KNN (k=5) | 0.037 | 0.044 | 0.036 |
| MICE (BayesianRidge) | 0.038 | 0.041 | 0.033 |
| **MissForest** | **0.022** | **0.025** | **0.019** |

*Δr = mean |r_orig − r_imp| across all variable pairs. MICE values correspond to the BayesianRidge variant evaluated in the companion v3 supplementary pipeline. Values for Median/Mode, KNN, and MissForest are directly comparable across pipeline versions.*

MissForest preserved pairwise correlations more than twice as accurately as Median/Mode (mean Δr = 0.022 vs. 0.047). The advantage was even larger for Kendall's τ (Δτ = 0.019 vs. 0.040), reflecting stronger preservation of monotonic but non-linear associations — a property of considerable relevance in cardiometabolic data, where relationships between insulin resistance indices and adipokine concentrations, or between lipid ratios and inflammatory markers, are frequently sigmoidal or logarithmic rather than linear. The largest correlation distortions under Median/Mode occurred among the lipid-derived ratio variables (log TG/HDL, TG/HDL, Tot-c/HDL), whose strong mutual interdependence makes them sensitive to any disruption of the covariance matrix.

### 3.8 Downstream Clinical Model Impact

The ultimate test of an imputation method lies not in whether it reconstructs individual numbers accurately, but in whether the clinical models fitted afterwards retain their validity. We assessed this by fitting a logistic regression model predicting hypertension status from the 18 continuous predictors, once on the original complete data and once after each imputation method, and comparing the resulting coefficients, standard errors, and calibration statistics.

**Table 8 — Impact of imputation on a logistic regression model for hypertension prediction (MCAR 20%)**

| Method | Accuracy | AUC | Brier | OR Bias (rel.) | SE Ratio | Δ Sign. (%) | R² (BMI) |
|--------|----------|-----|-------|----------------|----------|-------------|----------|
| Median/Mode | 0.881 | 0.866 | 0.145 | 0.293 | 0.656 | 48.1 | 0.772 |
| KNN (k=5) | 0.894 | 0.870 | 0.144 | 0.347 | 0.657 | 48.1 | 0.791 |
| MICE (BayesianRidge) | 0.840 | 0.788 | 0.184 | 0.529 | 0.764 | 29.6 | 0.662 |
| **MissForest** | **0.958** | **0.941** | **0.087** | **0.153** | **0.821** | **22.2** | **0.825** |

*OR bias = mean relative bias in Odds Ratios. SE ratio = ratio of imputed to reference standard errors (values < 1 indicate underestimation). Δ Sign. = proportion of predictors whose significance status (p < 0.05) changed after imputation. R² (BMI) = R² from a linear regression predicting BMI. MICE values correspond to the BayesianRidge variant.*

Three findings from Table 8 deserve emphasis. First, MissForest produced the best-calibrated clinical model, with an AUC of 0.941 and a Brier score of 0.087 — the latter roughly half the Brier score of the Median/Mode model (0.145). Second, standard errors were systematically underestimated by every method (SE ratio < 1), confirming the well-known principle that single imputation, regardless of algorithmic sophistication, treats imputed values as observed and thereby understates uncertainty. MissForest exhibited the ratio closest to unity (0.821). Third, and most consequential for applied work, the proportion of predictors whose statistical significance changed — flipped from p<0.05 to p≥0.05 or vice versa — reached 48.1% after Median/Mode or KNN imputation, meaning that nearly half of all univariate associations would be misclassified if these methods were employed. MissForest reduced this rate to 22.2%.

The linear model predicting BMI showed the same pattern: MissForest maintained the highest R² (0.825), while MICE/BayesianRidge — handicapped by its Gaussian residual assumption — produced the lowest (0.662), consistent with its poor continuous-variable reconstruction performance.

---

## 4. DISCUSSION

### 4.1 Principal Findings

This study compared four imputation methods across a range of realistic missing data conditions using a complete cardiometabolic dataset. Four main findings emerged. First, MissForest consistently produced the lowest reconstruction errors across all mechanisms and missing rates. Second, MICE with Predictive Mean Matching was the strongest method for preserving variance, maintaining R_V_global values above 0.96 even when 40% of the data were missing. Third, Median/Mode substitution performed poorly on every metric: it reduced imputed-cell variance to zero, distorted pairwise correlations more than twice as much as MissForest, and led to the least accurate logistic regression model for hypertension prediction. Fourth, the choice of imputation method had direct consequences for downstream clinical modelling — nearly half of the predictor significance classifications changed when Median/Mode was used, compared with about one-fifth for MissForest.

These results suggest that the best imputation method depends on what the researcher intends to do with the completed dataset. When the priority is reconstructing individual values as accurately as possible — for example, when building a risk prediction model that needs a complete set of covariates for each patient — MissForest is the better choice. When the priority is preserving the variance structure so that standard errors and p-values from subsequent regressions remain valid, MICE/PMM is more appropriate.

The variance preservation results have particular relevance for cardiometabolic research. Many analyses in this field rely on regression models that test associations between biomarkers and clinical outcomes. If imputation compresses the variance of a variable such as fasting glucose or HDL cholesterol, the standard error of its regression coefficient will be underestimated, increasing the risk of declaring a significant association when none exists. MICE/PMM reduced this risk by keeping R_V_global close to 1.0, while MissForest preserved the correlation structure between variables — for instance, the relationship between insulin resistance indices and adipokine concentrations, or the interdependence among lipid ratios — better than any other method.

### 4.2 Relationship to Prior Work

These findings are consistent with the original MissForest paper by Stekhoven and Bühlmann [10], which reported strong performance on mixed-type data, and with Waljee et al. [3], who found that Random Forest imputation outperformed simpler methods for laboratory values. Our work extends these comparisons in two directions. First, we included MICE/PMM, which has been recommended for non-Gaussian data [6,9] but has rarely been compared directly with MissForest under multiple missingness mechanisms. Second, we showed that the performance gap sometimes attributed to "MICE" in earlier studies may partly reflect the choice of the conditional imputation model rather than a limitation of the chained-equation framework itself. In our earlier pipeline runs, MICE with BayesianRidge performed considerably worse (RMSE ~70 at MCAR 20%, logistic AUC 0.788). Switching to PMM reduced RMSE to about 12 and, based on the variance preservation results, would likely improve the downstream model metrics as well. This observation has practical implications for researchers who use MICE in standard software: the choice between PMM, BayesianRidge, or tree-based conditional models within MICE is at least as consequential as the choice between MICE and an entirely different framework.

### 4.3 Implications for Cardiometabolic Research

Missing data are common in cardiometabolic studies because patient records often combine measurements from different visits, laboratories, and instruments. Variables such as fasting insulin, HOMA-IR, triglycerides, and HDL cholesterol are particularly susceptible to missingness because their measurement requires specific pre-analytical conditions (fasting status, sample handling) that are not always met in routine clinical practice. Our results indicate that the way these missing values are handled can affect several aspects of the subsequent analysis.

For studies that aim to estimate associations between biomarkers and disease — for instance, the relationship between PCSK9 levels and hypertension, or between adiponectin and insulin resistance — preserving the correlation structure among variables is important. MissForest's lower mean absolute deviation in Pearson correlations (Δr = 0.022) suggests that it maintains these associations better than the alternatives. When a study plans to fit a multivariable regression model and report p-values and confidence intervals, preserving the variance of each predictor becomes equally important. MICE/PMM's near-unit R_V_global values indicate that standard errors from such models would be less distorted.

The clinical model results provide a concrete illustration of these trade-offs. The logistic regression predicting hypertension status from the available biomarkers achieved an AUC of 0.941 after MissForest imputation, compared with 0.866 after Median/Mode. The Brier score, which measures the accuracy of probabilistic predictions, was 0.087 for MissForest and 0.145 for Median/Mode — a difference that could influence clinical decisions if the model were used for risk stratification. These differences are consistent with the reconstruction and variance preservation patterns and help explain why the choice of imputation method matters beyond abstract error metrics.

### 4.4 Practical Recommendations

Based on these results, we suggest the following guidance for researchers working with clinical datasets that have similar characteristics to ours (moderate sample size, mixed continuous and categorical variables, predominantly non-Gaussian distributions). For pipelines where the main objective is accurate value reconstruction — such as preparing data for machine learning classifiers or clinical decision support tools — MissForest is likely to perform best. For analyses where hypothesis testing and confidence interval estimation are central — such as epidemiological association studies or randomised trial analyses — MICE/PMM is likely to be safer because of its superior variance preservation. Median/Mode substitution should be avoided whenever the analysis involves any form of statistical inference. Its imputed-cell variance of zero indicates that the filled values carry no patient-specific information, which is problematic for any analysis that relies on between-subject variability.

Under MNAR, none of the standard methods performed reliably. In clinical settings where missingness may depend on unobserved factors — for example, when patients with higher disease severity are less likely to return for follow-up measurements — standard imputation should be accompanied by sensitivity analyses that explicitly model the missingness mechanism.

### 4.5 Limitations

Several limitations should be considered when interpreting these results. The Monte Carlo experiment used five repetitions per scenario, which is sufficient for stable averages but limits the precision of the uncertainty estimates — narrower confidence intervals would require 50 to 100 repetitions. The data came from a single clinical cohort, and the specific correlation structure of this dataset (strong inter-variable associations among metabolic biomarkers) may influence both the absolute RMSE values and the relative ranking of methods. External validation on independent datasets, particularly from different clinical settings or populations, would be needed to assess how well these findings generalise. The MNAR mechanism implemented here represents one form of non-ignorable missingness; real-world MNAR patterns can be more complex, involving intermittent missingness, informative dropout, or cluster-level mechanisms that our simulation does not capture. The low FMI values observed in the Rubin summaries reflect the strong inter-variable correlations in this particular dataset and may not hold in settings with weaker associations or higher-dimensional variable spaces. Finally, the clinical model evaluation used a single logistic regression predicting hypertension; other outcomes and model types may show different sensitivities to imputation choices.

---

## 5. CONCLUSION

In this Monte Carlo comparison using a complete cardiometabolic clinical database, MissForest produced the lowest reconstruction errors across all missingness mechanisms and rates, while MICE with Predictive Mean Matching was the best method for preserving variance. Median/Mode substitution should be avoided in any analysis that involves hypothesis testing, because it eliminates variability among the imputed values and can alter the significance status of nearly half of all predictors in a downstream regression model. Under MNAR, all methods performed worse, and sensitivity analyses that explicitly model the missingness mechanism remain necessary.

For researchers working with clinical datasets that share the characteristics of our study — moderate sample sizes, mixed variable types, and predominantly non-Gaussian biomarker distributions — the evidence supports using MissForest when accurate reconstruction of individual values is the main goal, and MICE/PMM when preserving the variance structure for valid inference matters more. Future work could extend this framework to survival outcomes, evaluate deep learning-based imputation methods, and validate the findings on independent multicentre cohorts.

---

## SUPPLEMENTARY FIGURES

![Figure S2 — ROC Curves for Clinical Model Evaluation](../figures/supplementary/FigS4_ROC_Curves.png)

**Figure S2.** Receiver Operating Characteristic curves evaluating the discriminative performance of clinical prediction models fitted after imputation by each method.

---

## REFERENCES

1. Rubin DB. *Multiple Imputation for Nonresponse in Surveys*. New York: Wiley; 1987.
2. Little RJA, Rubin DB. *Statistical Analysis with Missing Data*. 3rd ed. Hoboken: Wiley; 2019.
3. Waljee AK, Mukherjee A, Singal AG, et al. Comparison of imputation methods for missing laboratory data in medicine. *BMJ Open*. 2013;3(8):e002847.
4. Donders ART, van der Heijden GJMG, Stijnen T, Moons KGM. Review: a gentle introduction to imputation of missing values. *J Clin Epidemiol*. 2006;59(10):1087-1091.
5. Sterne JAC, White IR, Carlin JB, et al. Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls. *BMJ*. 2009;338:b2393.
6. van Buuren S. *Flexible Imputation of Missing Data*. 2nd ed. Boca Raton: CRC Press; 2018.
7. Josse J, Prost N, Scornet E, Varoquaux G. On the consistency of supervised learning with missing values. *arXiv*. 2019:1902.06931.
8. Clinical Imputation Study — Earlier pipeline version (v2) with MICE/BayesianRidge. RMSE at MCAR 20%: 70.28. Data available in accompanying repository.
9. Jakobsen JC, Gluud C, Wetterslev J, Winkel P. When and how should multiple imputation be used for handling missing data in randomised clinical trials. *BMC Med Res Methodol*. 2017;17(1):162.
10. Stekhoven DJ, Bühlmann P. MissForest — non-parametric missing value imputation for mixed-type data. *Bioinformatics*. 2012;28(1):112-118.
11. Azur MJ, Stuart EA, Frangakis C, Leaf PJ. Multiple imputation by chained equations: what is it and how does it work? *Int J Methods Psychiatr Res*. 2011;20(1):40-49.
12. Moons KGM, Altman DG, Reitsma JB, et al. Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): explanation and elaboration. *Ann Intern Med*. 2015;162(1):W1-W73.

---

## DATA AVAILABILITY

All numerical values in this manuscript can be traced to files in the `outputs/` directory generated during pipeline execution on 2026-07-24:

- `20260724_continuous_metrics.csv` — 3240 rows
- `20260724_categorical_metrics.csv` — 1980 rows
- `20260724_variance_ratio.csv` — 3240 rows
- `20260724_rubin_rules.csv` — 648 rows
- `robustness.json` — Friedman test and robustness summaries
- `summary_table.csv` — aggregated scenario-level means

## REPRODUCIBILITY

Pipeline command: `python run.py --mc 5 --seed 42`  
Timestamp: 2026-07-24 20:11:30 UTC  
Source code, configuration, and figures: available in the accompanying repository.
