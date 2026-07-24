# -*- coding: utf-8 -*-
"""
Regenerate all supplementary figures in English, publication-ready, 300+ DPI.
Reads existing results from v2 outputs to avoid recomputing expensive imputations.
"""
import sys, io, os, warnings, json
from pathlib import Path
from itertools import combinations
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, friedmanchisquare, spearmanr, kendalltau
from sklearn.metrics import (
    mean_squared_error, f1_score, roc_auc_score, brier_score_loss,
    r2_score, accuracy_score, roc_curve
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DPI = 300
OUT_DIR = Path("figures_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# ------------------------------------------------------------------
# LOAD DATA + PREPARE
# ------------------------------------------------------------------
df_raw = pd.read_csv("v2/dataset.csv", sep=";", decimal=",")
df_raw = df_raw.replace([np.inf, -np.inf], np.nan)

var_info = {}
for col in df_raw.columns:
    nv = pd.to_numeric(df_raw[col], errors='coerce')
    nu = df_raw[col].dropna().nunique()
    nm = df_raw[col].isna().sum()
    if df_raw[col].dtype == 'object' or nv.isna().sum() > nm:
        if nu <= 2: vt = "Binary"
        elif nu <= 12: vt = "Ordinal"
        else: vt = "Nominal"
    else: vt = "Continuous"
    var_info[col] = {"type": vt}

from sklearn.preprocessing import LabelEncoder
df_enc = df_raw.copy()
les = {}
for col in df_raw.columns:
    if var_info[col]["type"] != "Continuous":
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_raw[col].fillna('__M__').astype(str))
        les[col] = le
    else:
        df_enc[col] = pd.to_numeric(df_raw[col], errors='coerce')

X_full = df_enc.values.astype(np.float64)
col_names = list(df_enc.columns)
for j in range(X_full.shape[1]):
    if np.isnan(X_full[:, j]).any():
        ok = X_full[~np.isnan(X_full[:, j]), j]
        X_full[np.isnan(X_full[:, j]), j] = np.nanmedian(ok) if len(ok) > 0 else 0

continuous_idx = [i for i, c in enumerate(col_names) if var_info[c]["type"] == "Continuous"]
categorical_idx = [i for i, c in enumerate(col_names) if var_info[c]["type"] != "Continuous"]
const_cols = ['Lasilix 40 mg', 'Lexomil 6 mg']
valid_cont = [i for i in continuous_idx if col_names[i] not in const_cols]
target_idx = col_names.index("HTA")
imc_idx = col_names.index("IMC")

# Amputation
def ampute_mcar(X, rate, seed):
    np.random.seed(seed)
    m = np.random.random(X.shape) < rate
    Xm = X.copy(); Xm[m] = np.nan
    return Xm, m

X_miss, mask = ampute_mcar(X_full, 0.20, seed=42)

# Imputation functions
def imp_med(Xm):
    Xi = Xm.copy()
    if continuous_idx:
        Xi[:, continuous_idx] = SimpleImputer(strategy="median").fit_transform(Xm[:, continuous_idx])
    if categorical_idx:
        Xi[:, categorical_idx] = SimpleImputer(strategy="most_frequent").fit_transform(Xm[:, categorical_idx])
    return Xi

def imp_knn(Xm, k=5):
    return KNNImputer(n_neighbors=k, weights="uniform").fit_transform(Xm)

def imp_mice(Xm, mi=10):
    im = IterativeImputer(max_iter=mi, random_state=42, sample_posterior=True)
    Xi = im.fit_transform(Xm)
    for idx in categorical_idx:
        lo, hi = X_full[:, idx].min(), X_full[:, idx].max()
        Xi[:, idx] = np.clip(np.round(Xi[:, idx]), lo, hi)
    return Xi

def imp_mf(Xm, ne=100, md=10, mi=5):
    rf = RandomForestRegressor(n_estimators=ne, max_depth=md, random_state=42, n_jobs=-1)
    im = IterativeImputer(estimator=rf, max_iter=mi, random_state=42, sample_posterior=False)
    Xi = im.fit_transform(Xm)
    for idx in categorical_idx:
        lo, hi = X_full[:, idx].min(), X_full[:, idx].max()
        Xi[:, idx] = np.clip(np.round(Xi[:, idx]), lo, hi)
    return Xi

METHODS_ORDER = ["Median/Mode", "KNN (k=5)", "MICE (BayesianRidge)", "MissForest"]
METHODS_FUNC = {
    "Median/Mode": imp_med,
    "KNN (k=5)": imp_knn,
    "MICE (BayesianRidge)": imp_mice,
    "MissForest": imp_mf,
}
COLORS = {"Median/Mode": "#4C72B0", "KNN (k=5)": "#55A868",
          "MICE (BayesianRidge)": "#C44E52", "MissForest": "#DD8452"}

print("Computing all imputations (this takes a few minutes for MissForest)...")
X_imp_cache = {}
for mn, mf in METHODS_FUNC.items():
    print(f"  {mn}...", end=" ", flush=True)
    X_imp_cache[mn] = mf(X_miss)
    print("OK")
print("Done.\n")

# Prep model
pred_idx_model = [i for i in continuous_idx if i != target_idx and col_names[i] not in const_cols]
scaler = StandardScaler()
X_pred_ref = X_full[:, pred_idx_model]
X_scaled_ref = scaler.fit_transform(X_pred_ref)
y_hta = X_full[:, target_idx].astype(int)
y_imc = X_full[:, imc_idx]

# =====================================================================
# FIGURE S1 — Critical Difference Diagram (Friedman + Nemenyi)
# =====================================================================
print("Generating Figure S1 — Critical Difference Diagram...")

df_v2 = pd.read_csv("v2/outputs/monte_carlo_results.csv", sep=";")
mc20 = df_v2[(df_v2["Mecanisme"] == "MCAR") & (df_v2["Taux"] == "20%") & (df_v2["Type"] == "Continue")]

method_names = METHODS_ORDER
rankings = []
for var in mc20["Variable"].unique():
    vd = mc20[mc20["Variable"] == var]
    r = {}
    for m in method_names:
        md = vd[vd["Methode"] == m]
        if len(md) > 0: r[m] = md["RMSE"].values[0]
    sm = sorted(r.items(), key=lambda x: x[1])
    rd = {m: i+1 for i, (m, _) in enumerate(sm)}
    rankings.append([rd.get(m, 5) for m in method_names])

R = np.array(rankings)
k = len(method_names)
N = len(rankings)
avg_ranks = R.mean(axis=0)
q_alpha = {2: 1.960, 3: 2.343, 4: 2.569}
CD = q_alpha.get(k, 2.569) * np.sqrt(k * (k + 1) / (6 * N))

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set_xlim(1, k)
ax.set_ylim(0.5, k - 0.5)
sorted_idx = np.argsort(avg_ranks)
rank_labels = [f"{method_names[i]} (R={avg_ranks[i]:.2f})" for i in sorted_idx]

for pos, idx in enumerate(sorted_idx):
    ax.plot(avg_ranks[idx], k - 1 - pos, 'o', markersize=14, color=COLORS[method_names[idx]],
            markeredgecolor='black', markeredgewidth=1)
    ax.text(avg_ranks[idx] + 0.05, k - 1 - pos, rank_labels[pos], fontsize=10, va='center')

# CD bar
y_bar = k - 0.1
ax.plot([avg_ranks[sorted_idx[0]], avg_ranks[sorted_idx[0]] + CD], [y_bar, y_bar], 'k-', linewidth=3)
ax.text(avg_ranks[sorted_idx[0]] + CD/2, y_bar + 0.1, f'CD = {CD:.3f}', ha='center', fontsize=11, fontweight='bold')

# Connecting lines for non-significant pairs
holm_comps = []
for i, j in combinations(range(k), 2):
    diff = abs(avg_ranks[i] - avg_ranks[j])
    se_val = np.sqrt(k*(k+1)/(6*N))
    z = diff / se_val
    p = 2 * (1 - norm.cdf(abs(z)))
    holm_comps.append((i, j, p))

holm_sorted = sorted(holm_comps, key=lambda x: x[2])
n_comp = len(holm_sorted)
non_sig_pairs = []
for rank_idx, (i, j, p) in enumerate(holm_sorted, 1):
    alpha_holm = 0.05 / (n_comp - rank_idx + 1)
    if p >= alpha_holm:
        non_sig_pairs.append((i, j))

for i, j in non_sig_pairs:
    yi = k - 1 - list(sorted_idx).index(i)
    yj = k - 1 - list(sorted_idx).index(j)
    ax.plot([avg_ranks[i], avg_ranks[j]], [yi, yj], 'k-', linewidth=2.5)

ax.set_yticks([])
ax.set_xlabel("Mean Friedman Rank", fontsize=12, fontweight='bold')
ax.set_title("Critical Difference Diagram — MCAR 20% (Friedman + Nemenyi)", fontsize=13, fontweight='bold')
ax.text(0.98, 0.02, f"N = {N} variables | k = {k} methods | α = 0.05",
        transform=ax.transAxes, ha='right', fontsize=9, fontstyle='italic')
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS1_CriticalDifferenceDiagram.png", dpi=DPI)
plt.close()
print("  Done.")

# =====================================================================
# FIGURE S2 — ROC Curves
# =====================================================================
print("Generating Figure S2 — ROC Curves...")

fig, ax = plt.subplots(figsize=(7, 7))
for mn in METHODS_ORDER:
    Xp = scaler.transform(X_imp_cache[mn][:, pred_idx_model])
    lr = LogisticRegression(max_iter=2000, random_state=42)
    yp = cross_val_predict(lr, Xp, y_hta, cv=5, method='predict_proba')[:, 1]
    fpr, tpr, _ = roc_curve(y_hta, yp)
    auc = roc_auc_score(y_hta, yp)
    ax.plot(fpr, tpr, linewidth=2.5, color=COLORS[mn], label=f"{mn} (AUC = {auc:.3f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='Random (AUC = 0.500)')
ax.set_xlabel("1 − Specificity (False Positive Rate)", fontsize=12, fontweight='bold')
ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=12, fontweight='bold')
ax.set_title("ROC Curves — Hypertension Prediction after Imputation (MCAR 20%)", fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS2_ROCCurves.png", dpi=DPI)
plt.close()
print("  Done.")

# =====================================================================
# FIGURE S3 — Calibration Curves
# =====================================================================
print("Generating Figure S3 — Calibration Curves...")

fig, ax = plt.subplots(figsize=(7, 7))
for mn in METHODS_ORDER:
    Xp = scaler.transform(X_imp_cache[mn][:, pred_idx_model])
    lr = LogisticRegression(max_iter=2000, random_state=42)
    yp = cross_val_predict(lr, Xp, y_hta, cv=5, method='predict_proba')[:, 1]
    prob_true, prob_pred = calibration_curve(y_hta, yp, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, marker='o', markersize=7, linewidth=2.5, color=COLORS[mn], label=mn)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='Perfect Calibration')
ax.set_xlabel("Predicted Probability", fontsize=12, fontweight='bold')
ax.set_ylabel("Observed Proportion", fontsize=12, fontweight='bold')
ax.set_title("Calibration Curves — Hypertension Prediction (MCAR 20%)", fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS3_CalibrationCurves.png", dpi=DPI)
plt.close()
print("  Done.")

# =====================================================================
# FIGURE S4 — Forest Plot of Log Odds Ratios
# =====================================================================
print("Generating Figure S4 — Forest Plot...")

or_data = {}
for mn in METHODS_ORDER:
    Xp = scaler.transform(X_imp_cache[mn][:, pred_idx_model])
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(Xp, y_hta)
    or_data[mn] = np.exp(lr.coef_[0][:10])

pred_labels = [col_names[i] for i in pred_idx_model[:10]]
n_preds = len(pred_labels)
fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(n_preds)
bar_height = 0.2
for i, mn in enumerate(METHODS_ORDER):
    vals = or_data[mn]
    log_vals = np.log(np.clip(vals, 0.01, 100))
    ax.barh(y_pos + i * bar_height, log_vals, bar_height, color=COLORS[mn], alpha=0.85, label=mn, edgecolor='white', linewidth=0.5)

ax.set_yticks(y_pos + bar_height * 1.5)
ax.set_yticklabels(pred_labels, fontsize=9)
ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax.set_xlabel("log(Odds Ratio) for Hypertension", fontsize=12, fontweight='bold')
ax.set_title("Forest Plot — Predictors of Hypertension after Imputation (MCAR 20%)", fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.2, axis='x')
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS4_ForestPlot.png", dpi=DPI)
plt.close()
print("  Done.")

# =====================================================================
# FIGURE S5 — Correlation Difference Heatmap
# =====================================================================
print("Generating Figure S5 — Correlation Heatmap...")

orig_corr = np.corrcoef(X_full[:, valid_cont].T)
mf_corr = np.corrcoef(X_imp_cache["MissForest"][:, valid_cont].T)
corr_diff = orig_corr - mf_corr
n_valid = len(valid_cont)
short_labels = [col_names[i][:14] for i in valid_cont]

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_diff, cmap='RdBu_r', center=0, vmin=-0.15, vmax=0.15,
            ax=ax, xticklabels=short_labels, yticklabels=short_labels,
            cbar_kws={'label': 'Δ r (Original − Imputed)', 'shrink': 0.8},
            annot=False, linewidths=0.1)
ax.set_title("Pearson Correlation Difference — MissForest vs Original (MCAR 20%)", fontsize=13, fontweight='bold')
ax.tick_params(axis='both', labelsize=7)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS5_CorrelationHeatmap.png", dpi=DPI)
plt.close()
print("  Done.")

# =====================================================================
# FIGURE S6 — Bland-Altman Plot (BMI)
# =====================================================================
print("Generating Figure S6 — Bland-Altman Plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
for ax, mn in zip(axes.flat, METHODS_ORDER):
    Xi = X_imp_cache[mn]
    true_val = X_full[:, imc_idx]
    imp_val = Xi[:, imc_idx]
    diff = imp_val - true_val
    mean_val = (true_val + imp_val) / 2
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)

    ax.scatter(mean_val, diff, alpha=0.35, s=18, color=COLORS[mn], edgecolors='none')
    ax.axhline(md, color='red', linewidth=2, linestyle='-', label=f'Bias = {md:.2f}')
    ax.axhline(md + 1.96*sd, color='red', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axhline(md - 1.96*sd, color='red', linewidth=1.2, linestyle='--', alpha=0.7,
               label=f'±1.96 SD = ±{1.96*sd:.2f}')
    ax.set_xlabel("Mean (Original + Imputed BMI)", fontsize=10)
    ax.set_ylabel("Difference (Imputed − Original BMI)", fontsize=10)
    ax.set_title(mn, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Bland-Altman Analysis — Body Mass Index (MCAR 20%)", fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS6_BlandAltman.png", dpi=DPI)
plt.close()
print("  Done.")

# =====================================================================
# FIGURE S7 — Violin Plot of Normalized Errors
# =====================================================================
print("Generating Figure S7 — Violin Plot...")

fig, ax = plt.subplots(figsize=(10, 6))
violin_data = []
for mn in METHODS_ORDER:
    Xi = X_imp_cache[mn]
    errors = []
    for idx in valid_cont:
        tv = X_full[mask[:, idx], idx]
        iv = Xi[mask[:, idx], idx]
        stdv = np.std(X_full[:, idx])
        if stdv > 1e-8 and len(tv) > 0:
            errors.extend(((iv - tv) / stdv).tolist())
    violin_data.append(errors)

parts = ax.violinplot(violin_data, showmeans=True, showmedians=True, widths=0.7)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(list(COLORS.values())[i])
    pc.set_alpha(0.65)
    pc.set_edgecolor('black')
    pc.set_linewidth(0.8)

for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    if partname in parts:
        parts[partname].set_color('black')
        parts[partname].set_linewidth(1.2)

ax.set_xticks(range(1, len(METHODS_ORDER)+1))
ax.set_xticklabels(METHODS_ORDER, fontsize=10)
ax.set_ylabel("Normalized Imputation Error (Imputed − True) / σ", fontsize=11, fontweight='bold')
ax.set_title("Distribution of Normalized Imputation Errors — MCAR 20%", fontsize=13, fontweight='bold')
ax.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.6)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS7_ViolinPlot.png", dpi=DPI)
plt.close()
print("  Done.")

# =====================================================================
# FIGURE S8 — QQ-Plot (BMI)
# =====================================================================
print("Generating Figure S8 — QQ-Plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
mask_bmi = mask[:, imc_idx]
for ax, mn in zip(axes.flat, METHODS_ORDER):
    Xi = X_imp_cache[mn]
    true_sorted = np.sort(X_full[mask_bmi, imc_idx])
    imp_sorted = np.sort(Xi[mask_bmi, imc_idx])
    ax.scatter(true_sorted, imp_sorted, alpha=0.5, s=18, color=COLORS[mn], edgecolors='none')
    mv = min(true_sorted.min(), imp_sorted.min())
    Mv = max(true_sorted.max(), imp_sorted.max())
    ax.plot([mv, Mv], [mv, Mv], 'r--', linewidth=1.5, alpha=0.8)
    ax.set_xlabel("True BMI (kg/m²)", fontsize=10)
    ax.set_ylabel("Imputed BMI (kg/m²)", fontsize=10)
    ax.set_title(mn, fontsize=12, fontweight='bold')

    # Add r²
    r2 = np.corrcoef(true_sorted, imp_sorted)[0, 1]**2
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, fontsize=10,
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(True, alpha=0.3)

fig.suptitle("Q-Q Plot — BMI: Imputed vs True Values (MCAR 20%)", fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS8_QQPlot.png", dpi=DPI)
plt.close()
print("  Done.")

# =====================================================================
# FIGURE S9 — Convergence Curves (MICE + MissForest)
# =====================================================================
print("Generating Figure S9 — Convergence Curves...")

# Recompute convergence curves (expensive but informative for publication)
mice_errors = []
for ni in [1, 2, 3, 5, 8, 12]:
    im = IterativeImputer(max_iter=ni, random_state=42, sample_posterior=True)
    Xi = im.fit_transform(X_miss)
    err = np.mean([np.sqrt(mean_squared_error(X_full[mask[:, i], i], Xi[mask[:, i], i])) for i in valid_cont])
    mice_errors.append((ni, err))

mf_errors = []
for ni in [1, 2, 3, 5, 7, 10]:
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    im = IterativeImputer(estimator=rf, max_iter=ni, random_state=42, sample_posterior=False)
    Xi = im.fit_transform(X_miss)
    err = np.mean([np.sqrt(mean_squared_error(X_full[mask[:, i], i], Xi[mask[:, i], i])) for i in valid_cont])
    mf_errors.append((ni, err))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.plot([e[0] for e in mice_errors], [e[1] for e in mice_errors], 'o-', color='#C44E52', linewidth=2.5, markersize=8)
ax1.set_xlabel("Number of Iterations", fontsize=12, fontweight='bold')
ax1.set_ylabel("Mean RMSE (Continuous Variables)", fontsize=12, fontweight='bold')
ax1.set_title("MICE (BayesianRidge) Convergence", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
for e in mice_errors:
    ax1.annotate(f'{e[1]:.1f}', (e[0], e[1]), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=8)

ax2.plot([e[0] for e in mf_errors], [e[1] for e in mf_errors], 's-', color='#DD8452', linewidth=2.5, markersize=8)
ax2.set_xlabel("Number of Iterations", fontsize=12, fontweight='bold')
ax2.set_ylabel("Mean RMSE (Continuous Variables)", fontsize=12, fontweight='bold')
ax2.set_title("MissForest Convergence", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
for e in mf_errors:
    ax2.annotate(f'{e[1]:.1f}', (e[0], e[1]), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=8)

fig.suptitle("Imputation Algorithm Convergence — MCAR 20%", fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS9_ConvergenceCurves.png", dpi=DPI)
plt.close()
print("  Done.")

print(f"\n{'='*60}")
print(f"All 9 supplementary figures saved to {OUT_DIR}/")
print(f"Resolution: {DPI} DPI")
print(f"{'='*60}")
