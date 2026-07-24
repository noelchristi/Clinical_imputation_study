"""
Publication-quality figure generation for:
"A Monte Carlo Simulation Study Comparing Missing Data Imputation Methods
in Cardiometabolic Clinical Research"

Generates 8 journal-ready figures in English at 300 DPI.
Target journal style: Clinical Epidemiology / Statistics in Medicine
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Method colour palette (colorblind-safe)
METHOD_COLORS = {
    'Median/Mode':        '#1f77b4',   # blue
    'KNN (k=5)':          '#ff7f0e',   # orange
    'MICE (BayesianRidge)': '#2ca02c', # green
    'MissForest':         '#d62728',   # red
}
METHOD_MARKERS = {
    'Median/Mode':        'o',
    'KNN (k=5)':          's',
    'MICE (BayesianRidge)': '^',
    'MissForest':         'D',
}
METHOD_ORDER = ['Median/Mode', 'KNN (k=5)', 'MICE (BayesianRidge)', 'MissForest']

OUT_DIR = r'c:\Users\LENOVO\Desktop\THESE\article2\figures_final'
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = r'c:\Users\LENOVO\Desktop\THESE\article2\v2'

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
mc = pd.read_csv(os.path.join(DATA_DIR, 'outputs', 'monte_carlo_results.csv'), sep=';')
mc['Methode'] = mc['Methode'].str.strip()

# Map French method names to English
method_map = {
    'Mediane/Mode': 'Median/Mode',
    'KNN (k=5)': 'KNN (k=5)',
    'MICE (BayesianRidge)': 'MICE (BayesianRidge)',
    'MissForest': 'MissForest',
}
mc['Methode'] = mc['Methode'].map(method_map).fillna(mc['Methode'])

# Map French mechanism names
mech_map = {'MCAR': 'MCAR', 'MAR': 'MAR', 'MNAR': 'MNAR'}
mc['Mecanisme'] = mc['Mecanisme'].str.strip()

stability = pd.read_csv(os.path.join(DATA_DIR, 'outputs', 'stability_results.csv'), sep=';')
stability['Methode'] = stability['Methode'].str.strip().map(method_map).fillna(stability['Methode'])
stability['Mecanisme'] = stability['Mecanisme'].str.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Study Design Flowchart
# ─────────────────────────────────────────────────────────────────────────────
def fig1_flowchart():
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.axis('off')

    def box(ax, x, y, w, h, text, facecolor='#dce8f5', edgecolor='#2c5f8a',
            fontsize=9, bold=False, radius=0.25):
        fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle=f"round,pad={radius}",
                               facecolor=facecolor, edgecolor=edgecolor, linewidth=1.4)
        ax.add_patch(fancy)
        fw = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fw, wrap=True,
                multialignment='center')

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#2c5f8a',
                                   lw=1.5, connectionstyle='arc3,rad=0'))

    # Top box
    box(ax, 5.5, 8.3, 5.5, 0.9,
        'Complete Cardiometabolic Clinical Database\nN = 312 subjects  ·  P = 33 variables  ·  0% native missing',
        facecolor='#1f4e79', edgecolor='#1f4e79', fontsize=9.5, bold=True)
    ax.patches[-1].set_fc('#1f4e79')
    ax.texts[-1].set_color('white')

    # Quality audit box
    box(ax, 5.5, 7.15, 5.5, 0.75,
        'Data Quality Audit  ·  Variable Classification\n'
        '87.9% non-Gaussian  ·  2 quasi-constant excluded from multivariate analysis',
        facecolor='#d6e4f0')
    arrow(ax, 5.5, 7.85, 5.5, 7.53)

    # Three mechanism boxes
    for i, (mech, col) in enumerate([
            ('MCAR\n(Missing Completely\nAt Random)', '#e8f4e8'),
            ('MAR\n(Missing At Random)\nLogistic dependence on\nobserved covariates', '#fff3cd'),
            ('MNAR\n(Missing Not At Random)\nProportional to deviation\nfrom median', '#fce8e8')]):
        bx = 1.7 + i * 3.15
        box(ax, bx, 5.9, 2.7, 1.35, mech, facecolor=col, fontsize=8.5)
        arrow(ax, bx, 6.78, bx, 6.575)

    ax.annotate('', xy=(8.85, 6.78), xytext=(2.2, 6.78),
                arrowprops=dict(arrowstyle='-', color='#2c5f8a', lw=1.5))
    arrow(ax, 5.5, 6.78, 5.5, 6.575)
    arrow(ax, 5.5, 7.15 - 0.375, 5.5, 6.78)

    # Missing rates row
    for i, rate in enumerate(['10%', '20%', '40%']):
        bx = 2.7 + i * 1.5
        box(ax, bx, 5.0, 1.2, 0.55, f'τ = {rate}', facecolor='#f0f0f0', fontsize=9)

    ax.annotate('', xy=(5.7, 5.0), xytext=(1.7, 5.0),
                arrowprops=dict(arrowstyle='-', color='#555', lw=1.0))
    for i, (mech_x, rate_x) in enumerate([(1.7, 2.7), (4.85, 4.2), (8.85, 5.7)]):
        arrow(ax, mech_x, 5.225, rate_x, 5.275)

    # 180 runs box
    box(ax, 5.5, 4.25, 5.5, 0.7,
        '9 Scenarios × 5 Monte Carlo iterations × 4 Methods  =  180 independent imputation runs',
        facecolor='#2c5f8a', edgecolor='#1f4e79', bold=True, fontsize=9)
    ax.patches[-1].set_fc('#2c5f8a')
    ax.texts[-1].set_color('white')
    arrow(ax, 5.5, 4.725, 5.5, 4.6)

    # Four method boxes
    methods = ['Median/Mode\n(Baseline)', 'KNN\n(k = 5)', 'MICE\n(BayesianRidge)', 'MissForest\n(100 trees)']
    mcolors_face = ['#dce8f5', '#ffe0b2', '#d4edda', '#ffd5d5']
    for i, (m, fc) in enumerate(zip(methods, mcolors_face)):
        bx = 1.3 + i * 2.7
        box(ax, bx, 3.35, 2.3, 0.75, m, facecolor=fc, fontsize=8.5)
        arrow(ax, bx, 3.9, bx, 3.725)

    ax.annotate('', xy=(9.95, 3.9), xytext=(1.3, 3.9),
                arrowprops=dict(arrowstyle='-', color='#555', lw=1.0))
    arrow(ax, 5.5, 3.9, 5.5, 3.88)

    # Evaluation box
    box(ax, 5.5, 2.45, 9.8, 1.35,
        'Performance Evaluation\n'
        'Continuous: RMSE · MAE · KS test · Wilcoxon · Bias · Variance Ratio (R\u1d65)\n'
        'Categorical/Ordinal: F1 · Accuracy · Cohen\'s κ · MCC · χ² test\n'
        'Model preservation: AUC · Brier · OR bias · β bias · Significance change rate',
        facecolor='#f8f4ff', edgecolor='#6a0dad', fontsize=8.5)
    arrow(ax, 5.5, 2.975, 5.5, 2.975 - 0.001)
    for bx in [1.3, 4.0, 6.7, 9.4]:
        arrow(ax, bx, 2.975, 5.5, 2.975)

    # Stat comparison box
    box(ax, 5.5, 1.3, 9.8, 0.95,
        'Statistical Comparison  ·  Correlation & Variance Preservation  ·  Rubin\'s Rules (M = 5)\n'
        'Friedman test · Nemenyi post-hoc · Holm correction · Evidence-based Decision Algorithm',
        facecolor='#fff9e6', edgecolor='#b8860b', fontsize=8.5)
    arrow(ax, 5.5, 1.775, 5.5, 1.775 - 0.001)
    arrow(ax, 5.5, 1.775, 5.5, 1.775)

    plt.tight_layout(pad=0.1)
    out = os.path.join(OUT_DIR, 'Fig1_StudyDesign_Flowchart.tif')
    fig.savefig(out, dpi=300, format='tiff')
    print(f'  Saved {out}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Missing Data Pattern Matrix (MCAR 20%, first 50 patients)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_missing_heatmap():
    # Load dataset
    ds_path = os.path.join(DATA_DIR, 'dataset.csv')
    try:
        df = pd.read_csv(ds_path, sep=';', encoding='utf-8')
    except Exception:
        df = pd.read_csv(ds_path, sep=',', encoding='utf-8')

    np.random.seed(42)
    df_sub = df.head(50).copy()
    mask = np.random.rand(*df_sub.shape) < 0.20
    missing_mat = pd.DataFrame(mask.astype(float), columns=df_sub.columns)

    # Translate / clean column names
    col_rename = {
        'Sexe': 'Sex', 'IMC': 'BMI', 'TT': 'WC', 'PAS': 'SBP', 'PAD': 'DBP',
        'HTA': 'HBP', 'HTS/HTD': 'HBP type', 'Prot': 'Protein', 'Totc': 'TC',
        'TG': 'TG', 'Non HDL': 'Non-HDL', 'Glu': 'Glu', 'HOMA-IR': 'HOMA-IR',
        'Adipo HMW': 'HMW-Adipo', 'Glibenclamide 5mg': 'Glib.5mg',
        'Insuline 10 UI': 'Ins.10UI', 'Lasilix 40 mg': 'Lasix.40',
        'Lexomil 6 mg': 'Lexo.6', 'Metformine 500 mg': 'Metf.500',
        'Profil_glycemiq': 'GlycProf',
        'Ratio Tot-c/HDL': 'TC/HDL', 'Ratio LDL-c/HDL': 'LDL/HDL',
        'Ratio TG/HDL': 'TG/HDL', 'Ratio LogTG/HDL': 'LogTG/HDL',
        'Ins (µUI/mL)': 'Ins(µUI/mL)', 'lu(mmol/l)': 'Glu(mmol/l)',
    }
    missing_mat.columns = [col_rename.get(c, c) for c in missing_mat.columns]

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(missing_mat.T.values, aspect='auto', cmap=cmap,
                   vmin=0, vmax=1, interpolation='none')

    ax.set_xticks(range(50))
    ax.set_xticklabels([str(i+1) if (i+1) % 5 == 0 else '' for i in range(50)], fontsize=7)
    ax.set_yticks(range(len(missing_mat.columns)))
    ax.set_yticklabels(missing_mat.columns, fontsize=7.5)
    ax.set_xlabel('Patient index (first 50)', fontsize=9, labelpad=4)
    ax.set_ylabel('Variable', fontsize=9, labelpad=4)
    ax.set_title('Missing Data Pattern — MCAR 20% (first 50 patients)',
                 fontsize=11, fontweight='bold', pad=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Observed', 'Missing'], fontsize=8)
    cbar.ax.tick_params(length=0)

    # Grid lines between variables
    for y in np.arange(-0.5, len(missing_mat.columns), 1):
        ax.axhline(y, color='white', lw=0.3)

    plt.tight_layout(pad=0.5)
    out = os.path.join(OUT_DIR, 'Fig2_MissingDataPattern_MCAR20.tif')
    fig.savefig(out, dpi=300, format='tiff')
    print(f'  Saved {out}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Mean RMSE Curves by Mechanism and Missing Rate
# ─────────────────────────────────────────────────────────────────────────────
def fig3_rmse_curves():
    cont = mc[(mc['Type'] == 'Continue') & (~mc['Variable'].isin(['Lasilix 40 mg', 'Lexomil 6 mg']))].copy()

    # Aggregate mean RMSE per (mechanism, rate, method)
    agg = (cont.groupby(['Mecanisme', 'Taux_num', 'Methode'])['RMSE']
               .mean().reset_index())

    mechanisms = ['MCAR', 'MAR', 'MNAR']
    rates = [0.10, 0.20, 0.40]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    for ax, mech in zip(axes, mechanisms):
        sub = agg[agg['Mecanisme'] == mech]
        has_data = len(sub) > 0

        for method in METHOD_ORDER:
            msub = sub[sub['Methode'] == method].sort_values('Taux_num')
            if len(msub) == 0:
                continue
            ax.plot(msub['Taux_num'], msub['RMSE'],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    markersize=6, linewidth=2.0,
                    label=method, zorder=3)

        ax.set_xlabel('Missing Rate (τ)', fontsize=9, labelpad=4)
        ax.set_ylabel('Mean RMSE', fontsize=9, labelpad=4)
        ax.set_title(mech, fontsize=11, fontweight='bold')
        ax.set_xticks([0.10, 0.20, 0.40])
        ax.set_xticklabels(['10%', '20%', '40%'], fontsize=9)

        if not has_data:
            ax.text(0.5, 0.5, 'No valid\ndata available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray', style='italic')

        if mech == 'MCAR':
            ax.legend(loc='upper left', framealpha=0.85, fontsize=8.5,
                      handlelength=1.5, handletextpad=0.5)

    fig.suptitle('Mean RMSE of Imputation Methods by Missingness Mechanism and Missing Rate\n'
                 '(Continuous variables, N = 29, mean across M = 5 Monte Carlo iterations)',
                 fontsize=10, y=1.01)

    plt.tight_layout(pad=0.8, w_pad=1.5)
    out = os.path.join(OUT_DIR, 'Fig3_RMSE_Curves_by_Mechanism.tif')
    fig.savefig(out, dpi=300, format='tiff')
    print(f'  Saved {out}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — RMSE Heatmap per Variable and Method (MCAR 40%)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_rmse_heatmap():
    sub = mc[(mc['Mecanisme'] == 'MCAR') &
             (mc['Taux'] == '40%') &
             (mc['Type'] == 'Continue')].copy()

    pivot = (sub.groupby(['Variable', 'Methode'])['RMSE']
               .mean()
               .unstack('Methode')
               .reindex(columns=METHOD_ORDER)
               .dropna(how='all'))

    # Sort variables by MissForest RMSE descending
    if 'MissForest' in pivot.columns:
        pivot = pivot.sort_values('MissForest', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 9))

    # Use a perceptually uniform cmap with log normalisation (skewed data)
    vmax = pivot.values[np.isfinite(pivot.values)].max()
    norm = mcolors.LogNorm(vmin=max(0.1, pivot.values[pivot.values > 0].min()), vmax=vmax)
    cmap = plt.cm.YlOrRd

    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, norm=norm)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9.5, rotation=15, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8.5)
    ax.set_xlabel('Imputation Method', fontsize=10, labelpad=6)
    ax.set_ylabel('Variable', fontsize=10, labelpad=6)
    ax.set_title('Per-Variable RMSE by Imputation Method — MCAR 40%',
                 fontsize=11, fontweight='bold', pad=8)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                txt = f'{val:.1f}' if val < 1000 else f'{val:.0f}'
                fc = 'white' if val > vmax * 0.4 else 'black'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=fc)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label('RMSE (log scale)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0.5)
    out = os.path.join(OUT_DIR, 'Fig4_RMSE_Heatmap_Variables_MCAR40.tif')
    fig.savefig(out, dpi=300, format='tiff')
    print(f'  Saved {out}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Radar Plot: Multi-Metric Performance (MCAR)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_radar():
    # Values from Table 2 of the manuscript (normalised 0–1, higher = better)
    # Metrics: F1(Ord), Kappa(Ord), Accuracy(Bin), 1-NRMSE(Cont), Var_Ratio(Cont)
    radar_data = {
        # (MCAR 10%, 20%, 40%) for each method
        # Format: [F1_ord, Kappa_ord, Acc_bin, 1-NRMSE_norm, VarRatio_norm]
        10: {
            'Median/Mode':        [0.360, 0.000, 0.615, 0.45, 0.86],
            'KNN (k=5)':          [0.401, 0.170, 0.641, 0.52, 0.88],
            'MICE (BayesianRidge)':[0.441, 0.148, 0.638, 0.28, 0.78],
            'MissForest':         [0.865, 0.771, 0.744, 0.82, 0.92],
        },
        20: {
            'Median/Mode':        [0.397, 0.000, 0.643, 0.41, 0.78],
            'KNN (k=5)':          [0.363, 0.127, 0.672, 0.48, 0.80],
            'MICE (BayesianRidge)':[0.453, 0.145, 0.635, 0.22, 0.74],
            'MissForest':         [0.842, 0.731, 0.791, 0.75, 0.89],
        },
        40: {
            'Median/Mode':        [0.394, 0.000, 0.644, 0.42, 0.60],
            'KNN (k=5)':          [0.243, 0.064, 0.639, 0.47, 0.65],
            'MICE (BayesianRidge)':[0.364, 0.028, 0.585, 0.18, 0.70],
            'MissForest':         [0.798, 0.655, 0.745, 0.66, 0.80],
        },
    }

    metrics = ["F1\n(Ordinal)", "Cohen's κ\n(Ordinal)", "Accuracy\n(Binary)",
               "1−NRMSE\n(Continuous)", "Variance\nRatio"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                             subplot_kw=dict(polar=True))

    for ax, rate in zip(axes, [10, 20, 40]):
        for method in METHOD_ORDER:
            vals = radar_data[rate][method] + [radar_data[rate][method][0]]
            ax.plot(angles, vals, color=METHOD_COLORS[method],
                    linewidth=1.8, marker=METHOD_MARKERS[method],
                    markersize=5, label=method)
            ax.fill(angles, vals, color=METHOD_COLORS[method], alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=8, ha='center')
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7, color='gray')
        ax.set_ylim(0, 1)
        ax.set_title(f'MCAR {rate}%', fontsize=10, fontweight='bold', pad=14)
        ax.grid(alpha=0.3)
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_color('#cccccc')

    # Shared legend
    handles = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in METHOD_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Multi-Metric Performance Profile by Missing Rate (MCAR Mechanism)',
                 fontsize=11, fontweight='bold', y=1.02)

    plt.tight_layout(pad=0.8, w_pad=2.0)
    out = os.path.join(OUT_DIR, 'Fig5_RadarPlot_MCAR.tif')
    fig.savefig(out, dpi=300, format='tiff', bbox_inches='tight')
    print(f'  Saved {out}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Post-Imputation Variance Ratio
# ─────────────────────────────────────────────────────────────────────────────
def fig6_variance_ratio():
    # Use stability results which have per-scenario Var_Ratio
    stab = stability[stability['Mecanisme'].isin(['MCAR', 'MAR'])].copy()
    stab['Taux_num'] = stab['Taux'].str.rstrip('%').astype(float) / 100.0

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ls_map = {'MCAR': '-', 'MAR': '--'}
    for mech in ['MCAR', 'MAR']:
        for method in METHOD_ORDER:
            sub = stab[(stab['Mecanisme'] == mech) &
                       (stab['Methode'] == method)].sort_values('Taux_num')
            if len(sub) == 0:
                continue
            lbl = f'{mech} – {method}' if mech == 'MCAR' else None
            ax.plot(sub['Taux_num'], sub['Var_Ratio'],
                    color=METHOD_COLORS[method],
                    linestyle=ls_map[mech],
                    marker=METHOD_MARKERS[method],
                    markersize=6, linewidth=1.8,
                    label=f'{mech} – {method}')

    # Reference line
    ax.axhline(1.0, color='black', linestyle=':', linewidth=1.5, label='Reference (R\u1d65 = 1.0)')

    ax.set_xlabel('Missing Rate (τ)', fontsize=10, labelpad=4)
    ax.set_ylabel('Variance Ratio (R\u1d65 = Var\u1d2c\u1d62\u1d38 / Var\u1d12\u1d3f\u1d35\u1d33)', fontsize=10, labelpad=4)
    ax.set_title('Post-Imputation Variance Conservation by Mechanism and Missing Rate',
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks([0.10, 0.20, 0.40])
    ax.set_xticklabels(['10%', '20%', '40%'], fontsize=9)

    # Add shaded zones
    ax.axhspan(0.80, 1.20, alpha=0.06, color='green', label='Acceptable range [0.80, 1.20]')
    ax.axhspan(0, 0.80, alpha=0.06, color='red')
    ax.axhspan(1.20, ax.get_ylim()[1] if ax.get_ylim()[1] > 1.20 else 1.40, alpha=0.06, color='red')

    # Custom legend: methods + line styles
    from matplotlib.lines import Line2D
    method_handles = [Line2D([0], [0], color=METHOD_COLORS[m], lw=2,
                             marker=METHOD_MARKERS[m], markersize=5, label=m)
                      for m in METHOD_ORDER]
    style_handles = [Line2D([0], [0], color='gray', lw=2, ls='-', label='MCAR'),
                     Line2D([0], [0], color='gray', lw=2, ls='--', label='MAR'),
                     Line2D([0], [0], color='black', lw=1.5, ls=':', label='Reference (R\u1d65 = 1.0)'),
                     mpatches.Patch(fc='green', alpha=0.18, label='Acceptable zone [0.80–1.20]')]
    ax.legend(handles=method_handles + style_handles, loc='lower left',
              fontsize=8.5, framealpha=0.9, ncol=2)

    plt.tight_layout(pad=0.5)
    out = os.path.join(OUT_DIR, 'Fig6_VarianceRatio.tif')
    fig.savefig(out, dpi=300, format='tiff')
    print(f'  Saved {out}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Evidence-Based Decision Algorithm
# ─────────────────────────────────────────────────────────────────────────────
def fig7_decision_tree():
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color scheme
    HEADER_BG  = '#1f4e79'
    BRANCH_BG  = '#2c5f8a'
    COND_BG    = '#dce8f5'
    GOOD_BG    = '#d4edda'
    WARN_BG    = '#fff3cd'
    ALERT_BG   = '#fce8e8'
    CONST_BG   = '#f0f0f0'
    FOOT_BG    = '#f8f4ff'

    def box(x, y, w, h, text, fc=COND_BG, ec='#2c5f8a', fs=8.5, bold=False, tc='black'):
        fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle='round,pad=0.18',
                               facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=3)
        ax.add_patch(fancy)
        fw = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                fontweight=fw, multialignment='center', color=tc, zorder=4)

    def arr(x1, y1, x2, y2, label='', color='#2c5f8a'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.4),
                    zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.08, my, label, fontsize=7.5, color=color, style='italic', zorder=5)

    # Title
    box(6.5, 9.5, 12.5, 0.75,
        'EVIDENCE-BASED DECISION ALGORITHM FOR MISSING DATA IMPUTATION IN CLINICAL RESEARCH',
        fc=HEADER_BG, ec=HEADER_BG, fs=10, bold=True, tc='white')

    # Start: missing rate question
    box(6.5, 8.55, 5.0, 0.65,
        'What is the proportion of missing data per variable?',
        fc=BRANCH_BG, ec=BRANCH_BG, fs=9.5, bold=True, tc='white')
    arr(6.5, 9.125, 6.5, 8.875)

    # Branch 1: < 5%
    box(1.8, 7.45, 2.8, 0.75,
        '< 5% missing\nN > 30 and MCAR plausible',
        fc=GOOD_BG, ec='#28a745')
    arr(3.2, 8.55, 3.2, 7.825, '< 5%', '#28a745')
    ax.plot([3.2, 1.8], [8.55, 8.55], color='#2c5f8a', lw=1.4)

    box(1.8, 6.65, 2.8, 0.65,
        '✔  Complete-case analysis\n     acceptable',
        fc=GOOD_BG, ec='#28a745', fs=8)
    arr(1.8, 7.075, 1.8, 6.975)

    # Branch 2: 5–15%
    box(6.5, 7.45, 3.2, 0.75,
        '5–15% missing\nVariable type?',
        fc=WARN_BG, ec='#e6ac00')
    arr(6.5, 8.55, 6.5, 7.825, '5–15%', '#e6ac00')

    # Sub-branches for 5-15%
    sub_items = [
        (4.5, 6.4, 'Non-Gaussian continuous\nor mixed types',  '→ MissForest  (RECOMMENDED)', '#d4edda', '#28a745'),
        (6.5, 6.4, 'Gaussian continuous',                      '→ MICE or MissForest',         '#dce8f5', '#2c5f8a'),
        (8.5, 6.4, 'Binary / Ordinal\nor quasi-constant',      '→ MissForest or\n   Median/Mode', '#ffe0b2', '#e65c00'),
    ]
    ax.plot([4.5, 8.5], [7.075, 7.075], color='#e6ac00', lw=1.2)
    for bx, by, top, bot, fc, ec in sub_items:
        box(bx, by + 0.22, 1.9, 0.42, top, fc=WARN_BG, ec=ec, fs=7.5)
        box(bx, by - 0.28, 1.9, 0.42, bot, fc=fc, ec=ec, fs=7.5, bold=True)
        arr(bx, 7.075, bx, by + 0.44)
        arr(bx, by, bx, by - 0.07)

    box(6.5, 5.55, 5.5, 0.4,
        'Multiple imputation M ≥ 5 strongly recommended', fc='#f0e6ff', ec='#6a0dad', fs=8)
    arr(6.5, 5.875, 6.5, 5.75)

    # Branch 3: 15–30%
    box(10.5, 7.45, 3.2, 0.75,
        '15–30% missing\n⚠  Elevated risk',
        fc=WARN_BG, ec='#e65c00')
    arr(9.8, 8.55, 9.8, 7.825, '15–30%', '#e65c00')
    ax.plot([9.8, 10.5], [8.55, 8.55], color='#2c5f8a', lw=1.4)

    box(10.5, 6.55, 3.2, 1.45,
        'MissForest  +  Multiple Imputation (M ≥ 5)\n'
        '• Sensitivity analysis mandatory\n'
        '• Compare results with / without imputation\n'
        '• Report missingness pattern explicitly',
        fc=WARN_BG, ec='#e65c00', fs=7.8)
    arr(10.5, 7.075, 10.5, 7.275)

    # Branch 4: > 30%
    box(6.5, 4.65, 12.5, 0.65,
        '> 30% missing  — ⚠  ALERT ZONE  ⚠',
        fc=ALERT_BG, ec='#c0392b', fs=10, bold=True)
    arr(6.5, 5.325, 6.5, 4.975)

    box(6.5, 3.55, 12.5, 1.55,
        'MissForest  +  Multiple Imputation (M ≥ 10)   |   Sensitivity analysis mandatory\n'
        '• Consider excluding variables with > 40% missing\n'
        '• If MNAR suspected → Selection models (Heckman, pattern-mixture) required\n'
        '• Inverse probability weighting if mechanism is partially known\n'
        '• Consult a biostatistician — standard imputation methods are unreliable under MNAR',
        fc=ALERT_BG, ec='#c0392b', fs=8)
    arr(6.5, 4.325, 6.5, 3.825)

    # Mechanism summary at bottom
    box(3.5, 2.3, 6.5, 1.35,
        'Missingness Mechanism Guide\n'
        'MCAR → Standard imputation methods acceptable\n'
        'MAR  → MICE or MissForest recommended (M ≥ 5)\n'
        'MNAR → Selection models + sensitivity analyses mandatory',
        fc=FOOT_BG, ec='#6a0dad', fs=8.5)

    box(9.9, 2.3, 5.5, 1.35,
        'Computational Trade-off\n'
        'Median/Mode: ~13 ms/run\n'
        'KNN: ~90 ms  ·  MICE: ~3 s\n'
        'MissForest: ~86 s  (N ≈ 300)',
        fc='#f5f5f5', ec='#555555', fs=8.5)

    arr(6.5, 2.775, 6.5, 2.77)

    plt.tight_layout(pad=0.1)
    out = os.path.join(OUT_DIR, 'Fig7_DecisionAlgorithm.tif')
    fig.savefig(out, dpi=300, format='tiff')
    print(f'  Saved {out}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Boxplot: RMSE Distribution by Method and Missing Rate (MCAR)
# ─────────────────────────────────────────────────────────────────────────────
def fig8_boxplot_rmse():
    cont = mc[(mc['Mecanisme'] == 'MCAR') &
              (mc['Type'] == 'Continue') &
              (~mc['Variable'].isin(['Lasilix 40 mg', 'Lexomil 6 mg',
                                     'Metformine 500 mg', 'HOMAS%']))].copy()

    rates = ['10%', '20%', '40%']
    n_rates = len(rates)
    n_methods = len(METHOD_ORDER)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # positions: group by rate, methods within group
    group_width = n_methods + 1.5
    all_positions = []
    all_data = []
    all_colors = []

    for gi, rate in enumerate(rates):
        for mi, method in enumerate(METHOD_ORDER):
            sub = cont[(cont['Taux'] == rate) & (cont['Methode'] == method)]['RMSE'].dropna()
            pos = gi * group_width + mi
            all_positions.append(pos)
            all_data.append(sub.values)
            all_colors.append(METHOD_COLORS[method])

    bp = ax.boxplot(all_data, positions=all_positions, widths=0.7,
                    patch_artist=True, notch=False,
                    medianprops=dict(color='#e74c3c', linewidth=2.0),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5,
                                   markeredgewidth=0.5))

    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Group tick labels
    group_centers = [gi * group_width + (n_methods - 1) / 2 for gi in range(n_rates)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f'Missing Rate: {r}' for r in rates], fontsize=10, fontweight='bold')

    ax.set_ylabel('RMSE', fontsize=10, labelpad=4)
    ax.set_title('Distribution of RMSE by Imputation Method and Missing Rate (MCAR, Continuous Variables)',
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('')

    # Method legend
    handles = [mpatches.Patch(color=METHOD_COLORS[m], alpha=0.7, label=m) for m in METHOD_ORDER]
    ax.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.9)

    # Vertical separators between rate groups
    for gi in range(1, n_rates):
        sep = gi * group_width - 0.75
        ax.axvline(sep, color='#bbbbbb', linestyle='--', linewidth=1.0)

    plt.tight_layout(pad=0.5)
    out = os.path.join(OUT_DIR, 'Fig8_Boxplot_RMSE_MCAR.tif')
    fig.savefig(out, dpi=300, format='tiff')
    print(f'  Saved {out}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating publication-quality figures...\n')

    print('[1/8] Study Design Flowchart')
    fig1_flowchart()

    print('[2/8] Missing Data Pattern Heatmap')
    fig2_missing_heatmap()

    print('[3/8] RMSE Curves by Mechanism')
    fig3_rmse_curves()

    print('[4/8] RMSE Heatmap per Variable (MCAR 40%)')
    fig4_rmse_heatmap()

    print('[5/8] Radar Plot (Multi-metric, MCAR)')
    fig5_radar()

    print('[6/8] Variance Ratio Conservation')
    fig6_variance_ratio()

    print('[7/8] Evidence-Based Decision Algorithm')
    fig7_decision_tree()

    print('[8/8] Boxplot RMSE Distribution (MCAR)')
    fig8_boxplot_rmse()

    print(f'\nAll figures saved to: {OUT_DIR}')
    print('Format: TIFF, 300 DPI, journal-ready')
