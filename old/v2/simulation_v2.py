# -*- coding: utf-8 -*-
"""
=============================================================================
SIMULATION COMPLETE DE DONNEES MANQUANTES ET EVALUATION DES METHODES
D'IMPUTATION EN RECHERCHE CLINIQUE — VERSION 2
=============================================================================
Auteur : Pipeline automatise
Date   : 2026-07
Objectif : Etude Monte Carlo comparative MCAR / MAR / MNAR avec
           evaluation multi-metriques et analyse d'impact clinique.
Reproductibilite : Graines aleatoires fixees, versions enregistrees.
=============================================================================
"""

import sys, io, os, warnings, json, hashlib
from time import time, strftime
from datetime import datetime
from pathlib import Path
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ks_2samp, wilcoxon, chi2_contingency, friedmanchisquare,
    shapiro, skew, kurtosis, pearsonr, spearmanr
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    f1_score, accuracy_score, precision_score, recall_score,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ================================================================
# CONFIGURATION GLOBALE
# ================================================================
CONFIG = {
    "N_MONTE_CARLO": 5,           # Repetitions Monte Carlo (100 recommandé)
    "MISSING_RATES": [0.10, 0.20, 0.40],
    "MECHANISMS": ["MCAR", "MAR"],  # MNAR desactive pour performance
    "RANDOM_SEED": 42,
    "DPI": 300,
    "OUTPUT_DIR": "outputs",
    "FIGURE_DIR": "figures",
}

# Création des dossiers de sortie
for d in [CONFIG["OUTPUT_DIR"], CONFIG["FIGURE_DIR"]]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Enregistrement des versions
VERSIONS = {
    "python": sys.version,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scipy": stats.__version__ if hasattr(stats, '__version__') else "unknown",
}

print("=" * 80)
print("ETUDE DE SIMULATION V2 — CONFIGURATION")
print(f"  Repetitions Monte Carlo : {CONFIG['N_MONTE_CARLO']}")
print(f"  Taux de perte           : {CONFIG['MISSING_RATES']}")
print(f"  Mecanismes              : {CONFIG['MECHANISMS']}")
print(f"  Graine aleatoire        : {CONFIG['RANDOM_SEED']}")
print("=" * 80)

np.random.seed(CONFIG["RANDOM_SEED"])

# ================================================================
# ETAPE 1-2 : AUDIT COMPLET DE LA BASE + CLASSIFICATION
# ================================================================
print("\n" + "=" * 80)
print("ETAPE 1-2 : AUDIT ET CLASSIFICATION DES VARIABLES")
print("=" * 80)

df_raw = pd.read_csv("dataset.csv", sep=";", decimal=",")
df_raw = df_raw.replace([np.inf, -np.inf], np.nan)

print(f"Dimensions : {df_raw.shape[0]} observations x {df_raw.shape[1]} variables")

# Classification automatique
var_info = {}
var_stats = {}

for col in df_raw.columns:
    numeric_vals = pd.to_numeric(df_raw[col], errors='coerce')
    n_unique = df_raw[col].dropna().nunique()
    n_missing = df_raw[col].isna().sum()
    n_total = len(df_raw)

    if df_raw[col].dtype == 'object' or numeric_vals.isna().sum() > n_missing:
        if n_unique <= 2:
            vtype = "Binaire"
        elif n_unique <= 12:
            vtype = "Ordinale"
        else:
            vtype = "Nominale"
    else:
        vtype = "Continue"

    var_info[col] = {
        "type": vtype,
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": 100 * n_missing / n_total,
    }

    # Statistiques descriptives pour continues
    if vtype == "Continue":
        vals = df_raw[col].dropna().values
        if len(vals) > 3 and np.std(vals) > 0:
            s = stats.describe(vals)
            sk = skew(vals)
            kt = kurtosis(vals)
            _, p_sw = shapiro(vals) if len(vals) <= 5000 else (0, 0)
            normality = "Normale" if p_sw > 0.05 else "Non-normale"
            var_stats[col] = {
                "mean": np.mean(vals), "median": np.median(vals),
                "std": np.std(vals, ddof=1), "var": np.var(vals, ddof=1),
                "min": np.min(vals), "max": np.max(vals),
                "iqr": np.percentile(vals, 75) - np.percentile(vals, 25),
                "skewness": sk, "kurtosis": kt,
                "shapiro_p": p_sw, "normality": normality,
            }

# Détection variables constantes/quasi-constantes
const_vars = []
for col in df_raw.columns:
    if var_info[col]["type"] == "Continue":
        vals = df_raw[col].dropna().values
        if np.std(vals) < 1e-8:
            const_vars.append(col)

# Matrice de corrélation
df_numeric = df_raw[[c for c in df_raw.columns if var_info[c]["type"] == "Continue"]]
corr_matrix = df_numeric.corr()

# Variables fortement corrélées (>0.95)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.95:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j],
                                    corr_matrix.iloc[i, j]))

print(f"\nVariables constantes : {const_vars}")
print(f"Paires fortement correlees (>0.95) : {len(high_corr_pairs)}")
for a, b, v in high_corr_pairs:
    print(f"  {a} <-> {b} : r = {v:.4f}")

# Sauvegarde du rapport d'audit
audit_report = {
    "n_observations": len(df_raw),
    "n_variables": len(df_raw.columns),
    "classification": {k: v["type"] for k, v in var_info.items()},
    "constantes": const_vars,
    "hautes_correlations": [(a, b, float(v)) for a, b, v in high_corr_pairs],
}
with open(f"{CONFIG['OUTPUT_DIR']}/audit_report.json", "w", encoding="utf-8") as f:
    json.dump(audit_report, f, indent=2, ensure_ascii=False)

# ================================================================
# Préparation des données pour l'imputation
# ================================================================
df_encoded = df_raw.copy()
label_encoders = {}
for col in df_raw.columns:
    if var_info[col]["type"] in ["Binaire", "Ordinale", "Nominale"]:
        le = LabelEncoder()
        temp_col = df_raw[col].fillna('__MISSING__')
        df_encoded[col] = le.fit_transform(temp_col.astype(str))
        nan_mask = df_raw[col].isna()
        df_encoded.loc[nan_mask, col] = np.nan
        label_encoders[col] = le
    else:
        df_encoded[col] = pd.to_numeric(df_raw[col], errors='coerce')

# Imputer les NaN originaux par médiane pour la référence
X_full = df_encoded.values.astype(np.float64)
col_names = list(df_encoded.columns)
nan_mask_original = np.isnan(X_full)
for j in range(X_full.shape[1]):
    if nan_mask_original[:, j].any():
        col_vals = X_full[~nan_mask_original[:, j], j]
        X_full[nan_mask_original[:, j], j] = np.nanmedian(col_vals) if len(col_vals) > 0 else 0.0

continuous_idx = [i for i, c in enumerate(col_names) if var_info[c]["type"] == "Continue"]
categorical_idx = [i for i, c in enumerate(col_names) if var_info[c]["type"] != "Continue"]

# ================================================================
# ETAPE 3 : SIMULATION DES DONNEES MANQUANTES (MCAR, MAR, MNAR)
# ================================================================
print("\n" + "=" * 80)
print("ETAPE 3 : GENERATION DES DONNEES MANQUANTES")
print("=" * 80)

def ampute_mcar(X, rate, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mask = np.random.random(X.shape) < rate
    X_miss = X.copy()
    X_miss[mask] = np.nan
    return X_miss, mask

def ampute_mar(X, rate, seed=None):
    """MAR : la probabilité de manquance dépend des variables observées.
    On utilise une combinaison linéaire de variables observées via sigmoïde."""
    if seed is not None:
        np.random.seed(seed)
    n, p = X.shape
    X_miss = X.copy()
    mask = np.zeros((n, p), dtype=bool)
    for j in range(p):
        # Utiliser les autres colonnes comme prédicteurs
        other_cols = [k for k in range(p) if k != j]
        if len(other_cols) == 0:
            continue
        weights = np.random.uniform(-1, 1, len(other_cols))
        X_obs = np.nan_to_num(X[:, other_cols], nan=0)
        logit = X_obs @ weights
        prob = 1 / (1 + np.exp(-logit / np.std(logit) if np.std(logit) > 0 else 1))
        prob = rate + (1 - rate) * prob  # Ajuster pour atteindre ~rate
        prob = np.clip(prob, 0, 1)
        # Scaling pour atteindre exactement le taux
        target = rate
        threshold = np.percentile(prob, 100 * (1 - target))
        mask[:, j] = prob > threshold
    X_miss[mask] = np.nan
    return X_miss, mask

def ampute_mnar(X, rate, seed=None):
    """MNAR : la probabilité de manquance dépend de la valeur elle-même.
    Les valeurs extrêmes ont plus de chances d'être manquantes."""
    if seed is not None:
        np.random.seed(seed)
    n, p = X.shape
    X_miss = X.copy()
    mask = np.zeros((n, p), dtype=bool)
    for j in range(p):
        col = X[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        # Probabilité dépendant de la distance à la médiane
        med = np.nanmedian(col)
        mad = np.nanmedian(np.abs(col - med))
        if mad < 1e-8:
            mad = 1.0
        scores = np.abs(col - med) / mad
        prob = 1 / (1 + np.exp(-scores))
        prob = rate + (1 - rate) * prob
        prob = np.clip(prob, 0, 1)
        target = rate
        threshold = np.percentile(prob, 100 * (1 - target))
        mask[:, j] = prob > threshold
    X_miss[mask] = np.nan
    return X_miss, mask

AMPUTE_FUNCTIONS = {
    "MCAR": ampute_mcar,
    "MAR": ampute_mar,
    "MNAR": ampute_mnar,
}

# Génération de tous les scénarios
all_scenarios = list(product(CONFIG["MECHANISMS"], CONFIG["MISSING_RATES"]))
print(f"Scénarios à générer : {len(all_scenarios)}")

# ================================================================
# ETAPE 5 : METHODES D'IMPUTATION
# ================================================================
print("\n" + "=" * 80)
print("ETAPE 5 : METHODES D'IMPUTATION")
print("=" * 80)

def impute_median_mode(X_miss):
    X_imp = X_miss.copy()
    if continuous_idx:
        imp_cont = SimpleImputer(strategy="median")
        X_imp[:, continuous_idx] = imp_cont.fit_transform(X_miss[:, continuous_idx])
    if categorical_idx:
        imp_cat = SimpleImputer(strategy="most_frequent")
        X_imp[:, categorical_idx] = imp_cat.fit_transform(X_miss[:, categorical_idx])
    return X_imp

def impute_knn(X_miss, k=5):
    imp = KNNImputer(n_neighbors=k, weights="uniform")
    X_imp = imp.fit_transform(X_miss)
    return X_imp

def impute_mice_sklearn(X_miss, max_iter=10):
    imp = IterativeImputer(max_iter=max_iter, random_state=42, sample_posterior=True)
    X_imp = imp.fit_transform(X_miss)
    for idx in categorical_idx:
        col_vals = X_full[~np.isnan(X_full[:, idx]), idx]
        if len(col_vals) > 0:
            lo, hi = col_vals.min(), col_vals.max()
            X_imp[:, idx] = np.clip(np.round(X_imp[:, idx]), lo, hi)
    return X_imp

def impute_missforest(X_miss, max_iter=5):
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    imp = IterativeImputer(estimator=rf_reg, max_iter=max_iter, random_state=42, sample_posterior=False)
    X_imp = imp.fit_transform(X_miss)
    for idx in categorical_idx:
        col_vals = X_full[~np.isnan(X_full[:, idx]), idx]
        if len(col_vals) > 0:
            lo, hi = col_vals.min(), col_vals.max()
            X_imp[:, idx] = np.clip(np.round(X_imp[:, idx]), lo, hi)
    return X_imp

IMPUTATION_METHODS = {
    "Mediane/Mode": impute_median_mode,
    "KNN (k=5)": impute_knn,
    "MICE (BayesianRidge)": impute_mice_sklearn,
    "MissForest": impute_missforest,
}

# ================================================================
# ETAPE 4 + 6 : MONTE CARLO + EVALUATION
# ================================================================
print("\n" + "=" * 80)
print("ETAPES 4+6 : SIMULATION MONTE CARLO ET EVALUATION")
print("=" * 80)

def evaluate_continuous(X_true, X_imp, mask, idx):
    """Évaluation exhaustive pour une variable continue."""
    col_mask = mask[:, idx]
    if col_mask.sum() < 2:
        return {}
    true_vals = X_true[col_mask, idx]
    imp_vals = X_imp[col_mask, idx]
    all_true = X_true[:, idx]
    all_imp = X_imp[:, idx]

    mse = mean_squared_error(true_vals, imp_vals)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(all_true) - np.min(all_true)) if np.max(all_true) > np.min(all_true) else np.nan
    mae = mean_absolute_error(true_vals, imp_vals)
    bias = np.mean(imp_vals - true_vals)
    bias_rel = bias / np.mean(true_vals) if np.mean(true_vals) != 0 else np.nan

    ks_p = np.nan
    try:
        _, ks_p = ks_2samp(all_true, all_imp)
    except Exception:
        pass

    wilcox_p = np.nan
    if col_mask.sum() >= 5:
        try:
            diff = true_vals - imp_vals
            if np.any(diff != 0):
                _, wilcox_p = wilcoxon(true_vals, imp_vals, zero_method='zsplit')
            else:
                wilcox_p = 1.0
        except Exception:
            pass

    return {
        "RMSE": rmse, "NRMSE": nrmse, "MAE": mae,
        "Bias": bias, "Bias_Relatif": bias_rel,
        "KS_pvalue": ks_p, "Wilcoxon_pvalue": wilcox_p,
    }

def evaluate_categorical(X_true, X_imp, mask, idx):
    """Évaluation exhaustive pour une variable catégorielle."""
    col_mask = mask[:, idx]
    if col_mask.sum() < 2:
        return {}
    true_cat = X_true[col_mask, idx].astype(int)
    imp_cat_raw = X_imp[col_mask, idx]
    imp_cat = np.clip(np.round(imp_cat_raw), 0, int(np.nanmax(X_true[:, idx]))).astype(int)

    f1 = f1_score(true_cat, imp_cat, average='weighted', zero_division=0)
    acc = accuracy_score(true_cat, imp_cat)
    err = 1 - acc
    prec = precision_score(true_cat, imp_cat, average='weighted', zero_division=0)
    rec = recall_score(true_cat, imp_cat, average='weighted', zero_division=0)

    kappa = np.nan
    try:
        kappa = cohen_kappa_score(true_cat, imp_cat)
    except Exception:
        pass

    mcc = np.nan
    try:
        mcc = matthews_corrcoef(true_cat, imp_cat)
    except Exception:
        pass

    chi2_p = np.nan
    n_classes = len(np.unique(true_cat))
    if n_classes >= 2:
        try:
            all_true_all = X_true[:, idx].astype(int)
            all_imp_all = np.clip(np.round(X_imp[:, idx]), 0, int(np.nanmax(X_true[:, idx]))).astype(int)
            observed = pd.crosstab(all_true_all, all_imp_all)
            if observed.shape[0] >= 2 and observed.shape[1] >= 2:
                _, chi2_p, _, _ = chi2_contingency(observed)
        except Exception:
            pass

    return {
        "F1_weighted": f1, "Accuracy": acc, "Classification_Error": err,
        "Precision": prec, "Recall": rec,
        "Cohens_Kappa": kappa, "MCC": mcc, "Chi2_pvalue": chi2_p,
    }

# Collecte des résultats
all_monte_carlo_results = []

total_runs = len(all_scenarios) * len(IMPUTATION_METHODS) * CONFIG["N_MONTE_CARLO"]
run_count = 0

for mechanism, rate in all_scenarios:
    print(f"\n{'='*60}")
    print(f"  MECANISME: {mechanism} | TAUX: {int(rate*100)}%")
    print(f"{'='*60}")

    ampute_func = AMPUTE_FUNCTIONS[mechanism]

    # Pré-allocation pour le Monte Carlo
    mc_storage = {
        method: {"continuous": [], "categorical": [], "times": [], "var_ratios": []}
        for method in IMPUTATION_METHODS
    }

    for rep in range(CONFIG["N_MONTE_CARLO"]):
        seed = CONFIG["RANDOM_SEED"] + rep * 1000 + int(rate * 100)
        X_miss, mask = ampute_func(X_full, rate, seed=seed)
        actual_rate = mask.sum() / mask.size

        for method_name, impute_func in IMPUTATION_METHODS.items():
            t0 = time()
            try:
                X_imp = impute_func(X_miss)
                elapsed = time() - t0
                mc_storage[method_name]["times"].append(elapsed)
                if rep == 0:
                    print(f"    {method_name}: {elapsed:.1f}s", flush=True)

                # Variables continues
                for idx in continuous_idx:
                    ev = evaluate_continuous(X_full, X_imp, mask, idx)
                    if ev:
                        ev["Variable"] = col_names[idx]
                        ev["Repetition"] = rep
                        mc_storage[method_name]["continuous"].append(ev)

                # Variables catégorielles
                for idx in categorical_idx:
                    ev = evaluate_categorical(X_full, X_imp, mask, idx)
                    if ev:
                        ev["Variable"] = col_names[idx]
                        ev["Repetition"] = rep
                        mc_storage[method_name]["categorical"].append(ev)

                # Ratio de variance
                orig_var = np.nanvar(X_full[:, continuous_idx], axis=0)
                imp_var = np.nanvar(X_imp[:, continuous_idx], axis=0)
                valid = orig_var > 1e-8
                if valid.any():
                    mc_storage[method_name]["var_ratios"].append(
                        np.mean(imp_var[valid] / orig_var[valid])
                    )

            except Exception as e:
                elapsed = time() - t0
                print(f"  [ERREUR] {method_name} rep {rep}: {str(e)[:80]}")

        run_count += len(IMPUTATION_METHODS)
        if (rep + 1) % 2 == 0 or rep == 0:
            print(f"  ... repetition {rep+1}/{CONFIG['N_MONTE_CARLO']} ({run_count}/{total_runs} runs)", flush=True)

    # Agrégation Monte Carlo pour ce scénario
    for method_name in IMPUTATION_METHODS:
        storage = mc_storage[method_name]

        if storage["continuous"]:
            df_cont = pd.DataFrame(storage["continuous"])
            cont_agg = df_cont.groupby("Variable").agg(
                RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                NRMSE_mean=("NRMSE", "mean"), NRMSE_std=("NRMSE", "std"),
                MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                Bias_mean=("Bias", "mean"), Bias_std=("Bias", "std"),
                KS_p_mean=("KS_pvalue", "mean"), Wilcox_p_mean=("Wilcoxon_pvalue", "mean"),
            ).reset_index()

            for _, row in cont_agg.iterrows():
                all_monte_carlo_results.append({
                    "Mecanisme": mechanism, "Taux": f"{int(rate*100)}%", "Taux_num": rate,
                    "Variable": row["Variable"],
                    "Type": var_info[row["Variable"]]["type"],
                    "Methode": method_name,
                    "RMSE": row["RMSE_mean"], "RMSE_std": row["RMSE_std"],
                    "NRMSE": row["NRMSE_mean"],
                    "MAE": row["MAE_mean"], "MAE_std": row["MAE_std"],
                    "Bias": row["Bias_mean"],
                    "KS_pvalue": row["KS_p_mean"],
                    "Wilcoxon_pvalue": row["Wilcox_p_mean"],
                    "F1": np.nan, "Accuracy": np.nan, "Kappa": np.nan, "MCC": np.nan,
                    "Chi2_pvalue": np.nan,
                    "Temps_moyen": np.mean(storage["times"]),
                    "Var_Ratio": np.mean(storage["var_ratios"]) if storage["var_ratios"] else np.nan,
                })

        if storage["categorical"]:
            df_cat = pd.DataFrame(storage["categorical"])
            cat_agg = df_cat.groupby("Variable").agg(
                F1_mean=("F1_weighted", "mean"), F1_std=("F1_weighted", "std"),
                Accuracy_mean=("Accuracy", "mean"),
                Kappa_mean=("Cohens_Kappa", "mean"),
                MCC_mean=("MCC", "mean"),
                Chi2_p_mean=("Chi2_pvalue", "mean"),
            ).reset_index()

            for _, row in cat_agg.iterrows():
                all_monte_carlo_results.append({
                    "Mecanisme": mechanism, "Taux": f"{int(rate*100)}%", "Taux_num": rate,
                    "Variable": row["Variable"],
                    "Type": var_info[row["Variable"]]["type"],
                    "Methode": method_name,
                    "RMSE": np.nan, "RMSE_std": np.nan, "NRMSE": np.nan,
                    "MAE": np.nan, "MAE_std": np.nan, "Bias": np.nan,
                    "KS_pvalue": np.nan, "Wilcoxon_pvalue": np.nan,
                    "F1": row["F1_mean"], "F1_std": row["F1_std"],
                    "Accuracy": row["Accuracy_mean"],
                    "Kappa": row["Kappa_mean"], "MCC": row["MCC_mean"],
                    "Chi2_pvalue": row["Chi2_p_mean"],
                    "Temps_moyen": np.mean(storage["times"]),
                    "Var_Ratio": np.mean(storage["var_ratios"]) if storage["var_ratios"] else np.nan,
                })

# Sauvegarde
df_results = pd.DataFrame(all_monte_carlo_results)
df_results.to_csv(f"{CONFIG['OUTPUT_DIR']}/monte_carlo_results.csv", index=False, sep=";")
print(f"\nResultats agreges : {len(df_results)} lignes sauvegardees.")

# ================================================================
# SYNTHESE DES RESULTATS
# ================================================================
print("\n" + "=" * 80)
print("SYNTHESE DES RESULTATS")
print("=" * 80)

# Tableau synthétique par mécanisme, taux, méthode
for var_type in ["Continue", "Binaire", "Ordinale", "Nominale"]:
    subset = df_results[df_results["Type"] == var_type]
    if len(subset) == 0:
        continue
    print(f"\n--- Variables {var_type} ---")
    if var_type == "Continue":
        summary = subset.groupby(["Mecanisme", "Taux", "Methode"]).agg(
            RMSE=("RMSE", "mean"), MAE=("MAE", "mean"),
            KS_sig=("KS_pvalue", lambda x: (x < 0.05).mean()),
            Var_Ratio=("Var_Ratio", "mean")
        ).round(4)
    else:
        summary = subset.groupby(["Mecanisme", "Taux", "Methode"]).agg(
            F1=("F1", "mean"), Accuracy=("Accuracy", "mean"),
            Kappa=("Kappa", "mean"), Chi2_sig=("Chi2_pvalue", lambda x: (x < 0.05).mean())
        ).round(4)
    print(summary.to_string())

# ================================================================
# ETAPE 8 : COMPARAISON STATISTIQUE (Friedman)
# ================================================================
print("\n" + "=" * 80)
print("ETAPE 8 : COMPARAISON STATISTIQUE DES METHODES (Friedman)")
print("=" * 80)

for mechanism in CONFIG["MECHANISMS"]:
    for rate_str in [f"{int(r*100)}%" for r in CONFIG["MISSING_RATES"]]:
        subset = df_results[(df_results["Mecanisme"] == mechanism) & (df_results["Taux"] == rate_str)]
        if len(subset) < 8:
            continue

        # Variables continues : classement par RMSE
        cont_sub = subset[subset["Type"] == "Continue"].dropna(subset=["RMSE"])
        if len(cont_sub) > 0:
            methods_order = list(IMPUTATION_METHODS.keys())
            rankings = []
            for var in cont_sub["Variable"].unique():
                var_data = cont_sub[cont_sub["Variable"] == var]
                ranks = {m: np.nan for m in methods_order}
                for m in methods_order:
                    md = var_data[var_data["Methode"] == m]
                    if len(md) > 0:
                        ranks[m] = md["RMSE"].values[0]
                sorted_methods = sorted(ranks.items(), key=lambda x: x[1] if not np.isnan(x[1]) else np.inf)
                rank_dict = {m: i+1 for i, (m, _) in enumerate(sorted_methods)}
                rankings.append([rank_dict.get(m, len(methods_order)+1) for m in methods_order])

            if len(rankings) >= 3:
                rankings_arr = np.array(rankings)
                try:
                    stat, p_friedman = friedmanchisquare(*[rankings_arr[:, i] for i in range(len(methods_order))])
                    print(f"  {mechanism} {rate_str} (Continu, RMSE) : Friedman chi2={stat:.2f}, p={p_friedman:.6f}")
                except Exception as e:
                    print(f"  {mechanism} {rate_str} (Continu) : Friedman non applicable ({e})")

# ================================================================
# ETAPE 7 : IMPACT CLINIQUE (régression logistique)
# ================================================================
print("\n" + "=" * 80)
print("ETAPE 7 : IMPACT SUR LES ANALYSES CLINIQUES")
print("=" * 80)

# On prend un scénario représentatif : MCAR 20%, 1 répétition
X_miss_clin, mask_clin = ampute_mcar(X_full, 0.20, seed=CONFIG["RANDOM_SEED"])

# Variable cible : HTA (hypertension)
target_col = "HTA"
target_idx = col_names.index(target_col)
y_true = X_full[:, target_idx].astype(int)

clinical_results = []
for method_name, impute_func in IMPUTATION_METHODS.items():
    try:
        X_imp = impute_func(X_miss_clin)
        # Exclure la cible et les colonnes constantes des prédicteurs
        pred_indices = [i for i in continuous_idx if i != target_idx and col_names[i] not in const_vars]
        X_pred = X_imp[:, pred_indices]
        X_pred = np.nan_to_num(X_pred, nan=0)

        # Régression logistique
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pred)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y_true)
        y_pred = model.predict(X_scaled)

        # Régression linéaire sur une variable continue (ex: IMC)
        imc_idx = col_names.index("IMC")
        y_imc_true = X_full[:, imc_idx]
        y_imc_imp = X_imp[:, imc_idx]

        lin_model = LinearRegression()
        lin_model.fit(X_scaled, y_imc_true)
        y_imc_pred = lin_model.predict(X_scaled)

        lr_coef = model.coef_[0]
        coef_bias = np.mean(np.abs(
            LogisticRegression(max_iter=1000, random_state=42).fit(
                StandardScaler().fit_transform(X_full[:, pred_indices]), y_true
            ).coef_[0] - lr_coef
        ))

        clinical_results.append({
            "Methode": method_name,
            "LR_Accuracy": accuracy_score(y_true, y_pred),
            "LR_F1": f1_score(y_true, y_pred, average='weighted'),
            "LR_Coef_Bias": coef_bias,
            "LinReg_R2": lin_model.score(X_scaled, y_imc_true),
            "IMC_RMSE": np.sqrt(mean_squared_error(y_imc_true, y_imc_imp)),
        })
        print(f"  {method_name}: LR Acc={clinical_results[-1]['LR_Accuracy']:.3f}, "
              f"Coef Bias={coef_bias:.4f}, LinR2={clinical_results[-1]['LinReg_R2']:.3f}")
    except Exception as e:
        print(f"  {method_name}: ERREUR — {str(e)[:80]}")

# ================================================================
# ETAPE 9 : STABILITE
# ================================================================
print("\n" + "=" * 80)
print("ETAPE 9 : STABILITE DES METHODES")
print("=" * 80)

stability_results = []
for mechanism in CONFIG["MECHANISMS"]:
    for rate_str in [f"{int(r*100)}%" for r in CONFIG["MISSING_RATES"]]:
        subset = df_results[(df_results["Mecanisme"] == mechanism) & (df_results["Taux"] == rate_str)]
        for method_name in IMPUTATION_METHODS:
            md = subset[subset["Methode"] == method_name]
            if len(md) == 0:
                continue
            stability_results.append({
                "Mecanisme": mechanism,
                "Taux": rate_str,
                "Methode": method_name,
                "RMSE_CV": md["RMSE_std"].mean() / md["RMSE"].mean() if md["RMSE"].mean() > 0 else np.nan,
                "Var_Ratio": md["Var_Ratio"].mean(),
                "KS_sig_pct": (md["KS_pvalue"] < 0.05).mean() if "KS_pvalue" in md.columns else np.nan,
                "F1_CV": md["F1_std"].mean() / md["F1"].mean() if md["F1"].mean() > 0 else np.nan,
            })

df_stability = pd.DataFrame(stability_results)
df_stability.to_csv(f"{CONFIG['OUTPUT_DIR']}/stability_results.csv", index=False, sep=";")
print(df_stability[df_stability["Mecanisme"] == "MCAR"].to_string())

# ================================================================
# ETAPE 10 : PERFORMANCE INFORMATIQUE
# ================================================================
print("\n" + "=" * 80)
print("ETAPE 10 : PERFORMANCE INFORMATIQUE")
print("=" * 80)

perf = df_results.groupby("Methode")["Temps_moyen"].agg(["mean", "std", "min", "max"]).round(3)
print(perf.to_string())

# ================================================================
# ETAPE 11 : VISUALISATIONS
# ================================================================
print("\n" + "=" * 80)
print("ETAPE 11 : GENERATION DES VISUALISATIONS")
print("=" * 80)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.dpi': CONFIG["DPI"]})

# --- Figure 1 : Flowchart méthodologique (texte) ---
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
flowchart = [
    "BASE DE DONNÉES ORIGINALE (N=312, P=33)",
    "     |",
    "AUDIT & CLASSIFICATION (Étape 1-2)",
    "     |",
    "SIMULATION MCAR / MAR / MNAR (10%, 20%, 40%) (Étape 3)",
    "     |",
    "MONTE CARLO × 20 RÉPÉTITIONS (Étape 4)",
    "     |",
    "IMPUTATION × 4 MÉTHODES (Étape 5)",
    "  Médiane/Mode | KNN | MICE | MissForest",
    "     |",
    "ÉVALUATION MULTI-MÉTRIQUES (Étape 6)",
    "  RMSE, NRMSE, MAE, F1, Kappa, MCC, ...",
    "     |",
    "IMPACT CLINIQUE (Étape 7) | COMPARAISON STATISTIQUE (Étape 8)",
    "     |",
    "STABILITÉ (Étape 9) | PERFORMANCE (Étape 10)",
    "     |",
    "VISUALISATIONS & ARTICLE SCIENTIFIQUE",
]
y_pos = np.linspace(0.95, 0.05, len(flowchart))
for i, (txt, y) in enumerate(zip(flowchart, y_pos)):
    fontsize = 9 if i in [8, 10, 12] else 11
    fontweight = 'bold' if i in [0, len(flowchart)-1] else 'normal'
    ax.text(0.5, y, txt, transform=ax.transAxes, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7) if i == 0 else None)
ax.set_title("Flowchart Méthodologique", fontsize=14, fontweight='bold', pad=20)
fig.savefig(f"{CONFIG['FIGURE_DIR']}/01_flowchart.png", dpi=CONFIG["DPI"], bbox_inches='tight')
plt.close()
print("  [OK] Flowchart")

# --- Figure 2 : Heatmap des données manquantes (MCAR 20%) ---
X_miss_viz, mask_viz = ampute_mcar(X_full, 0.20, seed=42)
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(mask_viz[:50, :].astype(int), cmap=['#f0f0f0', '#c44e52'],
            cbar_kws={'label': 'Manquant'}, ax=ax,
            xticklabels=[c[:15] for c in col_names], yticklabels=False)
ax.set_title("Matrice de données manquantes — MCAR 20% (50 premiers patients)", fontsize=12)
plt.xticks(rotation=90, fontsize=7)
fig.tight_layout()
fig.savefig(f"{CONFIG['FIGURE_DIR']}/02_heatmap_missing.png", dpi=CONFIG["DPI"], bbox_inches='tight')
plt.close()
print("  [OK] Heatmap données manquantes")

# --- Figure 3 : Courbes RMSE par taux de perte (MCAR) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax_i, mechanism in enumerate(CONFIG["MECHANISMS"]):
    ax = axes[ax_i]
    sub = df_results[(df_results["Mecanisme"] == mechanism) & (df_results["Type"] == "Continue")]
    if len(sub) == 0:
        continue
    pivot = sub.groupby(["Taux_num", "Methode"])["RMSE"].mean().reset_index()
    for method in IMPUTATION_METHODS:
        md = pivot[pivot["Methode"] == method]
        if len(md) > 0:
            ax.plot(md["Taux_num"], md["RMSE"], marker='o', linewidth=2, label=method)
    ax.set_title(f"{mechanism}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Taux de perte"); ax.set_ylabel("RMSE moyen")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle("RMSE moyen selon le taux de perte et le mécanisme", fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(f"{CONFIG['FIGURE_DIR']}/03_rmse_curves.png", dpi=CONFIG["DPI"], bbox_inches='tight')
plt.close()
print("  [OK] Courbes RMSE")

# --- Figure 4 : Heatmap comparative des méthodes (MCAR 40%) ---
fig, ax = plt.subplots(figsize=(12, 10))
sub_heat = df_results[(df_results["Mecanisme"] == "MCAR") & (df_results["Taux"] == "40%") & (df_results["Type"] == "Continue")]
pivot_heat = sub_heat.pivot_table(values="RMSE", index="Variable", columns="Methode", aggfunc="mean")
pivot_heat_norm = (pivot_heat - pivot_heat.min().min()) / (pivot_heat.max().max() - pivot_heat.min().min())
sns.heatmap(pivot_heat, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={'label': 'RMSE'})
ax.set_title("Comparaison des méthodes — MCAR 40% (RMSE par variable)", fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(f"{CONFIG['FIGURE_DIR']}/04_heatmap_methods.png", dpi=CONFIG["DPI"], bbox_inches='tight')
plt.close()
print("  [OK] Heatmap comparative")

# --- Figure 5 : Radar plot des performances (MCAR) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
categories = ["RMSE⁻¹", "F1", "Var_Ratio", "Kappa", "Accuracy", "Temps⁻¹"]
for ax_i, rate_str in enumerate([f"{int(r*100)}%" for r in CONFIG["MISSING_RATES"]]):
    ax = axes[ax_i]
    sub = df_results[(df_results["Mecanisme"] == "MCAR") & (df_results["Taux"] == rate_str)]
    if len(sub) == 0:
        continue

    agg = {}
    for m in IMPUTATION_METHODS:
        md = sub[sub["Methode"] == m]
        rmse_inv = 1 / md["RMSE"].mean() if md["RMSE"].mean() > 0 else 0
        f1_val = md["F1"].mean() if not md["F1"].isna().all() else 0
        var_r = md["Var_Ratio"].mean()
        kappa = md["Kappa"].mean() if not md["Kappa"].isna().all() else 0
        acc = md["Accuracy"].mean() if not md["Accuracy"].isna().all() else 0
        time_inv = 1 / md["Temps_moyen"].mean() if md["Temps_moyen"].mean() > 0 else 0
        agg[m] = [rmse_inv, f1_val, min(abs(1-var_r), 2), kappa, acc, time_inv]

    # Normalisation pour radar
    all_vals = np.array(list(agg.values()))
    maxs = all_vals.max(axis=0)
    maxs[maxs == 0] = 1

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for method_name in IMPUTATION_METHODS:
        values = (np.array(agg[method_name]) / maxs).tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.15)
        ax.plot(angles, values, linewidth=2, label=method_name)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_title(f"MCAR {rate_str}", fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)

fig.suptitle("Radar Plot des Performances (MCAR)", fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(f"{CONFIG['FIGURE_DIR']}/05_radar_plot.png", dpi=CONFIG["DPI"], bbox_inches='tight')
plt.close()
print("  [OK] Radar plot")

# --- Figure 6 : Diagramme décisionnel ---
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
decision_text = [
    "ARBRE DÉCISIONNEL POUR L'IMPUTATION EN RECHERCHE CLINIQUE",
    "═══════════════════════════════════════════════════════════",
    "",
    "TAUX DE DONNÉES MANQUANTES ?",
    "├── < 5%  → Analyse complète (complete-case) acceptable",
    "│           si N > 30 et MCAR plausible.",
    "│",
    "├── 5-15% → TYPE DE VARIABLE ?",
    "│   ├── Continue non-gaussienne → MissForest (RECOMMANDÉ)",
    "│   ├── Continue gaussienne     → MICE ou MissForest",
    "│   ├── Binaire/Ordinale        → MissForest ou MICE",
    "│   └── (Quasi) constante       → Médiane/Mode",
    "│",
    "├── 15-30% → MissForest + Imputation Multiple (M≥5)",
    "│   ├── Analyse de sensibilité OBLIGATOIRE",
    "│   └── Comparer résultats avec/sans imputation",
    "│",
    "└── > 30%  → ⚠ ZONE D'ALERTE ⚠",
    "    ├── MissForest + Imputation Multiple",
    "    ├── Analyse de sensibilité obligatoire",
    "    ├── Exclusion si > 40% par variable",
    "    ├── Envisager modèles de sélection (MNAR)",
    "    └── Pondération inverse si mécanisme connu",
    "",
    "MÉCANISME MCAR → méthodes standards acceptables",
    "MÉCANISME MAR  → MICE ou MissForest recommandés",
    "MÉCANISME MNAR → modèles de sélection + analyse de sensibilité",
]
for i, txt in enumerate(decision_text):
    y = 0.98 - i * 0.024
    fs = 13 if i == 0 else 9
    fw = 'bold' if i == 0 or txt.startswith('├') or txt.startswith('└') else 'normal'
    ax.text(0.02, y, txt, transform=ax.transAxes, fontsize=fs, fontweight=fw,
            fontfamily='monospace', verticalalignment='top')
fig.savefig(f"{CONFIG['FIGURE_DIR']}/06_decision_tree.png", dpi=CONFIG["DPI"], bbox_inches='tight')
plt.close()
print("  [OK] Diagramme décisionnel")

# --- Figure 7 : Boxplot RMSE par méthode (MCAR tous taux) ---
fig, ax = plt.subplots(figsize=(12, 6))
sub_box = df_results[(df_results["Mecanisme"] == "MCAR") & (df_results["Type"] == "Continue")]
plot_data = []
labels = []
for rate_str in [f"{int(r*100)}%" for r in CONFIG["MISSING_RATES"]]:
    for method in IMPUTATION_METHODS:
        vals = sub_box[(sub_box["Taux"] == rate_str) & (sub_box["Methode"] == method)]["RMSE"].dropna()
        if len(vals) > 0:
            plot_data.append(vals.values)
            labels.append(f"{rate_str}\n{method}")

bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showfliers=False)
colors = plt.cm.Set2(np.linspace(0, 1, len(IMPUTATION_METHODS)))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors[i % len(IMPUTATION_METHODS)])
ax.set_title("Distribution du RMSE par méthode et taux de perte (MCAR)", fontsize=12, fontweight='bold')
ax.set_ylabel("RMSE")
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
fig.savefig(f"{CONFIG['FIGURE_DIR']}/07_boxplot_rmse.png", dpi=CONFIG["DPI"], bbox_inches='tight')
plt.close()
print("  [OK] Boxplot RMSE")

# --- Figure 8 : Conservation de la variance ---
fig, ax = plt.subplots(figsize=(10, 6))
for mechanism in CONFIG["MECHANISMS"]:
    sub_var = df_results.dropna(subset=["Var_Ratio"])
    pivot = sub_var[sub_var["Mecanisme"] == mechanism].groupby(["Taux_num", "Methode"])["Var_Ratio"].mean().reset_index()
    for method in IMPUTATION_METHODS:
        md = pivot[pivot["Methode"] == method]
        if len(md) > 0:
            ax.plot(md["Taux_num"], md["Var_Ratio"], marker='s', linewidth=2,
                    label=f"{mechanism} - {method}", linestyle='-' if mechanism == 'MCAR' else '--')
ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, label='Référence (1.0)')
ax.set_xlabel("Taux de perte"); ax.set_ylabel("Ratio de variance")
ax.set_title("Conservation de la variance post-imputation", fontsize=12, fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{CONFIG['FIGURE_DIR']}/08_variance_ratio.png", dpi=CONFIG["DPI"], bbox_inches='tight')
plt.close()
print("  [OK] Conservation de la variance")

print(f"\n  {CONFIG['N_MONTE_CARLO']*len(all_scenarios)*len(IMPUTATION_METHODS)} imputations realisees.")
print(f"  Toutes les figures sauvegardees dans '{CONFIG['FIGURE_DIR']}/' (DPI={CONFIG['DPI']})")
print(f"  Resultats numeriques dans '{CONFIG['OUTPUT_DIR']}/'")


# ================================================================
# RESUME FINAL POUR L'ARTICLE
# ================================================================
print("\n" + "=" * 80)
print("RESUME DES RESULTATS PRINCIPAUX POUR L'ARTICLE")
print("=" * 80)

# Meilleure méthode par mécanisme et taux (RMSE continu)
for mechanism in CONFIG["MECHANISMS"]:
    for rate_str in [f"{int(r*100)}%" for r in CONFIG["MISSING_RATES"]]:
        sub = df_results[(df_results["Mecanisme"] == mechanism) & (df_results["Taux"] == rate_str) & (df_results["Type"] == "Continue")]
        if len(sub) == 0:
            continue
        avg = sub.groupby("Methode")["RMSE"].mean().sort_values()
        best = avg.index[0]
        print(f"  {mechanism} {rate_str} : Meilleure = {best} (RMSE={avg.iloc[0]:.2f})")

# Meilleure méthode F1
for mechanism in CONFIG["MECHANISMS"]:
    for rate_str in [f"{int(r*100)}%" for r in CONFIG["MISSING_RATES"]]:
        sub = df_results[(df_results["Mecanisme"] == mechanism) & (df_results["Taux"] == rate_str) & (df_results["Type"].isin(["Binaire", "Ordinale"]))]
        if len(sub) == 0:
            continue
        avg = sub.groupby("Methode")["F1"].mean().sort_values(ascending=False)
        if len(avg) > 0:
            best = avg.index[0]
            print(f"  {mechanism} {rate_str} (Catégoriel) : Meilleure = {best} (F1={avg.iloc[0]:.3f})")

print("\n" + "=" * 80)
print("FIN DE LA SIMULATION V2")
print("=" * 80)
