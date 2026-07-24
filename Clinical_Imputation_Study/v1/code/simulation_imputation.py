# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import warnings
from time import time
from scipy import stats
from scipy.stats import ks_2samp, wilcoxon, chi2_contingency
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import miceforest as mf

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 1. CHARGEMENT
# ============================================================
print("=" * 80)
print("CHARGEMENT ET CLASSIFICATION DES VARIABLES")
print("=" * 80)

df_raw = pd.read_csv("dataset.csv", sep=";", decimal=",")
print(f"Dimensions : {df_raw.shape[0]} observations x {df_raw.shape[1]} variables")

# Nettoyage : remplacer inf par NaN
df_raw = df_raw.replace([np.inf, -np.inf], np.nan)

# Identification des types
var_info = {}
for col in df_raw.columns:
    numeric_vals = pd.to_numeric(df_raw[col], errors='coerce')
    n_nonnum = numeric_vals.isna().sum() - df_raw[col].isna().sum()  # NaN qui ne sont pas des vrais NaN
    n_unique = df_raw[col].dropna().nunique()
    
    if df_raw[col].dtype == 'object' or n_nonnum > 0:
        unique_vals = df_raw[col].dropna().unique()
        if n_unique <= 2:
            var_info[col] = "Binaire"
        elif n_unique <= 12:
            var_info[col] = "Ordinal"
        else:
            var_info[col] = "Catégoriel_nominal"
    else:
        var_info[col] = "Continu"

# Nettoyage spécial des colonnes ratio avec inf
ratio_cols = ['Ratio Tot-c/HDL', 'Ratio LDL-c/HDL', 'Ratio TG/HDL', 'Ratio LogTG/HDL']
for rc in ratio_cols:
    if rc in df_raw.columns:
        df_raw[rc] = pd.to_numeric(df_raw[rc], errors='coerce')
        df_raw[rc] = df_raw[rc].replace([np.inf, -np.inf], np.nan)

print("\nClassification des variables :")
for v, t in var_info.items():
    print(f"  {v:30s} -> {t}")

# Séparation
continuous_vars = [k for k, v in var_info.items() if v == "Continu"]
binary_vars = [k for k, v in var_info.items() if v == "Binaire"]
ordinal_vars = [k for k, v in var_info.items() if v == "Ordinal"]

# Encodage
df_encoded = df_raw.copy()
label_encoders = {}
for col in df_raw.columns:
    if var_info[col] in ["Binaire", "Ordinal", "Catégoriel_nominal"]:
        le = LabelEncoder()
        # Fill NaN temporairement pour l'encodage
        temp_col = df_raw[col].fillna('MISSING_TEMP')
        df_encoded[col] = le.fit_transform(temp_col.astype(str))
        # Remettre NaN la ou c'etait NaN
        nan_mask = df_raw[col].isna()
        df_encoded.loc[nan_mask, col] = np.nan
        label_encoders[col] = le
    else:
        df_encoded[col] = pd.to_numeric(df_raw[col], errors='coerce')

# Convertir en numpy float64
X_full = df_encoded.values.astype(np.float64)
col_names = list(df_encoded.columns)

# Remplacer les NaN residuels dans X_full par la mediane temporairement pour reference
# (certaines colonnes ont des NaN d'origine dans le dataset brut)
nan_mask_original = np.isnan(X_full)
for j in range(X_full.shape[1]):
    if nan_mask_original[:, j].any():
        col_vals = X_full[~nan_mask_original[:, j], j]
        if len(col_vals) > 0:
            X_full[nan_mask_original[:, j], j] = np.nanmedian(col_vals)
        else:
            X_full[nan_mask_original[:, j], j] = 0.0

print("\nCaracterisation de la distribution des variables continues :")
for v in continuous_vars:
    vals = df_raw[v].dropna().values
    if len(vals) > 3 and np.std(vals) > 0:
        skew = stats.skew(vals)
        _, p_shapiro = stats.shapiro(vals) if len(vals) <= 5000 else (0, 0)
        dist_type = "Normale" if p_shapiro > 0.05 else "Asymetrique"
        print(f"  {v:30s} -> Skewness={skew:+.3f}, Shapiro p={p_shapiro:.4f} -> {dist_type}")
    else:
        print(f"  {v:30s} -> variance nulle ou quasi-nulle (variable constante)")

# ============================================================
# 2. MCAR
# ============================================================
print("\n" + "=" * 80)
print("SIMULATION DE PERTE MCAR (10%, 20%, 40%)")
print("=" * 80)

def ampute_mcar(X, rate, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mask = np.random.random(X.shape) < rate
    X_miss = X.copy()
    X_miss[mask] = np.nan
    return X_miss, mask

missing_rates = [0.10, 0.20, 0.40]
amputed_data = {}
for rate in missing_rates:
    X_miss, mask = ampute_mcar(X_full, rate, seed=42 + int(rate * 100))
    amputed_data[rate] = {"X_miss": X_miss, "mask": mask}
    actual_rate = mask.sum() / mask.size
    print(f"  Taux {rate*100:.0f}% -> Taux reel = {actual_rate*100:.2f}%")

# ============================================================
# 3. IMPUTATION
# ============================================================
print("\n" + "=" * 80)
print("IMPUTATION PAR LES 4 METHODES")
print("=" * 80)

def impute_median_mode(X_miss):
    cont_cols = [i for i, c in enumerate(col_names) if var_info[c] == "Continu"]
    cat_cols = [i for i, c in enumerate(col_names) if var_info[c] != "Continu"]
    X_imp = X_miss.copy()
    if cont_cols:
        imp_cont = SimpleImputer(strategy="median")
        X_imp[:, cont_cols] = imp_cont.fit_transform(X_miss[:, cont_cols])
    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        X_imp[:, cat_cols] = imp_cat.fit_transform(X_miss[:, cat_cols])
    return X_imp

def impute_knn(X_miss, n_neighbors=5):
    imp = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    X_imp = imp.fit_transform(X_miss)
    return X_imp

def impute_missforest(X_miss):
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    imp = IterativeImputer(estimator=rf_reg, max_iter=10, random_state=42, sample_posterior=False)
    X_imp = imp.fit_transform(X_miss)
    # Arrondir les variables categorielles
    cat_indices = [i for i, c in enumerate(col_names) if var_info[c] != "Continu"]
    for idx in cat_indices:
        col_vals = X_full[~np.isnan(X_full[:, idx]), idx]
        if len(col_vals) > 0:
            lo, hi = col_vals.min(), col_vals.max()
            X_imp[:, idx] = np.clip(np.round(X_imp[:, idx]), lo, hi)
    return X_imp

def impute_miceforest(X_miss):
    df_miss = pd.DataFrame(X_miss, columns=col_names)
    variable_schema = {}
    for col in col_names:
        variable_schema[col] = "float" if var_info[col] == "Continu" else "int"
    try:
        kernel = mf.ImputationKernel(
            df_miss,
            num_datasets=5,
            variable_schema=variable_schema,
            save_all_iterations_data=False,
            random_state=42
        )
        kernel.mice(iterations=5, verbose=False)
        X_imp = kernel.complete_data(dataset=0).values
        return X_imp
    except Exception:
        pass
    # Fallback : variable_schema as list
    try:
        schema_list = [col for col in col_names]
        kernel = mf.ImputationKernel(
            df_miss,
            num_datasets=5,
            save_all_iterations_data=False,
            random_state=42
        )
        kernel.mice(iterations=5, verbose=False)
        X_imp = kernel.complete_data(dataset=0).values
        return X_imp
    except Exception as e2:
        raise e2


def impute_mice_sklearn(X_miss, max_iter=10):
    """MICE via IterativeImputer (sklearn)."""
    imp = IterativeImputer(max_iter=max_iter, random_state=42, sample_posterior=True)
    X_imp = imp.fit_transform(X_miss)
    # Arrondir les variables categorielles
    cat_indices = [i for i, c in enumerate(col_names) if var_info[c] != "Continu"]
    for idx in cat_indices:
        col_vals = X_full[~np.isnan(X_full[:, idx]), idx]
        if len(col_vals) > 0:
            lo, hi = col_vals.min(), col_vals.max()
            X_imp[:, idx] = np.clip(np.round(X_imp[:, idx]), lo, hi)
    return X_imp

imputation_methods = {
    "Mediane/Mode": impute_median_mode,
    "MICE (sklearn)": impute_mice_sklearn,
    "KNN (k=5)": impute_knn,
    "MissForest": impute_missforest,
}

# ============================================================
# 4. EVALUATION
# ============================================================
print("\n" + "=" * 80)
print("EVALUATION DES METHODES D'IMPUTATION")
print("=" * 80)

def evaluate_continuous(X_true, X_imp, mask, var_indices):
    results = []
    for idx in var_indices:
        col_mask = mask[:, idx]
        if col_mask.sum() < 2:
            continue
        true_vals = X_true[col_mask, idx]
        imp_vals = X_imp[col_mask, idx]
        rmse = np.sqrt(mean_squared_error(true_vals, imp_vals))
        mae = mean_absolute_error(true_vals, imp_vals)
        all_true = X_true[:, idx]
        all_imp = X_imp[:, idx]
        try:
            ks_stat, ks_pval = ks_2samp(all_true, all_imp)
        except Exception:
            ks_pval = np.nan
        if col_mask.sum() >= 5:
            try:
                diff = true_vals - imp_vals
                if np.any(diff != 0):
                    _, wilcox_pval = wilcoxon(true_vals, imp_vals, zero_method='zsplit')
                else:
                    wilcox_pval = 1.0
            except Exception:
                wilcox_pval = np.nan
        else:
            wilcox_pval = np.nan
        results.append({
            "Variable": col_names[idx],
            "RMSE": rmse, "MAE": mae,
            "KS_pvalue": ks_pval, "Wilcoxon_pvalue": wilcox_pval,
        })
    return results

def evaluate_categorical(X_true, X_imp, mask, var_indices):
    results = []
    for idx in var_indices:
        col_mask = mask[:, idx]
        if col_mask.sum() < 2:
            continue
        true_vals_cat = X_true[col_mask, idx].astype(int)
        imp_vals_cat = np.clip(np.round(X_imp[col_mask, idx]), 0, int(np.nanmax(X_true[:, idx]))).astype(int)
        f1 = f1_score(true_vals_cat, imp_vals_cat, average='weighted', zero_division=0)
        err_rate = 1 - accuracy_score(true_vals_cat, imp_vals_cat)
        n_classes_true = len(np.unique(true_vals_cat))
        if n_classes_true >= 2:
            try:
                all_true_all = X_true[:, idx].astype(int)
                all_imp_all = np.clip(np.round(X_imp[:, idx]), 0, int(np.nanmax(X_true[:, idx]))).astype(int)
                observed = pd.crosstab(all_true_all, all_imp_all)
                if observed.shape[0] >= 2 and observed.shape[1] >= 2:
                    chi2, chi2_pval, dof, _ = chi2_contingency(observed)
                else:
                    chi2_pval = np.nan
            except Exception:
                chi2_pval = np.nan
        else:
            chi2_pval = np.nan
        results.append({
            "Variable": col_names[idx],
            "F1_weighted": f1, "Classification_Error": err_rate,
            "Chi2_pvalue": chi2_pval,
        })
    return results

all_results = []
for rate in missing_rates:
    print(f"\n--- Taux de perte : {rate*100:.0f}% ---")
    X_miss = amputed_data[rate]["X_miss"]
    mask = amputed_data[rate]["mask"]
    for method_name, impute_func in imputation_methods.items():
        t0 = time()
        print(f"  Imputation par {method_name}...", end=" ", flush=True)
        try:
            X_imp = impute_func(X_miss)
            elapsed = time() - t0
            print(f"({elapsed:.1f}s)")
            cont_indices = [i for i, c in enumerate(col_names) if var_info[c] == "Continu"]
            if cont_indices:
                cont_results = evaluate_continuous(X_full, X_imp, mask, cont_indices)
                for r in cont_results:
                    all_results.append({
                        "Taux_Perte": f"{int(rate*100)}%", "Taux_Perte_num": rate,
                        "Variable": r["Variable"], "Type_Variable": "Continu",
                        "Methode": method_name,
                        "RMSE": r["RMSE"], "MAE": r["MAE"],
                        "KS_pvalue": r["KS_pvalue"], "Wilcoxon_pvalue": r["Wilcoxon_pvalue"],
                        "F1_weighted": np.nan, "Classification_Error": np.nan, "Chi2_pvalue": np.nan,
                    })
            cat_indices = [i for i, c in enumerate(col_names) if var_info[c] != "Continu"]
            if cat_indices:
                cat_results = evaluate_categorical(X_full, X_imp, mask, cat_indices)
                for r in cat_results:
                    all_results.append({
                        "Taux_Perte": f"{int(rate*100)}%", "Taux_Perte_num": rate,
                        "Variable": r["Variable"], "Type_Variable": var_info[r["Variable"]],
                        "Methode": method_name,
                        "RMSE": np.nan, "MAE": np.nan,
                        "KS_pvalue": np.nan, "Wilcoxon_pvalue": np.nan,
                        "F1_weighted": r["F1_weighted"], "Classification_Error": r["Classification_Error"],
                        "Chi2_pvalue": r["Chi2_pvalue"],
                    })
        except Exception as e:
            elapsed = time() - t0
            print(f"ERREUR ({elapsed:.1f}s): {str(e)[:100]}")

# ============================================================
# 5. SYNTHESE
# ============================================================
df_results = pd.DataFrame(all_results)
print(f"\nNombre total de resultats : {len(df_results)}")

if len(df_results) > 0:
    print("\n" + "=" * 80)
    print("TABLEAU COMPARATIF SYNTHETIQUE")
    print("=" * 80)

    display_cols = ["Taux_Perte", "Variable", "Type_Variable", "Methode",
                    "RMSE", "MAE", "F1_weighted", "Classification_Error",
                    "KS_pvalue", "Wilcoxon_pvalue", "Chi2_pvalue"]

    print("\n--- Variables continues : Resume par methode et taux ---")
    mask_cont = df_results["Type_Variable"] == "Continu"
    if mask_cont.any():
        summary_cont = df_results[mask_cont].groupby(["Taux_Perte", "Methode"])[["RMSE", "MAE"]].mean().round(4)
        print(summary_cont.to_string())

        print("\n--- Variables continues : Proportion de KS significatif (p < 0.05) ---")
        ks_sig = df_results[mask_cont].copy()
        ks_sig["KS_significatif"] = ks_sig["KS_pvalue"] < 0.05
        ks_summary = ks_sig.groupby(["Taux_Perte", "Methode"])["KS_significatif"].mean().round(3)
        print(ks_summary.to_string())

    print("\n--- Variables categorielles : Resume par methode et taux ---")
    mask_cat = df_results["Type_Variable"] != "Continu"
    if mask_cat.any():
        summary_cat = df_results[mask_cat].groupby(["Taux_Perte", "Methode"])[["F1_weighted", "Classification_Error"]].mean().round(4)
        print(summary_cat.to_string())

        print("\n--- Variables categorielles : Proportion de Chi2 significatif (p < 0.05) ---")
        chi2_sig = df_results[mask_cat].copy()
        chi2_sig["Chi2_significatif"] = chi2_sig["Chi2_pvalue"] < 0.05
        chi2_summary = chi2_sig.groupby(["Taux_Perte", "Methode"])["Chi2_significatif"].mean().round(3)
        print(chi2_summary.to_string())

    # Sauvegarde
    df_results.to_csv("resultats_imputation_v2.csv", index=False, sep=";", decimal=",")
    print("\nResultats sauvegardes dans 'resultats_imputation_v2.csv'")

    tableau_complet = df_results[display_cols + ["Taux_Perte_num"]].sort_values(["Taux_Perte_num", "Type_Variable", "Variable", "Methode"])
    tableau_complet = tableau_complet.drop(columns=["Taux_Perte_num"])
    tableau_complet.to_csv("tableau_synthetique_v2.csv", index=False, sep=";", decimal=",")
    print("Tableau synthetique sauvegarde dans 'tableau_synthetique_v2.csv'")

    # Analyse detaillee
    print("\n" + "=" * 80)
    print("ANALYSE DETAILLEE : MEILLEURE METHODE PAR VARIABLE ET PAR TAUX")
    print("=" * 80)

    for rate_str in ["10%", "20%", "40%"]:
        print(f"\n--- Taux {rate_str} (Continu - RMSE) ---")
        cont_sub = df_results[(df_results["Type_Variable"] == "Continu") & (df_results["Taux_Perte"] == rate_str)]
        for var in sorted(cont_sub["Variable"].unique()):
            var_data = cont_sub[cont_sub["Variable"] == var]
            best = var_data.loc[var_data["RMSE"].idxmin()]
            print(f"  {var:30s} -> Meilleure: {best['Methode']:20s} (RMSE={best['RMSE']:.4f})")

    for rate_str in ["10%", "20%", "40%"]:
        print(f"\n--- Taux {rate_str} (Categoriel - F1) ---")
        cat_sub = df_results[(df_results["Type_Variable"] != "Continu") & (df_results["Taux_Perte"] == rate_str)]
        for var in sorted(cat_sub["Variable"].unique()):
            var_data = cat_sub[cat_sub["Variable"] == var]
            best = var_data.loc[var_data["F1_weighted"].idxmax()]
            print(f"  {var:30s} -> Meilleure: {best['Methode']:20s} (F1={best['F1_weighted']:.4f})")

    # Analyse de la variance
    print("\n" + "=" * 80)
    print("ANALYSE DE LA VARIANCE POST-IMPUTATION")
    print("=" * 80)

    for rate in missing_rates:
        print(f"\n--- Taux {int(rate*100)}% ---")
        X_miss = amputed_data[rate]["X_miss"]
        cont_idx = [i for i, c in enumerate(col_names) if var_info[c] == "Continu"]
        var_original = np.nanvar(X_full[:, cont_idx], axis=0)
        for method_name, impute_func in imputation_methods.items():
            try:
                X_imp = impute_func(X_miss)
                var_imputed = np.nanvar(X_imp[:, cont_idx], axis=0)
                # Eviter division par zero
                valid = var_original > 0
                if valid.any():
                    var_ratio = np.mean(var_imputed[valid] / var_original[valid])
                    print(f"  {method_name:25s} -> Ratio variance imputee/originale = {var_ratio:.4f}")
            except Exception as e:
                print(f"  {method_name:25s} -> ERREUR: {str(e)[:80]}")

else:
    print("AUCUN RESULTAT - Toutes les imputations ont echoue.")

print("\n" + "=" * 80)
print("FIN DE L'ANALYSE")
print("=" * 80)
