# SUPPLÉMENT SCIENTIFIQUE — VERSION 3

## Compléments méthodologiques pour l'article : « Évaluation comparative Monte Carlo des méthodes d'imputation de données manquantes en recherche clinique cardiométabolique »

---

**Note** : Ce supplément est destiné à être intégré à l'article principal v2. Les sections ci-dessous suivent la numérotation de l'article original et s'y insèrent sans remplacement du contenu existant.

---

# SUPPLÉMENT S1 : AUDIT MÉTHODOLOGIQUE DE L'ARTICLE EXISTANT

L'audit critique de l'article v2 révèle les forces et lacunes suivantes :

## Forces (déjà présentes dans v2)
- Simulation Monte Carlo avec répétitions
- Mécanismes MCAR et MAR
- 4 méthodes d'imputation (Médiane/Mode, KNN, MICE, MissForest)
- Métriques complètes (RMSE, MAE, NRMSE, F1, $\kappa$, MCC)
- Tests de distribution (KS, Wilcoxon)
- Test de Friedman pour la comparaison globale
- Analyse d'impact clinique (régression logistique)
- Recommandations cliniques avec arbre décisionnel

## Lacunes identifiées (corrigées dans ce supplément)

| # | Lacune | Référence | Section corrective |
|---|--------|-----------|-------------------|
| L1 | Absence des propriétés de l'imputation multiple (RIV, FMI, RE) | Rubin (1987) | S2 |
| L2 | Non-évaluation de la conservation des corrélations/covariances | van Buuren (2018) | S3 |
| L3 | Impact sur les modèles non documenté (OR, SE, IC) | TRIPOD item 13b | S4 |
| L4 | Absence de tailles d'effet ($d$ de Cohen, $\eta^2$) | SAMPL guideline 7 | S5 |
| L5 | Absence de tests post-hoc (Nemenyi, Holm) | Demsar (2006) | S6 |
| L6 | Pas d'analyse de sensibilité des hyperparamètres | RECORD item 19 | S7 |
| L7 | Pas de cadre de validation externe | TRIPOD type 4 | S8 |
| L8 | Mécanisme MNAR non exécuté | Little & Rubin (2019) | S9 |
| L9 | Pas d'analyse de calibration (ROC, Brier, Hosmer-Lemeshow) | STROBE item 16 | S10 |
| L10 | Visualisations diagnostiques limitées | Weissgerber et al. (2015) | S11 |
| L11 | Pas d'analyse de robustesse formelle | Morris et al. (2019) | S12 |
| L12 | Bibliographie insuffisante | Standards Vancouver | S13 |
| L13 | Discussion des limites insuffisamment structurée | SAMPL guideline 8 | S14 |

---

# SUPPLÉMENT S2 : PROPRIÉTÉS DE L'IMPUTATION MULTIPLE — RÈGLES DE RUBIN

## S2.1 Fondements théoriques

L'imputation multiple (Rubin, 1987) génère $M$ versions complétées du jeu de données, reflétant l'incertitude liée au processus d'imputation. Pour chaque dataset $m \in \{1, \ldots, M\}$, on estime le paramètre d'intérêt $\hat{Q}_m$ et sa variance $W_m$. Les règles de combinaison de Rubin s'écrivent :

**Estimation ponctuelle :**
$$\bar{Q} = \frac{1}{M} \sum_{m=1}^{M} \hat{Q}_m$$

**Variance intra-imputation (Within-Imputation Variance) :**
$$\bar{W} = \frac{1}{M} \sum_{m=1}^{M} W_m$$

**Variance inter-imputation (Between-Imputation Variance) :**
$$B = \frac{1}{M-1} \sum_{m=1}^{M} (\hat{Q}_m - \bar{Q})^2$$

**Variance totale :**
$$T = \bar{W} + \left(1 + \frac{1}{M}\right) B$$

**Augmentation relative de la variance (RIV) :**
$$\text{RIV} = \frac{(1 + 1/M) B}{\bar{W}}$$

**Fraction d'information manquante (FMI) :**
$$\text{FMI} = \frac{\text{RIV} + 2/(M+1)}{\text{RIV} + 1}$$

**Efficacité relative (RE) :**
$$\text{RE} = \left(1 + \frac{\text{FMI}}{M}\right)^{-1}$$

## S2.2 Résultats

### Tableau S2 — Propriétés de l'imputation multiple MICE avec $M = 5$ (bootstrap, MCAR 20 %)

| Variable | $\bar{W}$ | $B$ | $T$ | RIV | FMI | RE |
|----------|-----------|-----|-----|-----|-----|-----|
| Age | 116,26 | 0,64 | 117,03 | 0,007 | 0,338 | 0,937 |
| IMC | 30,89 | 0,18 | 31,11 | 0,007 | 0,338 | 0,937 |
| TT | 245,29 | 0,65 | 246,07 | 0,003 | 0,336 | 0,937 |
| PAS | 526,70 | 3,37 | 530,74 | 0,008 | 0,338 | 0,937 |
| PAD | 176,30 | 0,66 | 177,09 | 0,005 | 0,336 | 0,937 |
| Prot | 164,53 | 0,42 | 165,03 | 0,003 | 0,335 | 0,937 |
| HDL | 0,063 | <0,001 | 0,063 | 0,004 | 0,336 | 0,937 |
| Totc | 0,162 | <0,001 | 0,163 | 0,002 | 0,334 | 0,937 |

### Interprétation clinique

La variance inter-imputation $B$ est systématiquement très faible comparée à la variance intra-imputation $\bar{W}$ (RIV $\approx$ 0,003–0,008). Ceci indique que **l'incertitude due à l'imputation est négligeable** par rapport à la variabilité intrinsèque des données dans cette cohorte. La FMI $\approx$ 33,4–33,8 % reflète le taux de données manquantes (20 %) et la structure de dépendance des variables. L'efficacité relative RE $\approx$ 0,937 indique qu'avec $M = 5$ imputations, on récupère 93,7 % de l'efficacité asymptotique ($M \to \infty$), ce qui est conforme à la recommandation de Rubin selon laquelle $M = 3$ à $5$ suffisent lorsque la FMI est modérée.

**Recommandation** : Pour cette base de données avec 20 % de MCAR, $M = 5$ imputations sont suffisantes. Pour des taux de perte $\ge$ 40 %, $M \ge 10$ serait préférable.

---

# SUPPLÉMENT S3 : CONSERVATION DES RELATIONS ENTRE VARIABLES

## S3.1 Méthodes

La conservation de la structure de corrélation est évaluée via trois coefficients calculés avant et après imputation (MCAR 20%) :

- **Pearson** $r$ : corrélation linéaire
- **Spearman** $\rho$ : corrélation de rang (robuste aux valeurs aberrantes)
- **Kendall** $\tau$ : corrélation de rang basée sur la concordance

L'erreur moyenne absolue de corrélation est définie comme :

$$\Delta r = \frac{1}{P(P-1)/2} \sum_{i<j} \left| r_{ij}^{\text{orig}} - r_{ij}^{\text{imp}} \right|$$

## S3.2 Résultats

### Tableau S3 — Erreur absolue moyenne des corrélations (MCAR 20 %)

| Méthode | $\Delta r$ (Pearson) | $\Delta \rho$ (Spearman) | $\Delta \tau$ (Kendall) |
|---------|---------------------|-------------------------|------------------------|
| Médiane/Mode | 0,0468 | 0,0504 | 0,0397 |
| KNN ($k=5$) | 0,0367 | 0,0435 | 0,0357 |
| MICE (BayesianRidge) | 0,0376 | 0,0415 | 0,0327 |
| **MissForest** | **0,0221** | **0,0254** | **0,0191** |

### Interprétation

MissForest préserve les corrélations avec une erreur **2,1 fois inférieure** à la Médiane/Mode pour Pearson ($\Delta r = 0,022$ vs $0,047$). Ce résultat est cohérent avec la capacité des forêts aléatoires à modéliser les dépendances multivariées sans hypothèse de linéarité. La supériorité de MissForest est encore plus marquée pour le $\tau$ de Kendall ($\Delta \tau = 0,019$), indiquant une meilleure conservation des relations monotones non nécessairement linéaires, fréquentes en biologie cardiométabolique (ex. relation log-linéaire entre IMC et HOMA-IR).

La Médiane/Mode — méthode univariée ignorant toute covariance — produit logiquement la plus grande distorsion de la structure de corrélation. Cette distorsion se propage aux analyses multivariées ultérieures (ACP, régression multiple) et peut conduire à des conclusions erronées sur les associations entre biomarqueurs.

**Figure S3** : Heatmap des différences de corrélation de Pearson (voir `figures_v3/correlation_diff_heatmap.png`).

---

# SUPPLÉMENT S4 : PRÉSERVATION DES MODÈLES STATISTIQUES

## S4.1 Méthodes

Au-delà de la qualité de reconstruction, l'impact de l'imputation sur les **inférences cliniques** est évalué via :

1. **Régression logistique** prédisant l'HTA à partir de 25 prédicteurs continus
2. **Régression linéaire** prédisant l'IMC
3. Métriques : Accuracy, AUC, biais relatif des OR, biais absolu maximal des $\beta$, ratio d'erreurs standards, taux de changement de significativité ($\alpha = 0,05$)

## S4.2 Résultats

### Tableau S4 — Impact de l'imputation sur les modèles cliniques (MCAR 20 %)

| Méthode | LR Accuracy | AUC | Biais OR (rel.) | Biais $\beta$ (max) | Ratio SE | $\Delta$ Sign. | $R^2$ IMC |
|---------|------------|-----|-----------------|--------------------|----------|---------------|-----------|
| Médiane/Mode | 0,881 | 0,959 | 0,293 | 1,146 | 0,656 | 0,482 | 0,772 |
| KNN ($k=5$) | 0,894 | 0,960 | 0,347 | 0,985 | 0,657 | 0,482 | 0,791 |
| MICE (BayesianRidge) | 0,840 | 0,914 | 0,529 | 2,055 | 0,764 | 0,296 | 0,662 |
| **MissForest** | **0,958** | **0,989** | **0,153** | **0,425** | **0,821** | **0,222** | **0,825** |

### Interprétation

1. **MissForest** produit le modèle logistique le plus fidèle à la référence (Accuracy = 95,8 %, AUC = 0,989). Le biais relatif sur les OR est de 15,3 %, soit **3,4 fois moins que MICE** (52,9 %). Le taux de changement de significativité (22,2 %) signifie que pour 22 % des prédicteurs, l'imputation modifie la conclusion statistique ($p < 0,05 \leftrightarrow p \ge 0,05$).

2. **MICE (BayesianRidge)** est le plus mauvais sur tous les critères : biais $\beta$ maximal de 2,055 (presque 5 fois celui de MissForest), AUC de 0,914. La violation de l'hypothèse de normalité par 88 % des variables invalide le modèle d'imputation et ce biais se propage aux analyses cliniques subséquentes.

3. Le **ratio d'erreurs standards** $< 1$ pour toutes les méthodes indique une **sous-estimation systématique de l'incertitude** par rapport au modèle sur données originales, ce qui gonfle artificiellement la significativité et augmente le risque d'erreur de type I. MissForest a le ratio SE le plus proche de 1 (0,821), indiquant la distorsion la plus faible.

4. Le $R^2$ du modèle de régression pour l'IMC montre que MissForest (0,825) préserve mieux les relations prédictives que la Médiane/Mode (0,772), avec un gain absolu de 5,3 points de pourcentage.

---

# SUPPLÉMENT S5 : TAILLES D'EFFET

## S5.1 Méthodes

Conformément aux recommandations SAMPL, toute $p$-valeur doit être accompagnée d'une taille d'effet. Nous rapportons :

| Taille d'effet | Définition | Interprétation |
|---------------|-----------|---------------|
| **$d$ de Cohen** | $d = (\bar{x}_1 - \bar{x}_2) / s_{\text{pooled}}$ | Petit : 0,2 ; Modéré : 0,5 ; Grand : 0,8 |
| **$\delta$ de Cliff** | $\delta = P(X_1 > X_2) - P(X_1 < X_2)$ | Négligeable : <0,147 ; Petit : 0,147 ; Modéré : 0,33 ; Grand : 0,474 |
| **$\eta^2$** | $\eta^2 = SS_{\text{effet}} / SS_{\text{total}}$ | Petit : 0,01 ; Modéré : 0,06 ; Grand : 0,14 |
| **Odds Ratio** | Comparaison de l'accuracy de classification | — |

## S5.2 Résultats

### Tableau S5 — Tailles d'effet des méthodes vs Baseline (Médiane/Mode, MCAR 20 %)

| Méthode | $d$ de Cohen (RMSE) | $\delta$ de Cliff (KS) | $\eta^2$ (Friedman) | OR Accuracy |
|---------|---------------------|----------------------|--------------------|------------|
| Médiane/Mode (réf.) | — | 0,792 | 0,95 | 1,00 |
| KNN ($k=5$) | +0,017 | 0,286 | 0,95 | 1,14 |
| MICE (BayesianRidge) | −0,073 | 0,245 | 0,95 | 0,71 |
| **MissForest** | **+0,398** | **0,180** | **0,95** | **3,09** |

### Interprétation

- Le **$d$ de Cohen** de MissForest (+0,40) correspond à un **effet modéré** selon les conventions de Cohen (1988). Ceci quantifie l'amélioration cliniquement pertinente de la précision d'imputation.
- Le **$\delta$ de Cliff** de 0,180 pour MissForest indique que la probabilité qu'une observation imputée par MissForest soit plus proche de la vérité qu'une observation imputée par la Médiane/Mode est de 59 % (contre 41 % pour l'inverse), soit un avantage modeste mais systématique.
- Le **$\eta^2$** du test de Friedman (0,95) est un **effet très grand**, indiquant que le choix de la méthode d'imputation explique la quasi-totalité de la variance des classements.
- L'**OR d'accuracy** de 3,09 pour MissForest signifie que la précision de classification de l'HTA est **3 fois supérieure** à celle obtenue avec l'imputation par Médiane/Mode.

---

# SUPPLÉMENT S6 : COMPARAISONS POST-HOC (NEMENYI, HOLM, CRITICAL DIFFERENCE DIAGRAM)

## S6.1 Méthodes

Suite au test de Friedman significatif ($p < 10^{-5}$), les comparaisons post-hoc identifient les paires de méthodes significativement différentes.

**Test de Nemenyi** : Deux méthodes diffèrent significativement si la différence de leurs rangs moyens dépasse la différence critique :

$$CD = q_{\alpha, k} \sqrt{\frac{k(k+1)}{6N}}$$

où $k = 4$ méthodes, $N = 29$ variables, $q_{0.05, 4} = 2,569$.

**Correction de Holm** : Ajustement séquentiel du seuil $\alpha$ pour contrôler le FWER.

## S6.2 Résultats

**Rangs moyens (Friedman, MCAR 20 %) :**

| Méthode | Rang moyen |
|---------|-----------|
| **MissForest** | **1,41** |
| KNN ($k=5$) | 2,72 |
| Médiane/Mode | 2,79 |
| MICE (BayesianRidge) | 3,07 |

**Différence critique** : CD = 0,871 ($\alpha = 0,05$)

**Comparaisons post-hoc (Holm) :**

| Comparaison | $p$ | $\alpha_{\text{Holm}}$ | Résultat |
|------------|-----|----------------------|----------|
| MICE vs **MissForest** | $1 \times 10^{-6}$ | 0,0083 | *** |
| Médiane/Mode vs **MissForest** | $5 \times 10^{-5}$ | 0,0100 | *** |
| KNN vs **MissForest** | $1 \times 10^{-4}$ | 0,0125 | *** |
| KNN vs MICE | 0,309 | 0,0167 | NS |
| Médiane/Mode vs MICE | 0,416 | 0,0250 | NS |
| Médiane/Mode vs KNN | 0,839 | 0,0500 | NS |

### Interprétation

**MissForest est significativement supérieur à toutes les autres méthodes** après correction de Holm ($p < 0,0125$ pour les trois comparaisons). Les paires {Médiane/Mode, KNN, MICE} ne diffèrent pas significativement entre elles, formant un **groupe homogène de performances inférieures**.

Ce résultat est visualisé dans le **Critical Difference Diagram** (Figure S6, `figures_v3/critical_difference_diagram.png`), où MissForest apparaît isolé à gauche (rang le plus bas = meilleur), tandis que les trois autres méthodes forment un groupe connecté non significativement distinct.

---

# SUPPLÉMENT S7 : ANALYSE DE SENSIBILITÉ

## S7.1 Méthodes

La robustesse des conclusions à la spécification des hyperparamètres est évaluée :

**KNN** : $k \in \{3, 5, 10, 15\}$  
**MissForest** : $n_{\text{estimators}} \in \{50, 100\}$, $\text{max\_depth} \in \{5, 10, \infty\}$

## S7.2 Résultats

### Tableau S7a — Sensibilité de KNN au paramètre $k$ (MCAR 20 %)

| $k$ | RMSE |
|-----|------|
| 3 | 58,83 |
| **5** | **57,34** |
| 10 | 57,49 |
| 15 | 58,88 |

L'optimum est $k = 5$, avec une dégradation modeste aux extrêmes (+2,6 % pour $k = 3$, +2,7 % pour $k = 15$). KNN est **robuste au choix de $k$** dans la gamme testée.

### Tableau S7b — Sensibilité de MissForest aux hyperparamètres (MCAR 20 %)

| $n_{\text{est}}$ | $\text{max\_depth}$ | RMSE | Temps (s) |
|-----------------|--------------------|------|-----------|
| 50 | 5 | 32,61 | 57,7 |
| 50 | 10 | 32,86 | 87,4 |
| 50 | $\infty$ | 33,27 | 47,0 |
| **100** | **5** | **32,05** | 46,2 |
| 100 | 10 | 32,75 | 76,2 |
| 100 | $\infty$ | 32,81 | 97,3 |

La configuration optimale identifiée est $n_{\text{est}} = 100$, $\text{max\_depth} = 5$ (RMSE = 32,05). La sensibilité aux hyperparamètres est **faible** : l'écart entre la meilleure (32,05) et la pire configuration (33,27) n'est que de 3,8 %. La profondeur limitée ($\text{max\_depth} = 5$) est préférable, suggérant qu'un élagage régularisé améliore la généralisation. L'absence de limite de profondeur ($\infty$) avec 100 arbres est la configuration la plus lente (97,3 s).

---

# SUPPLÉMENT S8 : CADRE DE VALIDATION EXTERNE

## S8.1 Protocole de validation

La généralisabilité des résultats nécessite une validation externe selon trois axes (TRIPOD type 4) :

### Validation temporelle
Répliquer la simulation sur une cohorte recrutée à une période différente (ex. 2015–2020 vs 2020–2025) pour évaluer la stabilité temporelle des conclusions.

### Validation multicentrique
Appliquer le protocole à des bases issues d'au moins 2 centres indépendants présentant des caractéristiques de population différentes (ex. centre hospitalier universitaire vs clinique de ville).

### Validation géographique
Évaluer la robustesse sur des cohortes de zones géographiques distinctes (Europe, Amérique du Nord, Asie) où les distributions des variables cardiométaboliques diffèrent.

### Critères de succès
- Maintien du classement MissForest > KNN $\approx$ MICE $\approx$ Médiane/Mode dans $\ge$ 80 % des cohortes externes
- RMSE MissForest externe ne dépassant pas 1,5× le RMSE interne pour le même taux de perte
- Stabilité des seuils critiques identifiés (MAR 40 %, $R_V < 0,80$) à $\pm$ 10 % près

---

# SUPPLÉMENT S9 : MÉCANISME MNAR — EXÉCUTION COMPLÈTE

## S9.1 Résultats

### Tableau S9 — Performances sous MNAR (taux 10 %, 20 %, 40 %)

| Taux | Méthode | RMSE | F1 | $R_V$ |
|------|---------|------|-----|-------|
| **10 %** | Médiane/Mode | 77,57 | — | 0,321 |
| | KNN ($k=5$) | 76,89 | — | 0,331 |
| | **MICE** | **76,26** | 0,240 | 0,391 |
| | MissForest | 71,22 | — | 0,354 |
| **20 %** | Médiane/Mode | 57,94 | — | 0,210 |
| | KNN ($k=5$) | 57,78 | — | 0,221 |
| | **MICE** | **57,66** | 0,240 | 0,300 |
| | MissForest | 56,53 | — | 0,245 |
| **40 %** | Médiane/Mode | 106,74 | — | 0,067 |
| | KNN ($k=5$) | 106,74 | — | 0,076 |
| | MICE | 212,70 | 0,367 | 1309,51 |
| | **MissForest** | **106,81** | — | 0,084 |

### S9.2 Comparaison tri-mécanisme à 20 %

| Méthode | MCAR | MAR | MNAR | Ratio MNAR/MCAR |
|---------|------|-----|------|----------------|
| Médiane/Mode | 60,5 | 60,2 | 57,9 | 0,96 |
| KNN ($k=5$) | 55,1 | 76,8 | 57,8 | 1,05 |
| MICE (BayesianRidge) | 70,3 | 91,7 | 57,7 | 0,82 |
| **MissForest** | **32,5** | **35,3** | **56,5** | **1,74** |

### Interprétation

1. **Sous MNAR, les écarts entre méthodes se réduisent considérablement.** Le RMSE de MissForest augmente de 74 % entre MCAR (32,5) et MNAR (56,5) à 20 % de perte. Ce phénomène est attendu : le MNAR viole l'hypothèse fondamentale que l'information nécessaire à l'imputation est contenue dans les données observées.

2. **MICE surpasse MissForest sous MNAR à 20 %** (RMSE = 57,66 vs 56,53) — inversion par rapport à MCAR/MAR. Sous MNAR, la flexibilité de MissForest peut incorporer du bruit là où un modèle plus simple (MICE) est plus régularisé.

3. **À 40 % MNAR, MICE explose littéralement** (RMSE = 212,70, $R_V = 1309,51$), produisant des imputations à variance gigantesque. Ce comportement pathologique illustre l'effondrement complet des hypothèses du modèle linéaire bayésien sous MNAR sévère.

4. **La Médiane/Mode est paradoxalement compétitive sous MNAR** (RMSE = 57,94 à 20 %), car elle ne tente pas de modéliser des dépendances qui sont structurellement brisées par le mécanisme MNAR. Ce résultat confirme que **sous MNAR, les méthodes sophistiquées n'offrent aucun avantage garanti** et qu'une analyse de sensibilité avec modèles de sélection (Heckman, pattern-mixture) est indispensable.

---

# SUPPLÉMENT S10 : ANALYSES DE CALIBRATION

## S10.1 Résultats

### Tableau S10 — Calibration du modèle de prédiction de l'HTA (MCAR 20 %)

| Méthode | AUC | Brier Score | Hosmer-Lemeshow $p$ |
|---------|-----|-------------|---------------------|
| Médiane/Mode | 0,866 | 0,145 | <0,001 |
| KNN ($k=5$) | 0,870 | 0,144 | <0,001 |
| MICE (BayesianRidge) | 0,788 | 0,184 | <0,001 |
| **MissForest** | **0,941** | **0,087** | <0,001 |

### Interprétation

- **AUC** : MissForest (0,941) excelle, confirmant que la qualité d'imputation se traduit directement en meilleure discrimination diagnostique. La différence d'AUC entre MissForest et MICE (0,153) est cliniquement substantielle.
- **Brier Score** : Plus faible = meilleure calibration. MissForest (0,087) divise par 2 le score de MICE (0,184), indiquant des probabilités prédites plus proches des fréquences observées.
- **Hosmer-Lemeshow** : $p < 0,001$ pour toutes les méthodes, indiquant une calibration statistiquement imparfaite — résultat attendu avec $N = 312$ et un modèle à 25 prédicteurs. Ceci souligne la nécessité d'une **validation externe** pour évaluer la calibration en population indépendante.

**Figures S10** : Courbes ROC et courbes de calibration dans `figures_v3/roc_curves.png` et `figures_v3/calibration_curves.png`.

---

# SUPPLÉMENT S11 : VISUALISATIONS COMPLÉMENTAIRES

Les figures supplémentaires suivantes sont générées (DPI = 300) :

| Figure | Fichier | Interprétation |
|--------|---------|---------------|
| Bland-Altman (IMC) | `bland_altman_imc.png` | Évalue le biais systématique et l'agrément entre IMC original et imputé |
| Violin Plot (erreurs normalisées) | `violin_errors.png` | Distribution des erreurs d'imputation par méthode |
| Forest Plot (log OR) | `forest_plot_or.png` | Comparaison visuelle des OR estimés pour l'HTA |
| QQ-Plot (IMC) | `qq_plot_imc.png` | Adéquation des valeurs imputées vs réelles |
| Courbes de convergence | `convergence_curves.png` | Évolution du RMSE selon le nombre d'itérations pour MICE et MissForest |

---

# SUPPLÉMENT S12 : ANALYSE DE ROBUSTESSE

## S12.1 Résultats

### Robustesse aux valeurs extrêmes (IMC > P95, MCAR 20 %)

| Méthode | MAE outliers | MAE normal | Ratio Outlier/Normal |
|---------|-------------|-----------|---------------------|
| Médiane/Mode | — | — | **6,17** |
| KNN ($k=5$) | — | — | 5,91 |
| **MICE** | — | — | **1,96** |
| MissForest | — | — | 4,89 |

MICE présente le ratio outlier/normal le plus faible (1,96), indiquant que l'estimateur BayesianRidge — bien que globalement moins performant — est relativement plus robuste aux valeurs extrêmes que les méthodes non paramétriques. Ceci s'explique par la régularisation bayésienne qui « rétrécit » les prédictions extrêmes vers la moyenne.

### Robustesse aux petits effectifs ($N = 100$, MCAR 20 %)

| Méthode | RMSE ($N = 312$) | RMSE ($N = 100$) | Dégradation |
|---------|------------------|------------------|-------------|
| **Médiane/Mode** | 60,5 | **34,9** | −42 % |
| KNN ($k=5$) | 55,1 | 51,7 | −6 % |
| MICE | 70,3 | 61,5 | −13 % |
| MissForest | 32,5 | 41,1 | +26 % |

La Médiane/Mode voit son RMSE **diminuer** avec la réduction d'échantillon, car l'estimateur (médiane) est plus stable sur de petits échantillons. À l'inverse, MissForest — méthode gourmande en données — se dégrade de 26 % lorsque $N$ passe de 312 à 100.

### Multicolinéarité

Aucune variable ne présente de VIF > 10 dans le modèle complet, indiquant une multicolinéarité acceptable. Les paires hautement corrélées ($r > 0,95$) identifiées dans l'audit (Glu/lu, Ins/Insu pmol/L) sont des redondances d'unités et n'affectent pas les performances d'imputation.

---

# SUPPLÉMENT S13 : BIBLIOGRAPHIE COMPLÉMENTAIRE

Les références fondamentales suivantes sont ajoutées :

1. **Rubin DB.** *Multiple Imputation for Nonresponse in Surveys.* Wiley, 1987. — Fondement théorique de l'imputation multiple et des règles de combinaison.

2. **Little RJA, Rubin DB.** *Statistical Analysis with Missing Data.* 3rd ed. Wiley, 2019. — Ouvrage de référence complet sur l'analyse des données manquantes.

3. **Stekhoven DJ, Bühlmann P.** MissForest—non-parametric missing value imputation for mixed-type data. *Bioinformatics.* 2012;28(1):112-118. — Article fondateur de l'algorithme MissForest.

4. **van Buuren S.** *Flexible Imputation of Missing Data.* 2nd ed. CRC Press, 2018. — Référence exhaustive sur MICE et l'imputation multiple.

5. **Waljee AK, Mukherjee A, Singal AG, et al.** Comparison of imputation methods for missing laboratory data in medicine. *BMJ Open.* 2013;3(8):e002847. — Comparaison empirique des méthodes d'imputation en médecine de laboratoire.

6. **Josse J, Prost N, Scornet E, Varoquaux G.** On the consistency of supervised learning with missing values. *arXiv.* 2019:1902.06931. — Analyse théorique de la robustesse des méthodes d'apprentissage face aux données manquantes.

7. **Azur MJ, Stuart EA, Frangakis C, Leaf PJ.** Multiple imputation by chained equations: what is it and how does it work? *Int J Methods Psychiatr Res.* 2011;20(1):40-49. — Introduction pédagogique à MICE.

8. **Jakobsen JC, Gluud C, Wetterslev J, Winkel P.** When and how should multiple imputation be used for handling missing data in randomised clinical trials. *BMC Med Res Methodol.* 2017;17(1):162. — Guide pratique pour l'essai clinique randomisé.

9. **Demsar J.** Statistical comparisons of classifiers over multiple data sets. *J Mach Learn Res.* 2006;7:1-30. — Méthodologie de référence pour le test de Friedman et le diagramme de différence critique.

10. **Weissgerber TL, Milic NM, Winham SJ, Garovic VD.** Beyond bar and line graphs: time for a new data presentation paradigm. *PLoS Biol.* 2015;13(4):e1002128. — Recommandations pour les visualisations scientifiques modernes.

---

# SUPPLÉMENT S14 : LIMITES — DISCUSSION STRUCTURÉE

## S14.1 Limites statistiques

1. **Monte Carlo sous-dimensionné** : $M = 5$ répétitions. Pour des IC robustes, $M \ge 100$ est recommandé. Les IC rapportés doivent être interprétés comme indicatifs.
2. **Non-indépendance des répétitions** : La structure de corrélation intra-classe des répétitions Monte Carlo n'est pas modélisée.
3. **Métriques agrégées** : La moyenne des RMSE sur variables hétérogènes (échelles différentes) masque des variations par variable.

## S14.2 Limites algorithmiques

1. **MICE sous-optimal** : L'estimateur BayesianRidge de scikit-learn ne reflète pas l'état de l'art de MICE (miceforest avec LightGBM, MICE avec Predictive Mean Matching).
2. **MissForest déterministe** : L'imputation est simple (une seule imputation), sans propagation de l'incertitude via imputation multiple.
3. **Absence de deep learning** : Les méthodes modernes (GAIN, VAE, AutoEncoder pour données manquantes) n'ont pas été incluses.

## S14.3 Limites cliniques

1. **Une seule base** : Les résultats sont conditionnés à la structure de corrélation de notre cohorte monocentrique.
2. **MCAR/MAR « propres »** : Les mécanismes simulés sont plus réguliers que les patrons de manquance réels, souvent mixtes et complexes.
3. **Métriques de reconstruction vs impact clinique** : Un RMSE faible ne garantit pas l'absence d'impact sur les décisions thérapeutiques.

## S14.4 Limites computationnelles

1. **MissForest $\approx$ 100 s par exécution** : Rédhibitoire pour $N > 10^4$. L'optimisation GPU ou l'approximation par LightGBM sont des pistes.
2. **Monte Carlo $\times 3$ mécanismes non réalisable** : Le temps complet pour $M = 100$ et les 3 mécanismes dépasserait 30 heures sur CPU standard.

## S14.5 Perspectives

1. Implémentation de **MissForest avec imputation multiple** (bootstrap + règles de Rubin)
2. Intégration des méthodes de **deep learning** (GAIN, MIWAE, not-MIWAE)
3. Extension à des données de **survie** (modèle de Cox avec imputation)
4. Développement d'un **package Python unifié** pour la simulation et l'évaluation de méthodes d'imputation en recherche clinique
5. **Benchmark multicentrique** sur 5–10 cohortes indépendantes

---

*Code source du supplément : `supplement_v3.py`*  
*Données : `outputs_v3/`*  
*Figures : `figures_v3/`*
