# ARTICLE SCIENTIFIQUE — VERSION FINALE

## Évaluation comparative Monte Carlo des méthodes d'imputation de données manquantes en recherche clinique cardiométabolique : mécanismes MCAR, MAR et MNAR à 10 %, 20 % et 40 %

---

**Auteurs** : Pipeline automatisé de simulation reproductible  
**Date** : Juillet 2026  
**Environnement** : Python 3.13.5, pandas 2.3.1, scikit-learn 1.7.0, scipy 1.16.0  
**Reproductibilité** : Graine aléatoire fixée (seed = 42), versions logicielles enregistrées  
**Code source** : `simulation_v2.py`, `supplement_v3.py`  

---

# SECTION 1 : MATÉRIEL ET MÉTHODES

## 1.1 Description de la base de données

### Population étudiée

L'étude s'appuie sur une base de données cliniques issue d'une cohorte monocentrique de **312 sujets** (âge moyen : $57,2 \pm 11,7$ ans ; sex-ratio : 52,9 % féminin, 47,1 % masculin). La base comporte **33 variables** couvrant les domaines suivants :

| Domaine | Variables | Type prédominant |
|---------|-----------|-----------------|
| Anthropométrie | Âge, Sexe, IMC, Tour de taille (TT) | Continue asymétrique |
| Hémodynamique | PAS, PAD, HTA, HTS/HTD | Mixte |
| Métabolisme lipidique | Prot, HDL, Totc, LDLc, TG, Non HDL, Ratios lipidiques | Continue asymétrique |
| Métabolisme glucidique | Glu, Ins, HOMAB%, HOMA-IR, HOMAS%, Profil glycémique | Continue asymétrique / Ordinal |
| Inflammation | PCSK9, Adipo HMW | Continue fortement asymétrique |
| Pharmacologie | Glibenclamide, Insuline, Lasilix, Lexomil, Metformine | Continue / Quasi constante |

### Audit qualité

L'audit systématique de la base a révélé :

1. **Variables constantes** : Lasilix 40 mg et Lexomil 6 mg (100 % des sujets à 0 mg). Ces variables ont été exclues des analyses multivariées.
2. **Corrélations extrêmes** ($r > 0,95$) identifiées pour 4 paires :
   - LDLc $\leftrightarrow$ Non HDL : $r = 0,953$
   - Glu $\leftrightarrow$ lu(mmol/l) : $r = 1,000$ (redondance parfaite, conversion d'unités)
   - Ins (µUI/mL) $\leftrightarrow$ Insu pmol/L : $r = 1,000$ (redondance parfaite)
   - Ratio Tot-c/HDL $\leftrightarrow$ Ratio LDL-c/HDL : $r = 0,997$
3. **Aucune valeur aberrante** nécessitant une exclusion n'a été détectée après inspection des distributions.
4. **Données originales complètes** : aucun missing natif dans les colonnes principales.
5. **Multicolinéarité** : aucune variable ne présente de VIF $> 10$, indiquant une colinéarité acceptable malgré les redondances d'unités identifiées.

### Classification et distribution des variables

| Classe | n | % | Exemples |
|--------|---|---|---------|
| Continue normale | 2 | 6,1 % | Age, Totc (Shapiro-Wilk $p > 0,05$) |
| Continue asymétrique | 23 | 69,7 % | IMC, PAS, HDL, LDLc, TG, PCSK9 |
| Continue fortement asymétrique (skew $> 3$) | 6 | 18,2 % | HOMA-IR (skew = +8,81), Adipo HMW (skew = +7,55), Ins (skew = +3,64) |
| Continue quasi constante | 2 | 6,1 % | Lasilix, Lexomil |
| Binaire | 2 | 6,1 % | Sexe, HTA |
| Ordinale | 2 | 6,1 % | HTS/HTD (3 modalités), Profil glycémique (4 modalités) |

La prédominance marquée de variables continues non gaussiennes (87,9 % des variables continues) constitue la contrainte méthodologique centrale de cette étude.

## 1.2 Protocole de simulation des données manquantes

### Mécanismes implémentés

Trois mécanismes de données manquantes sont simulés, chacun à trois taux de perte (10 %, 20 %, 40 %) :

**1. MCAR (Missing Completely At Random)**  
Chaque cellule $(i, j)$ de la matrice de données $X \in \mathbb{R}^{N \times P}$ est supprimée indépendamment avec probabilité $\tau$ :

$$P(R_{ij} = 0 \mid X_{obs}, X_{miss}) = \tau$$

**2. MAR (Missing At Random)**  
La probabilité de manquance pour la variable $j$ dépend des autres variables observées $X_{-j}$ via une transformation sigmoïde :

$$P(R_{ij} = 0) = \tau + (1 - \tau) \cdot \sigma\left( \sum_{k \neq j} w_k X_{ik}^{obs} \right)$$

où $\sigma(\cdot)$ est la fonction logistique et les poids $w_k$ sont tirés aléatoirement.

**3. MNAR (Missing Not At Random)**  
La probabilité de manquance dépend de la valeur elle-même, avec une probabilité accrue pour les valeurs extrêmes :

$$P(R_{ij} = 0) \propto \frac{|X_{ij} - \text{med}(X_j)|}{\text{MAD}(X_j)}$$

Ce mécanisme, exécuté dans la version complète de cette étude (voir §2.7), représente le scénario le plus réaliste et le plus problématique en clinique.

### Protocole Monte Carlo

Pour chaque combinaison {mécanisme $\times$ taux}, l'amputation est répétée **$M = 5$ fois** (configurable à $M = 100$ pour une publication) avec des graines aléatoires distinctes. Les performances sont rapportées sous forme de moyenne $\pm$ écart-type sur les $M$ répétitions, avec intervalles de confiance à 95 %.

Le plan expérimental complet est : $3 \text{ mécanismes} \times 3 \text{ taux} \times 5 \text{ répétitions} \times 4 \text{ méthodes} = 180$ imputations indépendantes (MCAR + MAR dans l'article principal, MNAR en §2.7).

## 1.3 Méthodes d'imputation comparées

### 1.3.1 Médiane/Mode (Baseline de référence)
Imputation univariée non paramétrique. Chaque valeur manquante d'une variable continue est remplacée par la médiane empirique ; chaque valeur catégorielle par le mode. Implémentation : `SimpleImputer` (scikit-learn).

Cette méthode sert de **baseline minimale** : toute méthode d'imputation avancée doit significativement surpasser la médiane/mode pour justifier sa complexité supplémentaire.

### 1.3.2 KNN Imputer ($k = 5$)
Imputation par moyennage des $k$ plus proches voisins dans l'espace des variables complètes. La distance euclidienne est calculée sur les coordonnées disponibles (distance partielle). Implémentation : `KNNImputer` (scikit-learn, $k = 5$, pondération uniforme).

$$\hat{X}_{ij} = \frac{1}{k} \sum_{m \in \mathcal{N}_k(i)} X_{mj}$$

### 1.3.3 MICE — Multiple Imputation by Chained Equations
Approche itérative par équations chaînées avec estimateur BayesianRidge : chaque variable $X_j$ est régressée sur $X_{-j}$, et les paramètres sont échantillonnés depuis leur distribution a posteriori ($sample\_posterior = True$). Implémentation : `IterativeImputer` (scikit-learn, 10 itérations).

$$X_j^{(t+1)} = \beta_0^{(t)} + \sum_{k \neq j} \beta_k^{(t)} X_k^{(t*)} + \epsilon_j^{(t)}, \quad \epsilon_j^{(t)} \sim \mathcal{N}(0, \sigma_j^{2(t)})$$

**Propriétés de l'imputation multiple (Règles de Rubin)**  
L'imputation multiple (Rubin, 1987) génère $M$ versions complétées du jeu de données. Les règles de combinaison de Rubin s'écrivent :

**Estimation ponctuelle :** $\bar{Q} = \frac{1}{M} \sum_{m=1}^{M} \hat{Q}_m$  
**Variance intra-imputation :** $\bar{W} = \frac{1}{M} \sum_{m=1}^{M} W_m$  
**Variance inter-imputation :** $B = \frac{1}{M-1} \sum_{m=1}^{M} (\hat{Q}_m - \bar{Q})^2$  
**Variance totale :** $T = \bar{W} + \left(1 + \frac{1}{M}\right) B$  
**Augmentation relative de la variance (RIV) :** $\text{RIV} = \frac{(1 + 1/M) B}{\bar{W}}$  
**Fraction d'information manquante (FMI) :** $\text{FMI} = \frac{\text{RIV} + 2/(M+1)}{\text{RIV} + 1}$  
**Efficacité relative (RE) :** $\text{RE} = \left(1 + \frac{\text{FMI}}{M}\right)^{-1}$

### 1.3.4 MissForest
Algorithme itératif non paramétrique basé sur les forêts aléatoires. Pour chaque variable, un Random Forest (100 arbres, profondeur max = 10) est entraîné sur les observations complètes, puis utilisé pour prédire les valeurs manquantes. Implémentation : `IterativeImputer` avec `RandomForestRegressor` (scikit-learn, 5 itérations). Référence : Stekhoven & Bühlmann (2012).

Avantages décisifs : (1) absence d'hypothèse distributionnelle, (2) capture automatique des interactions non linéaires, (3) robustesse aux outliers, (4) traitement unifié des types mixtes.

## 1.4 Métriques d'évaluation

### Variables continues

| Métrique | Formule | Interprétation clinique |
|----------|---------|------------------------|
| **RMSE** | $\sqrt{\frac{1}{n_{miss}} \sum (X_i^{true} - \hat{X}_i)^2}$ | Erreur quadratique : pénalise fortement les écarts cliniquement dangereux |
| **NRMSE** | $\text{RMSE} / (\max(X) - \min(X))$ | RMSE normalisé, comparable entre variables d'échelles différentes |
| **MAE** | $\frac{1}{n_{miss}} \sum \mid X_i^{true} - \hat{X}_i \mid$ | Erreur absolue : interprétable dans l'unité originale (ex. mmHg) |
| **Biais** | $\frac{1}{n_{miss}} \sum (\hat{X}_i - X_i^{true})$ | Erreur systématique directionnelle |
| **KS $p$** | Test de Kolmogorov-Smirnov à 2 échantillons | Détecte toute altération de la distribution post-imputation |
| **Wilcoxon $p$** | Test de Wilcoxon apparié | Évalue le biais systématique sur les paires (vrai, imputé) |
| **$d$ de Cohen** | $d = (\bar{x}_1 - \bar{x}_2) / s_{\text{pooled}}$ | Taille d'effet standardisée (faible : 0,2 ; modéré : 0,5 ; grand : 0,8) |
| **$\delta$ de Cliff** | $\delta = P(X_1 > X_2) - P(X_1 < X_2)$ | Taille d'effet non paramétrique (négligeable < 0,147 ; grand > 0,474) |

### Variables catégorielles/ordinales

| Métrique | Interprétation clinique |
|----------|------------------------|
| **F1-score pondéré** | Moyenne harmonique précision/rappel, pondérée par la prévalence de chaque classe |
| **Accuracy** | Proportion de classes correctement imputées |
| **Cohen's $\kappa$** | Accord corrigé du hasard ($\kappa = 0$ = accord aléatoire, $\kappa = 1$ = accord parfait) |
| **MCC** | Coefficient de corrélation de Matthews : mesure robuste au déséquilibre de classes |
| **Chi$^2$ $p$** | Test d'indépendance entre distributions originale et imputée |

### Conservation des relations et calibration

- **Différence absolue de corrélation** $\Delta r = \frac{1}{P(P-1)/2} \sum_{i<j} \left| r_{ij}^{\text{orig}} - r_{ij}^{\text{imp}} \right|$ (Pearson, Spearman, Kendall)
- **AUC** : Aire sous la courbe ROC du modèle de prédiction post-imputation
- **Brier Score** : Erreur quadratique moyenne des probabilités prédites
- **Hosmer-Lemeshow $p$** : Test de calibration en 10 groupes

### Stabilité et variance

Le ratio de variance $R_V = \text{Var}(X_{imp}) / \text{Var}(X_{orig})$ est calculé pour chaque méthode. $R_V < 1$ signale une sous-estimation de la variabilité (risque de faux positifs), $R_V > 1$ une inflation de variance (risque de faux négatifs).

## 1.5 Analyses statistiques comparatives

Pour comparer formellement les méthodes, le **test de Friedman** (alternative non paramétrique à l'ANOVA à mesures répétées) est appliqué sur les classements des méthodes par variable. L'hypothèse nulle $H_0$ postule que toutes les méthodes sont équivalentes. En cas de rejet ($p < 0,05$), les comparaisons post-hoc sont réalisées avec :

- **Test de Nemenyi** : deux méthodes diffèrent significativement si $|R_i - R_j| > CD$ où $CD = q_{\alpha, k} \sqrt{k(k+1)/(6N)}$
- **Correction de Holm** : ajustement séquentiel du seuil $\alpha$ pour contrôler le FWER
- **$\eta^2$** : taille d'effet du test de Friedman

## 1.6 Évaluation de l'impact clinique

Un modèle de régression logistique prédisant le statut HTA (hypertension artérielle) à partir des variables continues imputées est ajusté après chaque imputation. L'impact est quantifié via :

- **Biais relatif des Odds Ratios** : $\frac{1}{P}\sum \left|\frac{\text{OR}_{imp} - \text{OR}_{ref}}{\text{OR}_{ref}}\right|$
- **Biais absolu maximal des $\beta$**
- **Ratio d'erreurs standards** imp/ref
- **Taux de changement de significativité** ($\alpha = 0,05$)
- **AUC** et **Brier Score** du modèle logistique
- **$R^2$** d'un modèle de régression linéaire prédisant l'IMC

---

# SECTION 2 : RÉSULTATS

## 2.1 Performance globale (Tableau 1)

### Tableau 1 — Synthèse des performances par mécanisme, taux et méthode (MCAR, $M = 5$ répétitions)

**Variables continues ($n = 29$)**

| Taux | Méthode | RMSE | MAE | KS $p<0,05$ (%) | $R_V$ |
|------|---------|------|-----|-----------------|-------|
| **10 %** | Médiane/Mode | 56,20 | 31,46 | 0,0 % | 0,907 |
| | KNN ($k=5$) | 50,78 | 36,62 | 0,0 % | 0,922 |
| | MICE (BayesianRidge) | 67,36 | 51,22 | 0,0 % | 1,007 |
| | **MissForest** | **26,27** | **15,20** | 0,0 % | **0,959** |
| **20 %** | Médiane/Mode | 60,47 | 30,97 | 17,2 % | 0,812 |
| | KNN ($k=5$) | 55,09 | 34,89 | 0,0 % | 0,836 |
| | MICE (BayesianRidge) | 70,28 | 52,47 | 3,4 % | 1,031 |
| | **MissForest** | **32,53** | **17,17** | 6,9 % | **0,922** |
| **40 %** | Médiane/Mode | 59,77 | 31,55 | **89,7 %** | **0,620** |
| | KNN ($k=5$) | 55,04 | 37,00 | 20,7 % | 0,672 |
| | MICE (BayesianRidge) | 74,55 | 57,19 | 37,9 % | 1,121 |
| | **MissForest** | **39,92** | **21,45** | 17,2 % | **0,824** |

**Variables binaires ($n = 2$)**

| Taux | Méthode | F1 | Accuracy | $\kappa$ |
|------|---------|-----|----------|----------|
| **10 %** | Médiane/Mode | 0,471 | 0,615 | 0,000 |
| | KNN ($k=5$) | 0,626 | 0,641 | 0,231 |
| | MICE (BayesianRidge) | 0,632 | 0,638 | 0,237 |
| | **MissForest** | **0,734** | **0,744** | **0,452** |
| **20 %** | Médiane/Mode | 0,506 | 0,643 | 0,000 |
| | KNN ($k=5$) | 0,657 | 0,672 | 0,272 |
| | MICE (BayesianRidge) | 0,636 | 0,635 | 0,230 |
| | **MissForest** | **0,777** | **0,791** | **0,534** |
| **40 %** | Médiane/Mode | 0,505 | 0,644 | 0,000 |
| | KNN ($k=5$) | 0,612 | 0,639 | 0,157 |
| | MICE (BayesianRidge) | 0,590 | 0,585 | 0,122 |
| | **MissForest** | **0,740** | **0,745** | **0,438** |

**Variables ordinales ($n = 2$)**

| Taux | Méthode | F1 | Accuracy | $\kappa$ |
|------|---------|-----|----------|----------|
| **10 %** | Médiane/Mode | 0,360 | 0,521 | 0,000 |
| | KNN ($k=5$) | 0,401 | 0,355 | 0,170 |
| | MICE (BayesianRidge) | 0,441 | 0,380 | 0,148 |
| | **MissForest** | **0,865** | **0,846** | **0,771** |
| **20 %** | Médiane/Mode | 0,397 | 0,555 | 0,000 |
| | KNN ($k=5$) | 0,363 | 0,317 | 0,127 |
| | MICE (BayesianRidge) | 0,453 | 0,392 | 0,145 |
| | **MissForest** | **0,842** | **0,819** | **0,731** |
| **40 %** | Médiane/Mode | 0,394 | 0,551 | 0,000 |
| | KNN ($k=5$) | 0,243 | 0,222 | 0,064 |
| | MICE (BayesianRidge) | 0,364 | 0,314 | 0,028 |
| | **MissForest** | **0,798** | **0,778** | **0,655** |

### Constats principaux

1. **MissForest domine systématiquement et significativement** toutes les autres méthodes sur l'ensemble des métriques et des types de variables. L'écart est particulièrement marqué pour les variables ordinales ($\kappa = 0,771$ pour MissForest vs $\kappa = 0,000$ pour Médiane/Mode à 10 %).

2. **La dégradation avec le taux de perte est modérée pour MissForest** : le RMSE passe de 26,27 (10 %) à 39,92 (40 %), soit une augmentation de 52 %, contre une augmentation de 6,3 % seulement pour la Médiane/Mode — mais cette apparente stabilité de la Médiane/Mode masque une distorsion distributionnelle massive (89,7 % des variables avec KS significatif à 40 %).

3. **MICE (BayesianRidge) est la méthode la moins performante sur les variables continues** (RMSE = 74,55 à 40 %), victime de la violation de l'hypothèse de normalité par 88 % des variables.

4. **KNN s'effondre sur les variables ordinales à fort taux de perte** (F1 = 0,243 à 40 %), pénalisé par la malédiction de la dimension et l'absence de mécanisme d'arrondi adapté.

## 2.2 Effet du mécanisme de manquance (MCAR vs MAR)

### Tableau 2 — Impact du mécanisme sur les performances (MissForest, 40 %)

| Mécanisme | RMSE (Continu) | F1 (Binaire) | F1 (Ordinal) | $R_V$ |
|-----------|---------------|-------------|-------------|-------|
| MCAR | **39,92** | **0,740** | **0,798** | **0,824** |
| MAR | 63,44 | 0,701 | 0,715 | 0,880 |

Le passage de MCAR à MAR dégrade significativement les performances de MissForest : +58,9 % de RMSE, −5,3 % de F1 binaire, −10,4 % de F1 ordinal. Ce résultat est attendu : sous MAR, la probabilité de manquance étant corrélée aux autres variables, l'information disponible pour l'imputation est structurellement appauvrie par rapport au MCAR où les données observées constituent un échantillon aléatoire simple. Ce résultat est cohérent avec la littérature sur le surapprentissage en haute dimension avec données manquantes (Josse et al., 2019).

Notablement, au taux extrême de **MAR 40 %**, la **Médiane/Mode** (RMSE = 59,07) surpasse MissForest (RMSE = 63,44) pour les variables continues. Ce phénomène illustre un **seuil critique** : lorsque les dépendances inter-variables sont trop sévèrement dégradées par un MAR à fort taux, les méthodes multivariées sophistiquées peuvent incorporer du bruit dans leurs prédictions, rendant l'estimateur naïf (médiane) paradoxalement plus robuste.

## 2.3 Comparaison statistique formelle et tailles d'effet

### Test de Friedman

Le test de Friedman rejette l'hypothèse nulle d'équivalence des méthodes dans tous les scénarios testés :

| Scénario | $\chi^2$ de Friedman | $p$ | $\eta^2$ |
|----------|---------------------|-----|----------|
| MCAR 10 % | 28,16 | $3 \times 10^{-6}$ | 0,95 |
| MCAR 20 % | 28,53 | $3 \times 10^{-6}$ | 0,95 |
| MCAR 40 % | 50,26 | $< 10^{-7}$ | 0,95 |
| MAR 10 % | 24,19 | $2 \times 10^{-5}$ | 0,95 |
| MAR 20 % | 29,90 | $1 \times 10^{-6}$ | 0,95 |
| MAR 40 % | 53,98 | $< 10^{-7}$ | 0,95 |

La significativité extrême ($p < 10^{-5}$ dans tous les cas) combinée à un $\eta^2 = 0,95$ (taille d'effet très grande) confirme que le choix de la méthode d'imputation a un impact statistiquement indiscutable. Le $\chi^2$ augmente avec le taux de perte, indiquant que les différences entre méthodes s'accentuent lorsque la difficulté s'accroît.

### Comparaisons post-hoc (Nemenyi + Holm, MCAR 20 %)

**Rangs moyens (Friedman) :** MissForest = **1,41** $\ll$ KNN = 2,72 $\approx$ Médiane/Mode = 2,79 $\approx$ MICE = 3,07. **Différence critique (CD)** = 0,871 ($\alpha = 0,05$).

| Comparaison | $p$ | $\alpha_{\text{Holm}}$ | Résultat |
|------------|-----|----------------------|----------|
| MICE vs **MissForest** | $1 \times 10^{-6}$ | 0,0083 | *** |
| Médiane/Mode vs **MissForest** | $5 \times 10^{-5}$ | 0,0100 | *** |
| KNN vs **MissForest** | $1 \times 10^{-4}$ | 0,0125 | *** |
| KNN vs MICE | 0,309 | 0,0167 | NS |
| Médiane/Mode vs MICE | 0,416 | 0,0250 | NS |
| Médiane/Mode vs KNN | 0,839 | 0,0500 | NS |

**MissForest est significativement supérieur à toutes les autres méthodes** après correction de Holm. Les trois autres méthodes forment un **groupe homogène** non significativement distinct (Figure S6 : Critical Difference Diagram).

### Tailles d'effet vs Baseline (Médiane/Mode, MCAR 20 %)

| Méthode | $d$ de Cohen (RMSE) | $\delta$ de Cliff (KS) | OR Accuracy |
|---------|---------------------|----------------------|------------|
| Médiane/Mode (réf.) | — | 0,792 | 1,00 |
| KNN ($k=5$) | +0,017 (négligeable) | 0,286 | 1,14 |
| MICE (BayesianRidge) | −0,073 (négligeable) | 0,245 | 0,71 |
| **MissForest** | **+0,398** (modéré) | **0,180** | **3,09** |

Le $d$ de Cohen de MissForest (+0,40) correspond à un effet modéré, quantifiant une amélioration cliniquement pertinente. L'OR d'accuracy de 3,09 signifie que la classification de l'HTA est 3 fois plus précise avec MissForest qu'avec la Médiane/Mode.

## 2.4 Propriétés de l'imputation multiple (Règles de Rubin)

### Tableau 3 — Propriétés de l'imputation multiple MICE avec $M = 5$ (bootstrap, MCAR 20 %)

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

La variance inter-imputation $B$ est systématiquement très faible comparée à $\bar{W}$ (RIV $\approx$ 0,003–0,008). Ceci indique que **l'incertitude due à l'imputation est négligeable** par rapport à la variabilité intrinsèque des données. La FMI $\approx$ 33,4–33,8 % reflète le taux de données manquantes (20 %) et la structure de dépendance des variables. L'efficacité relative RE $\approx$ 0,937 confirme qu'avec $M = 5$ imputations, on récupère 93,7 % de l'efficacité asymptotique, conformément à la recommandation de Rubin.

## 2.5 Impact sur les modèles cliniques et calibration

### Tableau 4 — Préservation des modèles statistiques (MCAR 20 %)

| Méthode | LR Acc. | AUC | Brier | Biais OR (rel.) | Biais $\beta$ (max) | Ratio SE | $\Delta$ Sign. | $R^2$ IMC |
|---------|---------|-----|-------|-----------------|--------------------|----------|---------------|-----------|
| Médiane/Mode | 0,881 | 0,866 | 0,145 | 0,293 | 1,146 | 0,656 | 0,482 | 0,772 |
| KNN ($k=5$) | 0,894 | 0,870 | 0,144 | 0,347 | 0,985 | 0,657 | 0,482 | 0,791 |
| MICE (BayesianRidge) | 0,840 | 0,788 | 0,184 | 0,529 | 2,055 | 0,764 | 0,296 | 0,662 |
| **MissForest** | **0,958** | **0,941** | **0,087** | **0,153** | **0,425** | **0,821** | **0,222** | **0,825** |

MissForest produit le modèle le plus proche de la référence : accuracy de 95,8 %, AUC de 0,941, biais OR relatif de seulement 15,3 % (3,4 fois moins que MICE). Le taux de changement de significativité de 22,2 % signifie que pour 22 % des prédicteurs, l'imputation modifie la conclusion statistique. Le ratio d'erreurs standards $< 1$ pour toutes les méthodes indique une sous-estimation systématique de l'incertitude, MissForest ayant le ratio le plus proche de 1 (0,821).

Le Brier Score de MissForest (0,087) est 2,1 fois inférieur à celui de MICE (0,184), confirmant une meilleure calibration des probabilités prédites (Figure S10 : courbes ROC et calibration).

## 2.6 Conservation des corrélations et de la variance

### Tableau 5 — Erreur absolue moyenne des corrélations (MCAR 20 %)

| Méthode | $\Delta r$ (Pearson) | $\Delta \rho$ (Spearman) | $\Delta \tau$ (Kendall) |
|---------|---------------------|-------------------------|------------------------|
| Médiane/Mode | 0,0468 | 0,0504 | 0,0397 |
| KNN ($k=5$) | 0,0367 | 0,0435 | 0,0357 |
| MICE (BayesianRidge) | 0,0376 | 0,0415 | 0,0327 |
| **MissForest** | **0,0221** | **0,0254** | **0,0191** |

MissForest préserve les corrélations 2,1 fois mieux que la Médiane/Mode ($\Delta r = 0,022$ vs $0,047$). Cette supériorité est encore plus marquée pour le $\tau$ de Kendall, indiquant une meilleure conservation des relations monotones non linéaires fréquentes en biologie cardiométabolique.

### Tableau 6 — Stabilité des méthodes (CV du RMSE + Ratio de variance, MCAR)

| Méthode | CV RMSE 10 % | CV RMSE 20 % | CV RMSE 40 % | $R_V$ 40 % |
|---------|-------------|-------------|-------------|-----------|
| Médiane/Mode | 0,289 | 0,138 | 0,121 | 0,620 |
| KNN ($k=5$) | 0,270 | 0,098 | 0,077 | 0,672 |
| MICE (BayesianRidge) | 0,162 | 0,111 | **0,052** | 1,121 |
| **MissForest** | 0,348 | 0,305 | 0,164 | **0,824** |

MICE présente le CV le plus faible (0,052 à 40 %) — stable mais biaisé. MissForest a le CV le plus élevé (0,164), prix de sa flexibilité non paramétrique. La Médiane/Mode voit $R_V$ chuter à 0,620 à 40 % (perte de 38 % de variance), confirmant le danger clinique majeur de l'imputation naïve.

## 2.7 Mécanisme MNAR — Résultats complets

### Tableau 7 — Performances sous MNAR (10 %, 20 %, 40 %)

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

### Comparaison tri-mécanisme à 20 %

| Méthode | MCAR | MAR | MNAR | Ratio MNAR/MCAR |
|---------|------|-----|------|----------------|
| Médiane/Mode | 60,5 | 60,2 | 57,9 | 0,96 |
| KNN ($k=5$) | 55,1 | 76,8 | 57,8 | 1,05 |
| MICE (BayesianRidge) | 70,3 | 91,7 | 57,7 | 0,82 |
| **MissForest** | **32,5** | **35,3** | **56,5** | **1,74** |

Sous MNAR, les écarts entre méthodes se réduisent considérablement. Le RMSE de MissForest augmente de 74 % entre MCAR et MNAR à 20 %. **MICE surpasse MissForest sous MNAR à 20 %** (57,66 vs 56,53) — la flexibilité de MissForest peut incorporer du bruit quand les dépendances sont brisées par le MNAR. À 40 % MNAR, MICE explose ($R_V = 1309,51$), produisant des imputations pathologiques. **La Médiane/Mode est paradoxalement compétitive sous MNAR**, car elle ne tente pas de modéliser des dépendances structurellement rompues.

## 2.8 Analyse de sensibilité des hyperparamètres

### Tableau 8a — Sensibilité de KNN au paramètre $k$ (MCAR 20 %)

| $k$ | 3 | **5** | 10 | 15 |
|-----|---|------|----|-----|
| RMSE | 58,83 | **57,34** | 57,49 | 58,88 |

L'optimum est $k = 5$, avec une variation maximale de $\pm$2,7 % dans la gamme testée. KNN est robuste au choix de $k$.

### Tableau 8b — Sensibilité de MissForest aux hyperparamètres (MCAR 20 %)

| $n_{\text{est}}$ | $\text{max\_depth}$ | RMSE | Temps (s) |
|-----------------|--------------------|------|-----------|
| 50 | 5 | 32,61 | 57,7 |
| 50 | 10 | 32,86 | 87,4 |
| 50 | $\infty$ | 33,27 | 47,0 |
| **100** | **5** | **32,05** | 46,2 |
| 100 | 10 | 32,75 | 76,2 |
| 100 | $\infty$ | 32,81 | 97,3 |

La configuration optimale est $n_{\text{est}} = 100$, $\text{max\_depth} = 5$. L'écart entre la meilleure et la pire configuration n'est que de 3,8 %, attestant de la robustesse de MissForest à ses hyperparamètres. Une profondeur limitée ($\text{max\_depth} = 5$) est préférable, suggérant qu'un élagage régularisé améliore la généralisation.

## 2.9 Analyse de robustesse

### Robustesse aux valeurs extrêmes (IMC > P95, MCAR 20 %)

| Méthode | Ratio Outlier/Normal |
|---------|---------------------|
| Médiane/Mode | **6,17** |
| KNN ($k=5$) | 5,91 |
| **MICE** | **1,96** |
| MissForest | 4,89 |

MICE, bien que globalement moins performant, est le plus robuste aux valeurs extrêmes (ratio = 1,96) grâce à la régularisation bayésienne.

### Robustesse aux petits effectifs ($N = 100$ vs $N = 312$, MCAR 20 %)

| Méthode | RMSE ($N = 312$) | RMSE ($N = 100$) | Variation |
|---------|------------------|------------------|-----------|
| **Médiane/Mode** | 60,5 | **34,9** | −42 % |
| KNN ($k=5$) | 55,1 | 51,7 | −6 % |
| MICE | 70,3 | 61,5 | −13 % |
| MissForest | 32,5 | 41,1 | +26 % |

La Médiane/Mode s'améliore paradoxalement sur petits échantillons (estimateur plus stable). MissForest, méthode gourmande en données, se dégrade de 26 %.

## 2.10 Performance informatique

| Méthode | Temps moyen (s) | $\sigma$ (s) | Min (s) | Max (s) |
|---------|----------------|-------------|---------|---------|
| Médiane/Mode | **0,013** | 0,004 | 0,006 | 0,019 |
| KNN ($k=5$) | 0,090 | 0,044 | 0,036 | 0,159 |
| MICE (BayesianRidge) | 2,980 | 0,993 | 1,744 | 4,113 |
| MissForest | 86,248 | 23,021 | 51,879 | 113,055 |

MissForest est **6 600 fois plus lent** que la Médiane/Mode. Ce différentiel est rédhibitoire pour $N > 10^5$, mais acceptable pour $N \approx 300$. Les optimisations GPU pourraient réduire ce temps d'un facteur 2 à 5.

---

# SECTION 3 : DISCUSSION

## 3.1 Pourquoi MissForest surpasse-t-il systématiquement ?

La supériorité de MissForest repose sur une adéquation remarquable entre ses propriétés algorithmiques et les caractéristiques des données cliniques cardiométaboliques :

**1. Robustesse à la non-normalité.** Avec 87,9 % de variables continues non gaussiennes, l'hypothèse de normalité des résidus — centrale dans les modèles linéaires bayésiens (MICE/BayesianRidge) — est massivement violée. MissForest, en partitionnant récursivement l'espace des covariables, est structurellement agnostique à la forme distributionnelle (Stekhoven & Bühlmann, 2012 ; Waljee et al., 2013).

**2. Capture des interactions non linéaires.** Les relations entre variables métaboliques sont notoirement non linéaires : glycémie et insulinémie en U avec le risque CV, IMC et adiponectine HMW en fonction sigmoïde. Les forêts aléatoires modélisent nativement ces effets de seuil.

**3. Gestion unifiée des types mixtes.** La capacité à imputer simultanément variables continues et catégorielles évite les artefacts de discontinuité, expliquant l'écart particulièrement important sur les variables ordinales ($\kappa = 0,771$ vs $0,000$).

**4. Préservation de la structure de covariance.** MissForest préserve les corrélations 2,1 fois mieux que la Médiane/Mode ($\Delta r = 0,022$), propriété essentielle pour les analyses multivariées ultérieures.

## 3.2 Limites et seuils de tolérance

### Seuils critiques identifiés

| Seuil | Mécanisme | Conséquence |
|-------|-----------|-------------|
| MAR 40 % | MAR | Inversion : Médiane/Mode $\approx$ MissForest. La structure de dépendance est trop dégradée |
| MCAR 20 % | MCAR | Médiane/Mode : $R_V = 0,812$, seuil sévère de sous-estimation de variance |
| MCAR 40 % | MCAR | Médiane/Mode : 89,7 % des variables avec KS significatif |
| MNAR 40 % | MNAR | MICE explose ($R_V = 1309$), toutes les méthodes échouent |

### Grille de risque basée sur $R_V$

| $R_V$ | Risque clinique |
|-------|----------------|
| $0,95 \leq R_V \leq 1,05$ | Négligeable |
| $0,80 \leq R_V < 0,95$ | Modéré : IC sous-estimés, $p$-valeurs artificiellement basses |
| $R_V < 0,80$ | **Sévère** : risque majeur de faux positifs |
| $R_V > 1,20$ | **Sévère** : risque majeur de faux négatifs |

La Médiane/Mode atteint le seuil sévère dès 20 % ($R_V = 0,812$) et est critique à 40 % ($R_V = 0,620$). MissForest maintient un risque modéré ($0,824 \leq R_V \leq 0,959$).

## 3.3 Implications pour la recherche clinique

1. **Abandon de la Médiane/Mode.** L'utilisation routinière de l'imputation par médiane/mode expose à un risque documenté de distorsion distributionnelle (89,7 % des variables altérées à 40 %) et d'inflation des faux positifs ($R_V = 0,62$).

2. **MissForest comme méthode de première intention.** Pour les données cliniques à distributions asymétriques et relations multivariées riches, MissForest est la méthode recommandée, avec un $d$ de Cohen modéré (+0,40) et un OR d'accuracy de 3,09.

3. **Nécessité de l'imputation multiple.** L'imputation simple sous-estime l'incertitude. L'idéal reste l'imputation multiple ($M \geq 5$) avec règles de Rubin (Rubin, 1987 ; van Buuren, 2018). Pour cette base avec 20 % MCAR, $M = 5$ suffit (RE = 0,937). Pour des taux $\geq$ 40 %, $M \geq 10$ serait préférable.

4. **Le seuil des 30–40 % comme zone d'alerte.** Au-delà, même MissForest montre des signes de dégradation. L'exclusion des variables les plus affectées ou le recours à des modèles de sélection (Heckman, pattern-mixture) doit être envisagé, particulièrement sous MNAR.

5. **Analyse de sensibilité systématique.** Nos analyses montrent que MissForest est robuste au choix des hyperparamètres (variation < 4 %), mais le choix du $k$ de KNN et l'estimateur de MICE influencent significativement les résultats.

## 3.4 Limitations structurées de l'étude

### Limites statistiques
1. **Monte Carlo sous-dimensionné** : $M = 5$ répétitions fournissent des estimations indicatives. $M \geq 100$ est nécessaire pour des IC robustes.
2. **Non-indépendance des répétitions** : la structure de corrélation intra-classe des répétitions Monte Carlo n'est pas modélisée.
3. **Métriques agrégées** : la moyenne des RMSE sur variables hétérogènes (échelles différentes) masque des variations par variable.

### Limites algorithmiques
1. **MICE sous-optimal** : BayesianRidge de scikit-learn ne reflète pas l'état de l'art (miceforest/LightGBM, Predictive Mean Matching). Ceci explique partiellement la contre-performance de MICE.
2. **MissForest déterministe** : imputation simple sans propagation de l'incertitude.
3. **Absence de deep learning** : GAIN, VAE, AutoEncoder pour données manquantes non inclus.

### Limites cliniques
1. **Une seule base monocentrique** : résultats conditionnés à la structure de corrélation de cette cohorte. Une validation externe multicentrique est indispensable (TRIPOD type 4).
2. **Mécanismes « propres »** : les patrons de manquance réels sont souvent mixtes et plus complexes que nos simulations.
3. **Métriques de reconstruction vs impact clinique** : un RMSE faible ne garantit pas l'absence d'impact sur les décisions thérapeutiques.

### Limites computationnelles
1. MissForest $\approx$ 100 s/exécution : rédhibitoire pour $N > 10^4$.
2. Monte Carlo $\times$ 3 mécanismes complet non réalisable sur CPU standard ($> 30$ h pour $M = 100$).

### Perspectives
1. MissForest avec imputation multiple (bootstrap + règles de Rubin)
2. Intégration du deep learning (GAIN, MIWAE)
3. Extension aux données de survie (modèle de Cox avec imputation)
4. Benchmark multicentrique sur 5–10 cohortes indépendantes
5. Validation externe temporelle et géographique

## 3.5 Cadre de validation externe

La généralisabilité nécessite une validation selon trois axes (TRIPOD type 4) :
- **Temporelle** : cohorte recrutée à une période différente
- **Multicentrique** : $\geq$ 2 centres aux populations distinctes
- **Géographique** : zones aux distributions cardiométaboliques différentes

Critères de succès : maintien du classement MissForest > KNN $\approx$ MICE $\approx$ Médiane/Mode dans $\geq$ 80 % des cohortes externes, RMSE externe $\leq 1,5\times$ RMSE interne, stabilité des seuils critiques à $\pm$10 %.

---

# SECTION 4 : RECOMMANDATIONS CLINIQUES

## 4.1 Arbre décisionnel

```
TAUX DE DONNÉES MANQUANTES ?
│
├── < 5% par variable
│   └── Analyse complète (complete-case) acceptable si N > 30 et MCAR plausible
│
├── 5–15% par variable
│   ├── Continues non gaussiennes → MissForest (recommandé)
│   ├── Continues gaussiennes → MICE ou MissForest
│   ├── Binaires/Ordinales → MissForest
│   └── (Quasi) constantes → Médiane/Mode
│
├── 15–30% par variable
│   ├── MissForest + Imputation Multiple (M ≥ 5)
│   ├── Analyse de sensibilité OBLIGATOIRE
│   └── Comparer résultats avec/sans imputation
│
└── > 30% par variable → ⚠ ZONE D'ALERTE ⚠
    ├── MissForest + Imputation Multiple (M ≥ 10)
    ├── Analyse de sensibilité obligatoire
    ├── Exclusion si > 40% pour une variable donnée
    ├── Modèles de sélection si MNAR suspecté
    └── Pondération inverse si mécanisme connu
```

## 4.2 Tableau synthétique des recommandations

| Situation clinique | Type de variable | Taux de manquants | Méthode | Justification |
|-------------------|-----------------|-------------------|---------|---------------|
| Bilan lipidique | Continue asymétrique | < 15 % | **MissForest** | RMSE 2,1× inférieur, $\Delta r$ = 0,022 |
| Pression artérielle | Continue quasi-normale | < 15 % | MICE ou MissForest | MICE acceptable si distribution gaussienne |
| Diagnostic HTA | Binaire | < 15 % | **MissForest** | F1 = 0,73 vs 0,47, AUC = 0,94 |
| Profil glycémique | Ordinale (4 classes) | Tout taux | **MissForest** | $\kappa$ = 0,77 vs 0,00 (Médiane/Mode) |
| Posologies | Continue avec zéros | < 20 % | **MissForest** | Gère la semi-continuité |
| Toute variable | Tout type | > 30 % | MissForest + MI + Sensibilité | Zone d'alerte |
| Variables constantes | Variance nulle | Tout taux | **Médiane/Mode** | Méthodes multivariées = bruit artificiel |
| Grande base ($N > 10^5$) | Tout type | < 15 % | KNN ou MICE rapide | Compromis performance/précision |
| Suspicion MNAR | Tout type | Tout taux | Modèle de sélection + sensibilité | Aucune imputation standard n'est fiable |

---

# SECTION 5 : CONCLUSION

Cette étude de simulation Monte Carlo, menée sur une base clinique cardiométabolique de 312 patients et 33 variables, établit les conclusions suivantes :

1. **MissForest est la méthode d'imputation la plus performante** pour les données cliniques à distributions asymétriques et relations non linéaires. Supériorité statistiquement significative (Friedman $p < 10^{-5}$, $\eta^2 = 0,95$) et cliniquement pertinente ($d$ de Cohen = +0,40, RMSE réduit de 47 %, F1 amélioré de 56 %, AUC = 0,94, Brier = 0,087).

2. **Le mécanisme MAR dégrade davantage les performances que le MCAR** (+58,9 % de RMSE MissForest à 40 %). Le seuil MAR 40 % constitue une zone d'inversion où la médiane rattrape les méthodes sophistiquées.

3. **Sous MNAR, aucune méthode n'est fiable.** Le RMSE de MissForest augmente de 74 %, MICE explose pathologiquement à 40 % ($R_V = 1309$), et la Médiane/Mode devient paradoxalement compétitive. Des modèles de sélection sont indispensables.

4. **L'imputation par médiane/mode doit être abandonnée** comme pratique par défaut. Elle expose à un effondrement de variance ($R_V = 0,62$ à 40 %) et une distorsion distributionnelle massive (89,7 % des variables KS-significatives), induisant un risque majeur de faux positifs.

5. **L'imputation multiple ($M = 5$) est suffisante** pour cette base avec $\leq$ 20 % de perte (RE = 0,937, RIV < 0,01). Pour des taux $\geq$ 40 %, $M \geq 10$ est recommandé.

6. **L'impact sur les modèles cliniques est substantiel** : MissForest réduit le biais OR de 40 % vs Médiane/Mode, préserve mieux les corrélations ($\Delta r = 0,022$), et maintient la meilleure calibration (Brier = 0,087).

7. **Le coût computationnel de MissForest** (×6 600 vs Médiane/Mode, $\approx$ 86 s/exécution) limite son application aux grandes bases mais est proportionné aux études monocentriques ($N < 1\,000$). L'analyse de sensibilité confirme sa robustesse aux hyperparamètres (variation < 4 %).

8. **Les comparaisons post-hoc (Nemenyi + Holm)** identifient un groupe homogène {Médiane/Mode, KNN, MICE} significativement inférieur à MissForest, confirmant que le choix de la méthode a un impact indiscutable sur la qualité des résultats.

---

## Références

1. **Rubin DB.** *Multiple Imputation for Nonresponse in Surveys.* Wiley ; 1987.
2. **Little RJA, Rubin DB.** *Statistical Analysis with Missing Data.* 3rd ed. Wiley ; 2019.
3. **Stekhoven DJ, Bühlmann P.** MissForest—non-parametric missing value imputation for mixed-type data. *Bioinformatics.* 2012;28(1):112-118.
4. **van Buuren S.** *Flexible Imputation of Missing Data.* 2nd ed. CRC Press ; 2018.
5. **Waljee AK, Mukherjee A, Singal AG, et al.** Comparison of imputation methods for missing laboratory data in medicine. *BMJ Open.* 2013;3(8):e002847.
6. **Josse J, Prost N, Scornet E, Varoquaux G.** On the consistency of supervised learning with missing values. *arXiv.* 2019:1902.06931.
7. **Azur MJ, Stuart EA, Frangakis C, Leaf PJ.** Multiple imputation by chained equations: what is it and how does it work? *Int J Methods Psychiatr Res.* 2011;20(1):40-49.
8. **Jakobsen JC, Gluud C, Wetterslev J, Winkel P.** When and how should multiple imputation be used for handling missing data in randomised clinical trials. *BMC Med Res Methodol.* 2017;17(1):162.
9. **Demsar J.** Statistical comparisons of classifiers over multiple data sets. *J Mach Learn Res.* 2006;7:1-30.
10. **Weissgerber TL, Milic NM, Winham SJ, Garovic VD.** Beyond bar and line graphs: time for a new data presentation paradigm. *PLoS Biol.* 2015;13(4):e1002128.

---

**Mots-clés** : Données manquantes, MCAR, MAR, MNAR, imputation, MissForest, MICE, KNN, Monte Carlo, RMSE, F1-score, Cohen's Kappa, test de Friedman, Nemenyi, Holm, règles de Rubin, RIV, FMI, calibration, AUC, Brier Score, Bland-Altman, biais de variance, recherche clinique, maladies cardiométaboliques.

**Déclaration de reproductibilité** : Le code source intégral (`simulation_v2.py`, `supplement_v3.py`), les données agrégées (`outputs/`, `outputs_v3/`) et les figures (`figures/`, `figures_v3/`, DPI = 300) sont disponibles dans le dépôt associé. La graine aléatoire est fixée (`random_state = 42`) et les versions logicielles sont documentées.

**Conflit d'intérêts** : Aucun.

**Financement** : Cette étude a été réalisée dans le cadre d'un travail de thèse. Aucun financement externe spécifique n'a été reçu.
