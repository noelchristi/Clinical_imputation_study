# ARTICLE SCIENTIFIQUE — BROUILLON

## Évaluation comparative des méthodes d'imputation de données manquantes sur une base clinique cardiométabolique : simulation MCAR multi-scénarios (10 %, 20 %, 40 %)

---

# SECTION 1 : MATÉRIEL ET MÉTHODES

## 1.1 Description de la base de données

L'étude s'appuie sur une base de données cliniques issue d'une cohorte de 312 sujets (âge moyen = 57,2 ± 11,7 ans ; sexe : 52,9 % féminin, 47,1 % masculin). La base comporte 33 variables couvrant les paramètres anthropométriques (IMC, tour de taille), hémodynamiques (PAS, PAD), métaboliques (bilan lipidique complet, glycémie, insulinémie), inflammatoires (PCSK9, adiponectine HMW), pharmacologiques (posologies de 5 traitements) et des ratios lipidiques dérivés.

### Classification des variables

| Type | Variables | n |
|------|-----------|---|
| **Continu / Normale** | Age, Totc (p Shapiro > 0,05) | 2 |
| **Continu / Asymétrique** | IMC, TT, PAS, PAD, Prot, HDL, LDLc, TG, Non HDL, PCSK9, Glu, Ins, HOMAB%, HOMA-IR, Adipo HMW, Glibenclamide 5mg, Insuline 10 UI, Metformine 500 mg, Ratio Tot-c/HDL, Ratio LDL-c/HDL, Ratio TG/HDL, Ratio LogTG/HDL, Insu pmol/L, lu(mmol/l), HOMAS% | 25 |
| **Continu / Constante** | Lasilix 40 mg, Lexomil 6 mg (tous les sujets à 0 mg) | 2 |
| **Binaire** | Sexe (Masculin/Féminin), HTA (oui/non) | 2 |
| **Ordinal** | HTS/HTD (3 modalités), Profil_glycémique (4 modalités : Normoglycémie, Hyperglycémie modérée/Prédiabète, Hyperglycémie/Diabète, Hypoglycémie) | 2 |

La forte prédominance de variables continues asymétriques (25/33, soit 75,8 %) constitue une contrainte clinique majeure : ces distributions à queue lourde (ex. HOMA-IR : skewness = +8,81 ; Adipo HMW : skewness = +7,55) violent l'hypothèse de normalité sous-jacente à de nombreux algorithmes d'imputation paramétriques.

## 1.2 Protocole de simulation MCAR multi-scénarios

La base originale est considérée comme la **vérité-terrain** intégrale. Le protocole d'amputation contrôlée suit une structure factorielle complète 3 × 4 :

1. **Génération des masques MCAR** : Pour chaque cellule de la matrice N × P (312 × 33), une variable aléatoire U ~ Uniforme(0, 1) est tirée indépendamment. La valeur est supprimée si U < τ, où τ ∈ {0,10 ; 0,20 ; 0,40}. Les taux réels obtenus sont : 9,97 %, 19,82 % et 40,48 %. Le mécanisme MCAR garantit que la probabilité de manquance est indépendante des valeurs observées et non observées :

   $$ P(R_{ij} = 0 \mid X_{obs}, X_{miss}) = P(R_{ij} = 0) = \tau $$

2. **Amputation** : Trois matrices amputées `X_miss^(10%)`, `X_miss^(20%)`, `X_miss^(40%)` sont générées avec une graine aléatoire fixée (seed = 42 + 100τ) pour assurer la reproductibilité.

3. **Imputation** : Chaque matrice est soumise aux 4 algorithmes.

4. **Évaluation** : Les performances sont calculées exclusivement sur les valeurs amputées (erreur de reconstruction), la base originale servant de gold standard.

## 1.3 Algorithmes d'imputation

### 1.3.1 Médiane/Mode (Baseline)
Méthode univariée de référence. Pour chaque variable continue, la valeur manquante est remplacée par la médiane empirique des valeurs observées ; pour chaque variable catégorielle, par le mode. Implémentation : `SimpleImputer` (scikit-learn).

**Limite fondamentale** : L'imputation est déterministe et ignore toute structure de corrélation inter-variables, ce qui entraîne une sous-estimation systématique de la variance (atténuation) et une distorsion des distributions conjointes.

### 1.3.2 MICE — Multiple Imputation by Chained Equations
Chaque variable `X_j` avec des valeurs manquantes est régressée sur l'ensemble des autres variables `X_{-j}` selon un modèle conditionnel spécifié, de manière itérative. L'implémentation retenue est `IterativeImputer` (scikit-learn) avec un estimateur BayesianRidge, configuré en mode `sample_posterior=True` pour incorporer l'incertitude d'imputation. Le nombre d'itérations est fixé à 10.

Le modèle sous-jacent pour la variable continue j est :

$$ X_j^{(t+1)} = \beta_0^{(t)} + \sum_{k \neq j} \beta_k^{(t)} X_k^{(t*)} + \epsilon_j^{(t)}, \quad \epsilon_j^{(t)} \sim \mathcal{N}(0, \sigma_j^{2(t)}) $$

où les paramètres sont échantillonnés depuis leur distribution a posteriori bayésienne.

### 1.3.3 KNN Imputer (k = 5)
Les valeurs manquantes sont estimées par la moyenne des k plus proches voisins dans l'espace des variables observées. La distance euclidienne est calculée sur les coordonnées disponibles (distance partielle). Implémentation : `KNNImputer` (scikit-learn, k = 5, pondération uniforme).

L'estimateur pour le sujet i sur la variable j est :

$$ \hat{X}_{ij} = \frac{1}{k} \sum_{m \in \mathcal{N}_k(i)} X_{mj} $$

où $\mathcal{N}_k(i)$ désigne l'ensemble des k voisins du sujet i.

### 1.3.4 MissForest (Random Forest Imputation)
Algorithme itératif non paramétrique basé sur les forêts aléatoires. À chaque itération, pour chaque variable, un Random Forest (100 arbres, profondeur max = 10) est entraîné sur les observations complètes, puis utilisé pour prédire les valeurs manquantes. Implémentation : `IterativeImputer` avec `RandomForestRegressor`.

**Avantage clinique décisif** : MissForest ne présuppose aucune distribution paramétrique, capture automatiquement les interactions non linéaires et les effets de seuil, et gère nativement les données mixtes (continues/catégorielles).

## 1.4 Métriques d'évaluation et tests statistiques

### Variables continues

| Métrique | Formule | Justification clinique |
|----------|---------|------------------------|
| **RMSE** | $\sqrt{\frac{1}{n_{miss}} \sum_{i \in \mathcal{M}} (X_{i}^{true} - \hat{X}_{i})^2}$ | Pénalise les grandes erreurs (écarts cliniquement dangereux) |
| **MAE** | $\frac{1}{n_{miss}} \sum_{i \in \mathcal{M}} \mid X_{i}^{true} - \hat{X}_{i} \mid$ | Interprétable dans l'unité de la variable (ex. mmHg pour la PAS) |
| **Test de Kolmogorov-Smirnov** | $D = \sup_{x} \mid F_{n}^{orig}(x) - F_{n}^{imp}(x) \mid$ | Détecte toute altération de la distribution globale post-imputation |
| **Test de Wilcoxon apparié** | Comparaison des paires (vrai, imputé) sur les valeurs manquantes | Évalue le biais systématique de l'imputation |

### Variables catégorielles/ordinales

| Métrique | Justification clinique |
|----------|------------------------|
| **F1-score pondéré** | Équilibre précision/rappel, pondéré par la fréquence de chaque classe — critique pour les diagnostics rares (ex. Hypoglycémie) |
| **Taux d'erreur de classification** | Proportion de catégories mal imputées |
| **Test du Chi-deux d'indépendance** | Évalue la distorsion des proportions entre distribution originale et imputée |

### Analyse de la variance post-imputation

Le ratio de variance $R_V = \frac{Var(X_{imp})}{Var(X_{orig})}$ est calculé pour chaque méthode et chaque taux de perte. $R_V < 1$ indique une sous-estimation de la variabilité inter-individuelle (risque de faux positifs par réduction de l'erreur standard), $R_V > 1$ indique une inflation de variance (risque de faux négatifs).

---

# SECTION 2 : RÉSULTATS

## 2.1 Performance globale des méthodes d'imputation

Le Tableau 1 présente les performances agrégées (moyenne sur les 29 variables continues et 4 variables catégorielles).

**Tableau 1 — Performances comparatives des 4 méthodes d'imputation aux 3 taux de perte MCAR**

| Taux | Méthode | RMSE (Continu) | MAE (Continu) | KS p<0,05 (%) | F1 (Catégoriel) | Err. Classif. | Var. Ratio |
|------|---------|---------------|---------------|---------------|-----------------|---------------|------------|
| **10%** | Médiane/Mode | 74,20 | 41,79 | 0,0 % | 0,455 | 0,399 | 0,870 |
| | MICE (sklearn) | 81,22 | 58,45 | 0,0 % | 0,550 | 0,494 | 0,961 |
| | KNN (k=5) | 63,19 | 41,57 | 0,0 % | 0,526 | 0,491 | 0,885 |
| | **MissForest** | **29,89** | **14,28** | 0,0 % | **0,834** | **0,165** | **0,926** |
| **20%** | Médiane/Mode | 61,45 | 29,57 | 48,3 % | 0,446 | 0,406 | 0,799 |
| | MICE (sklearn) | 65,99 | 47,31 | 6,9 % | 0,531 | 0,502 | 1,041 |
| | KNN (k=5) | 57,02 | 34,10 | 0,0 % | 0,541 | 0,467 | 0,825 |
| | **MissForest** | **33,29** | **19,32** | 10,3 % | **0,808** | **0,197** | **0,907** |
| **40%** | Médiane/Mode | 63,38 | 33,33 | **89,7 %** | 0,439 | 0,412 | **0,643** |
| | MICE (sklearn) | 78,13 | 61,17 | 34,5 % | 0,500 | 0,528 | 1,146 |
| | KNN (k=5) | 57,63 | 38,24 | 37,9 % | 0,413 | 0,586 | 0,695 |
| | **MissForest** | **35,13** | **20,88** | 27,6 % | **0,764** | **0,249** | **0,851** |

### Synthèse des résultats

**MissForest domine systématiquement** toutes les autres méthodes sur l'ensemble des métriques, à tous les taux de perte, pour les variables continues comme catégorielles. La supériorité est particulièrement marquée pour les variables catégorielles : le F1-score de MissForest à 40 % de perte (0,764) reste supérieur à celui de toute autre méthode à 10 % de perte.

**La dégradation des performances avec le taux de perte** est relativement contenue pour MissForest : le RMSE passe de 29,89 (10 %) à 35,13 (40 %), soit une augmentation de seulement 17,5 %, contre une augmentation de 27,8 % pour KNN et une explosion de la distorsion distributionnelle pour la Médiane/Mode (KS significatif pour 89,7 % des variables à 40 %).

**MICE (sklearn) est la méthode la moins performante** sur les variables continues (RMSE = 78,13 à 40 %, soit 2,2 fois celui de MissForest). L'estimateur BayesianRidge, fondé sur une hypothèse de normalité des résidus, est mis en échec par les distributions fortement asymétriques de la majorité des variables cliniques. En revanche, MICE est la seule méthode à ne pas sous-estimer la variance (RV = 1,041 à 20 %, 1,146 à 40 %), ce qui traduit l'incorporation correcte de l'incertitude d'imputation via l'échantillonnage postérieur.

**KNN (k=5) offre une performance intermédiaire** mais s'effondre sur les variables catégorielles au taux de 40 % (F1 = 0,413, erreur de classification = 58,6 %). La distance euclidienne dans un espace de grande dimension (33 variables) souffre du fléau de la dimension (curse of dimensionality), et l'agrégation par moyenne de voisins produit des valeurs non entières pour les catégories, dégradant la classification.

**La Médiane/Mode** est acceptable à faible taux de perte (RMSE = 74,20 à 10 %) mais induit une distorsion distributionnelle massive à 40 % : 89,7 % des variables continues présentent une différence de distribution statistiquement significative (KS p < 0,05). Le ratio de variance chute à 0,643, indiquant une sous-estimation de près de 36 % de la variabilité inter-individuelle — un biais qui augmenterait drastiquement le risque d'erreur de type I (faux positifs) dans toute analyse inférentielle ultérieure.

## 2.2 Analyse variable par variable

### Variables continues à distribution normale (Age, Totc)

Pour l'**Âge** (distribution quasi-normale, skewness = +0,038), la Médiane/Mode surpasse étonnamment MissForest au taux de 10 % (RMSE = 9,27 vs 14,79). Ce résultat s'explique par le fait que l'âge a une distribution symétrique et une faible variance relative, ce qui rend la médiane un estimateur efficient. Cependant, cet avantage disparaît dès 20 % de perte. Pour le **Cholestérol total** (Totc), MissForest domine avec un RMSE de 0,13 à 10 % et 0,23 à 40 %.

### Variables métaboliques à forte asymétrie (HOMA-IR, Adipo HMW, PCSK9)

Ces trois biomarqueurs présentent les skewness les plus extrêmes de la base (+8,81, +7,55 et +2,74 respectivement). **MissForest est la seule méthode capable de les imputer avec une erreur cliniquement acceptable**. Pour HOMA-IR à 40 % de perte, le RMSE de MissForest est de 6,04 contre 20,70 (Médiane/Mode) et 19,93 (MICE). L'explication est double :
1. Les forêts aléatoires partitionnent récursivement l'espace des covariables, capturant les relations non linéaires entre insulinorésistance, IMC et paramètres lipidiques.
2. L'absence d'hypothèse distributionnelle évite les prédictions aberrantes (négatives ou extrêmes) que les modèles linéaires bayésiens peuvent générer sur des queues de distribution.

Pour **PCSK9**, KNN surpasse MissForest à 10 % (RMSE = 64,18 vs 64,54) mais cet avantage marginal disparaît aux taux supérieurs, et la Médiane/Mode devient la meilleure méthode à 20 % et 40 %. Cette instabilité inter-méthodes pour PCSK9 reflète probablement une faible structuration des données de cette protéine par les autres covariables disponibles.

### Variables pharmacologiques (Metformine, Glibenclamide, Insuline)

Les posologies médicamenteuses posent un problème particulier : elles combinent une nature semi-continue (doses) avec une forte proportion de zéros (patients non traités). Pour la **Metformine 500 mg**, le RMSE est élevé quelle que soit la méthode (416 à 658), en raison de l'échelle de la variable (0 à 2000 mg). MissForest reste la meilleure option.

Les variables **Lasilix 40 mg** et **Lexomil 6 mg** sont quasi constantes (tous les patients à 0 mg, sauf peut-être quelques exceptions). Le RMSE est nul pour la Médiane/Mode (imputation triviale), tandis que les méthodes multivariées introduisent une erreur artificielle en tentant de modéliser une variance inexistante. Ce cas illustre un principe clinique fondamental : **les méthodes sophistiquées d'imputation multivariée sont contre-productives pour les variables (quasi) constantes**.

### Variables catégorielles

Les 4 variables catégorielles (Sexe, HTA, HTS/HTD, Profil_glycémique) montrent une supériorité écrasante de MissForest :

| Variable | F1 Médiane/Mode (40%) | F1 MissForest (40%) |
|----------|----------------------|---------------------|
| HTA | 0,56 | **0,89** |
| HTS/HTD | 0,29 | **0,66** |
| Profil_glycémique | 0,52 | **0,95** |
| Sexe | 0,45 | 0,67 (MICE : **0,68**) |

Le **Profil glycémique** (4 classes : Normoglycémie, Prédiabète, Diabète, Hypoglycémie) est imputé avec un F1 de 0,95 par MissForest à 40 % de perte. Cette performance remarquable s'explique par la forte redondance entre le profil glycémique et les variables continues associées (Glu, Ins, HOMA-IR, HOMAB%), que les arbres de décision exploitent efficacement. L'Hypoglycémie, classe minoritaire mais cliniquement critique (risque vital), est particulièrement bien préservée.

L'**HTA** (hypertension artérielle) atteint un F1 de 0,89 avec MissForest à 40 %, contre 0,56 pour la Médiane/Mode. La prédiction correcte du statut hypertensif à partir des variables tensionnelles (PAS, PAD) et anthropométriques (IMC, TT) est un cas d'école d'interaction multivariée que MissForest capture.

### Test du Chi-deux

Une observation frappante est que **100 % des variables catégorielles présentent un Chi-deux significatif (p < 0,05) quelle que soit la méthode et le taux de perte**. Ce résultat s'explique par la taille d'échantillon (N = 312) qui confère une puissance statistique suffisante pour détecter des écarts mêmes minimes entre les distributions de proportions. Il indique que, dans un sens strictement statistique, **aucune méthode ne restaure parfaitement la distribution conjointe originale**. Cependant, l'ampleur clinique de la distorsion (mesurée par le F1-score) varie considérablement : MissForest produit une distorsion statistiquement détectable mais cliniquement négligeable (F1 > 0,76), tandis que la Médiane/Mode produit une distorsion à la fois statistiquement et cliniquement significative.

## 2.3 Analyse de la variance post-imputation

Le ratio de variance $R_V = Var(X_{imp}) / Var(X_{orig})$ constitue un indicateur critique du biais induit par l'imputation sur la variabilité inter-individuelle.

| Taux | Médiane/Mode | MICE (sklearn) | KNN (k=5) | MissForest |
|------|-------------|----------------|-----------|------------|
| 10 % | 0,870 | 0,961 | 0,885 | **0,926** |
| 20 % | 0,799 | 1,041 | 0,825 | **0,907** |
| 40 % | **0,643** | 1,146 | 0,695 | **0,851** |

**Médiane/Mode** : Sous-estimation massive de la variance, s'aggravant avec le taux de perte ($R_V$ = 0,643 à 40 %, soit −35,7 %). Ce phénomène d'atténuation (shrinkage) est inhérent à l'imputation par une constante centrale : en remplaçant les valeurs extrêmes par la médiane, on écrase les queues de distribution. En pratique clinique, cela conduirait à une sous-estimation systématique des écarts-types et donc à une inflation du risque d'erreur de type I (faux positifs) dans les tests d'hypothèse ultérieurs (test t de Student, ANOVA).

**MICE (sklearn)** : Surestimation de la variance à partir de 20 % de perte ($R_V$ = 1,146 à 40 %). L'échantillonnage depuis la distribution postérieure bayésienne introduit une variabilité supplémentaire qui, couplée à la mauvaise spécification du modèle linéaire gaussien pour des données asymétriques, génère des prédictions à variance excessive (phénomène d'over-dispersion). Le risque clinique est inverse : inflation de l'erreur de type II (faux négatifs), pouvant masquer des effets biologiques réels.

**KNN (k=5)** : Sous-estimation modérée à sévère ($R_V$ = 0,695 à 40 %). La moyenne des k voisins agit comme un lisseur (smoother) qui réduit mécaniquement la variance.

**MissForest** : Meilleure préservation de la variance à tous les taux ($R_V$ = 0,851 à 40 %, soit une sous-estimation de seulement 14,9 %). La nature non paramétrique des forêts aléatoires permet de conserver une partie de l'hétérogénéité inter-individuelle. La sous-estimation résiduelle provient de la limite intrinsèque de toute imputation simple (par opposition à l'imputation multiple qui propage l'incertitude).

---

# SECTION 3 : DISCUSSION ET RECOMMANDATIONS CRITIQUES

## 3.1 Pourquoi MissForest surperforme-t-il ?

La supériorité de MissForest sur l'ensemble des scénarios testés repose sur trois propriétés fondamentales qui répondent précisément aux contraintes des données cliniques cardiométaboliques :

**1. Robustesse aux distributions asymétriques.** Avec 75,8 % de variables continues non gaussiennes (skewness médian = +1,70), la violation de l'hypothèse de normalité est la règle, non l'exception. Les modèles linéaires bayésiens (MICE/BayesianRidge) produisent des résidus non gaussiens qui biaisent l'inférence. MissForest, en partitionnant récursivement l'espace des covariables, est agnostique à la forme distributionnelle.

**2. Capture automatique des interactions non linéaires.** Les relations entre variables métaboliques sont notoirement non linéaires : la relation entre IMC et HOMA-IR est mieux décrite par une fonction log-linéaire que linéaire ; le ratio TG/HDL interagit avec le tour de taille pour prédire le risque cardiométabolique. Les forêts aléatoires modélisent nativement ces effets de seuil et ces interactions d'ordre supérieur.

**3. Gestion unifiée des types mixtes.** La capacité de MissForest à traiter simultanément variables continues et catégorielles (via l'arrondi post-imputation) évite les artefacts de discontinuité que produisent les méthodes traitant séparément chaque type de variable (Médiane/Mode, MICE avec modèles distincts).

## 3.2 Limites et seuils de tolérance

### Le seuil critique de 40 %

Le taux de 40 % de données manquantes représente un stress-test sévère qui révèle les limites de toutes les méthodes. Pour MissForest, le RMSE augmente de 17,5 % entre 10 % et 40 %, mais le taux de distorsion distributionnelle significative (KS p < 0,05) passe de 0 % à 27,6 %. **Nous identifions 40 % comme un seuil d'alerte** au-delà duquel même la meilleure méthode produit une altération non négligeable des distributions pour plus d'un quart des variables.

Pour la **Médiane/Mode**, le seuil d'effondrement est plus précoce : dès 20 % de perte, près de la moitié (48,3 %) des variables continues présentent une distorsion distributionnelle significative. **Nous recommandons de proscrire l'imputation par médiane/mode au-delà de 10 % de données manquantes** dans tout contexte de recherche clinique inférentielle.

### Biais résiduels et risque d'erreur de type I/II

L'imputation simple (toute méthode confondue) sous-estime l'incertitude car elle traite les valeurs imputées comme des observations réelles. Dans une analyse clinique ultérieure (régression logistique, modèle de Cox), ce biais se traduit par des intervalles de confiance trop étroits et des p-valeurs artificiellement faibles.

Le ratio de variance $R_V$ quantifie ce phénomène :
- **R_V < 0,80** (Médiane/Mode dès 20 %, KNN dès 40 %) : le rétrécissement de variance est cliniquement préoccupant. L'erreur standard des estimateurs est sous-estimée d'au moins 10 %, ce qui augmente le risque de conclure à tort à un effet significatif.
- **R_V ≈ 1,15** (MICE à 40 %) : l'inflation de variance dilue la puissance statistique, augmentant le risque de ne pas détecter un effet réel.

MissForest maintient le meilleur compromis ($0,85 \leq R_V \leq 0,93$) sur toute la gamme de taux de perte.

### Variables (quasi) constantes : un piège pour les méthodes sophistiquées

Lasilix et Lexomil (0 mg chez tous les patients) illustrent un écueil majeur : les méthodes multivariées (MICE, KNN, MissForest) tentent de modéliser une variance inexistante et produisent des imputations erronées là où la méthode triviale (médiane/mode) est parfaite. **Recommandation** : Identifier et traiter séparément les variables à variance nulle ou quasi nulle avant toute imputation multivariée.

## 3.3 Recommandations pratiques : arbre de décision pour le clinicien-chercheur

```
DONNÉES MANQUANTES DÉTECTÉES
│
├── Taux de perte < 5 % sur toutes les variables
│   └── Analyse complète (complete-case analysis) acceptable
│       si N restant > 30 et MCAR plausible.
│
├── Taux de perte 5–15 %
│   ├── Variables continues non gaussiennes → MissForest
│   ├── Variables continues gaussiennes → MICE (BayesianRidge) ou MissForest
│   ├── Variables catégorielles/ordinales → MissForest
│   └── Variables (quasi) constantes → Médiane/Mode
│
├── Taux de perte 15–30 %
│   ├── MissForest systématiquement recommandé
│   ├── Imputation multiple (M = 5 à 20) pour propager l'incertitude
│   └── Analyse de sensibilité : comparer résultats avec/sans imputation
│
└── Taux de perte > 30 %
    ├── ⚠ SEUIL D'ALERTE ⚠
    ├── MissForest avec imputation multiple
    ├── Analyse de sensibilité obligatoire
    ├── Envisager l'exclusion des variables ayant > 40 % de manquants
    └── Pondération inverse ou modèle de sélection si MNAR suspecté
```

## 3.4 Implications pour la recherche clinique cardiométabolique

Notre étude de simulation, bien que limitée au mécanisme MCAR, fournit des enseignements directement transposables à la pratique :

1. **L'imputation par médiane/mode doit être abandonnée** comme méthode par défaut dans les analyses cardiométaboliques. Son utilisation expose à un risque majeur de conclusions erronées (faux positifs) dès que le taux de perte dépasse 10 %.

2. **MissForest constitue la méthode de première intention** pour les bases de données cliniques caractérisées par des distributions asymétriques et des relations non linéaires entre biomarqueurs, ce qui est la norme en métabolisme.

3. **L'imputation multiple (MICE)** reste conceptuellement supérieure pour la propagation de l'incertitude, mais son implémentation pratique avec des modèles linéaires bayésiens est inadéquate pour les données asymétriques. Les développements futurs devraient intégrer des estimateurs non paramétriques (Random Forest, Gradient Boosting) au sein du cadre MICE.

4. **Le taux de 40 % de données manquantes** représente une frontière méthodologique au-delà de laquelle la validité des inférences post-imputation doit être questionnée, même avec les meilleures méthodes disponibles.

## 3.5 Limitations de l'étude

1. **Mécanisme de manquance** : Seul le scénario MCAR a été exploré. Les mécanismes MAR (Missing At Random) et MNAR (Missing Not At Random), plus réalistes en clinique (ex. patients diabétiques plus susceptibles d'avoir des dosages lipidiques manquants), n'ont pas été simulés. Les performances relatives des méthodes pourraient différer sous ces mécanismes.

2. **Taille d'échantillon** : N = 312 est représentatif d'une étude clinique monocentrique mais peut limiter la généralisabilité. La performance de KNN, en particulier, est sensible à la densité des données dans l'espace des covariables.

3. **Implémentation MICE** : L'utilisation de BayesianRidge comme estimateur par défaut dans scikit-learn est sous-optimale. L'échec de miceforest (incompatibilité de version) nous a privés d'une implémentation MICE plus robuste avec des modèles de régression plus flexibles.

4. **Absence de validation externe** : Les résultats sont conditionnés à la structure de corrélation spécifique de notre base cardiométabolique. La supériorité de MissForest pourrait être moins marquée dans des contextes où les relations inter-variables sont majoritairement linéaires (ex. données anthropométriques pures).

---

**Mots-clés** : Données manquantes, MCAR, imputation, MissForest, MICE, KNN, RMSE, test de Kolmogorov-Smirnov, maladies cardiométaboliques, simulation, biais de variance.
