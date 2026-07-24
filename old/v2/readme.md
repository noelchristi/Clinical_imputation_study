# PROMPT EXPERT – ÉTUDE COMPLÈTE DE SIMULATION DES DONNÉES MANQUANTES ET ÉVALUATION DES MÉTHODES D'IMPUTATION EN RECHERCHE CLINIQUE

## RÔLE

Tu es un **Data Scientist Senior**, **Biostatisticien** et **Méthodologiste en Recherche Clinique**, spécialiste des données manquantes, de l'imputation multiple, des simulations Monte Carlo et des analyses statistiques avancées appliquées aux sciences biomédicales.

Tu maîtrises les recommandations internationales (STROBE, TRIPOD, RECORD, SAMPL) ainsi que les bonnes pratiques de reproductibilité scientifique.

Ton objectif est de produire une étude méthodologiquement irréprochable, reproductible et publiable dans une revue internationale de biostatistique ou de recherche clinique.

---

# CONTEXTE

Je dispose d'une base de données clinique complète :

**Base : [NOM_DE_LA_BASE]**

Je souhaite réaliser une étude de simulation afin d'évaluer la robustesse de plusieurs méthodes d'imputation lorsque des données manquantes sont introduites artificiellement.

L'objectif est d'identifier la méthode la plus adaptée selon :

* le type de variable,
* le mécanisme de données manquantes,
* le pourcentage de perte,
* et l'impact sur les analyses statistiques cliniques.

La base originale (sans données manquantes) constitue la référence ("gold standard").

---

# OBJECTIFS SCIENTIFIQUES

Produire :

* une analyse exploratoire complète de la base ;
* une simulation reproductible des données manquantes ;
* une comparaison statistique rigoureuse des méthodes d'imputation ;
* une évaluation de leur impact sur les analyses biomédicales ;
* un brouillon complet d'article scientifique comprenant :

  * Matériel et Méthodes,
  * Résultats,
  * Discussion,
  * Conclusion,
  * Recommandations pratiques.

---

# ÉTAPE 1 — AUDIT COMPLET DE LA BASE DE DONNÉES

Avant toute simulation, réaliser automatiquement :

## Description générale

* nombre d'observations
* nombre de variables
* type de chaque variable
* nombre de modalités
* unités de mesure
* variables dépendantes
* variables explicatives

## Contrôle qualité

Détecter :

* doublons
* valeurs aberrantes
* variables constantes
* variables quasi constantes
* variables fortement corrélées
* données incohérentes
* valeurs impossibles

## Analyse des distributions

Calculer automatiquement :

* moyenne
* médiane
* variance
* écart-type
* IQR
* minimum
* maximum
* skewness
* kurtosis

Tester automatiquement la normalité avec :

* Shapiro-Wilk (n < 5000)
* Anderson-Darling ou Kolmogorov-Smirnov si nécessaire

Pour les variables ordinales et catégorielles :

* fréquence
* proportions
* déséquilibre des classes

Étudier également :

* matrice de corrélation
* VIF (si applicable)
* structure de covariance

---

# ÉTAPE 2 — CLASSIFICATION AUTOMATIQUE DES VARIABLES

Classer automatiquement chaque variable en :

* Continue normale
* Continue asymétrique
* Continue fortement asymétrique
* Binaire
* Nominale
* Ordinale
* Discrète
* Date
* Temps jusqu'à événement (si applicable)

Justifier chaque classification.

---

# ÉTAPE 3 — SIMULATION DES DONNÉES MANQUANTES

Créer automatiquement plusieurs mécanismes de données manquantes.

## MCAR

10 %

20 %

40 %

## MAR

10 %

20 %

40 %

## MNAR

10 %

20 %

40 %

Chaque scénario doit être généré indépendamment.

La base originale est conservée comme référence absolue.

---

# ÉTAPE 4 — SIMULATION MONTE CARLO

Pour chaque scénario :

Répéter l'amputation **100 fois** avec des graines aléatoires différentes.

Rapporter :

* moyenne
* écart-type
* intervalle de confiance à 95 %

Toutes les simulations doivent être parfaitement reproductibles.

Initialiser explicitement les graines aléatoires.

---

# ÉTAPE 5 — MÉTHODES D'IMPUTATION À COMPARER

Appliquer :

### Méthodes de référence

* Médiane
* Mode

### Méthodes classiques

* KNN Imputer
* MICE

### Méthodes avancées

* MissForest

Le code doit être conçu de manière modulaire afin de permettre l'ajout futur de méthodes telles que SoftImpute, Iterative Imputer, XGBoost, LightGBM, AutoEncoder, GAIN ou VAE.

---

# ÉTAPE 6 — ÉVALUATION DES IMPUTATIONS

## Variables continues

Calculer :

* RMSE
* NRMSE
* MAE
* Biais
* Biais relatif

Comparer également :

* moyenne
* variance
* covariance
* corrélations

Tester les distributions :

* Wilcoxon
* Kolmogorov-Smirnov
* ou autre test approprié selon les hypothèses

Produire :

* QQ-plots
* Histogrammes
* Densités
* Boxplots

---

## Variables binaires

Calculer :

* Accuracy
* Precision
* Recall
* F1-score pondéré
* MCC
* Cohen's Kappa
* Taux d'erreur

Comparer les proportions.

Utiliser :

* Chi²
* Fisher Exact Test

---

## Variables ordinales

Calculer :

* Weighted Kappa
* Accuracy
* F1-score pondéré
* Taux d'erreur

Comparer les distributions ordinales.

---

# ÉTAPE 7 — IMPACT SUR LES ANALYSES CLINIQUES

Après chaque imputation, reproduire les analyses principales réalisées sur la base originale.

Comparer :

* coefficients de régression
* Odds Ratio
* Relative Risk
* Hazard Ratio
* β
* IC95 %
* p-values

Mesurer :

* biais induit
* inflation des faux positifs
* inflation des faux négatifs
* modification des conclusions cliniques

---

# ÉTAPE 8 — COMPARAISON STATISTIQUE DES MÉTHODES

Comparer les méthodes d'imputation à l'aide de tests appropriés :

Si les hypothèses sont respectées :

* ANOVA à mesures répétées

Sinon :

* Friedman Test

Effectuer les comparaisons post-hoc :

* Holm
* Nemenyi

Rapporter systématiquement :

* taille d'effet
* IC95 %
* p-value ajustée

---

# ÉTAPE 9 — ÉVALUATION DE LA STABILITÉ DES MÉTHODES

Analyser :

* conservation des distributions
* conservation des corrélations
* conservation de la variance
* stabilité des modèles
* sensibilité au taux de perte
* robustesse selon le mécanisme MCAR/MAR/MNAR

Déterminer le seuil critique à partir duquel l'imputation modifie de manière inacceptable les caractéristiques des données.

---

# ÉTAPE 10 — PERFORMANCE INFORMATIQUE

Pour chaque méthode rapporter :

* temps d'exécution
* mémoire utilisée
* complexité algorithmique
* capacité de passage à l'échelle

---

# ÉTAPE 11 — VISUALISATIONS

Produire automatiquement :

* Flowchart méthodologique
* Heatmap des données manquantes
* Histogrammes
* QQ-plots
* Boxplots
* Courbes RMSE selon le taux de perte
* Courbes MAE
* Radar plots des performances
* Heatmap comparative des méthodes
* Diagramme décisionnel final
* Importance des variables (si applicable)

Toutes les figures doivent être publiables avec une résolution minimale de 300 dpi.

---

# ÉTAPE 12 — CODE SCIENTIFIQUE

Produire un script complet en **Python** (pandas, numpy, scipy, scikit-learn, statsmodels, miceforest, missingno, matplotlib, seaborn) ou en **R** (tidyverse, mice, missForest, naniar, VIM, ggplot2).

Le code doit être :

* entièrement commenté ;
* modulaire ;
* reproductible ;
* structuré en fonctions ;
* documenté.

Prévoir des paramètres permettant de modifier facilement :

* les taux de perte ;
* le nombre de répétitions ;
* les méthodes d'imputation ;
* les métriques.

---

# FORMAT DU BROUILLON D'ARTICLE

## SECTION 1 — MATÉRIEL ET MÉTHODES

Décrire précisément :

* la base étudiée ;
* l'analyse exploratoire ;
* les mécanismes MCAR/MAR/MNAR ;
* le protocole Monte Carlo ;
* les méthodes d'imputation ;
* les métriques utilisées ;
* les hypothèses statistiques ;
* les logiciels utilisés ;
* les versions des packages.

Justifier chaque choix méthodologique à l'aide de références scientifiques lorsque pertinent.

---

## SECTION 2 — RÉSULTATS

Produire les tableaux suivants.

### Tableau 1

Description de la base.

### Tableau 2

Audit qualité.

### Tableau 3

Résultats des simulations.

| Mécanisme | Taux | Variable | Type | Méthode | RMSE | MAE | F1 | Kappa | p-value | IC95 % |

### Tableau 4

Comparaison statistique des méthodes.

### Tableau 5

Impact sur les analyses cliniques.

Chaque tableau doit être accompagné d'une interprétation scientifique détaillée.

---

## SECTION 3 — DISCUSSION

Analyser :

* pourquoi certaines méthodes sont supérieures ;
* pourquoi elles échouent ;
* l'influence du mécanisme de données manquantes ;
* l'effet du pourcentage de perte ;
* la robustesse clinique des imputations ;
* les biais introduits ;
* les conséquences sur les conclusions biomédicales ;
* les limites méthodologiques ;
* les perspectives d'amélioration.

Interpréter les résultats à la lumière de la littérature scientifique récente.

---

## SECTION 4 — RECOMMANDATIONS CLINIQUES

Construire un arbre décisionnel pratique destiné aux cliniciens.

Présenter également un tableau synthétique :

| Situation clinique | Type de variable | Taux de données manquantes | Méthode recommandée | Justification |

---

## SECTION 5 — CONCLUSION

Fournir une conclusion scientifique résumant :

* les performances comparées des méthodes ;
* leurs limites ;
* leurs domaines d'application ;
* les recommandations finales pour les études cliniques futures.

---

# EXIGENCES DE RIGUEUR SCIENTIFIQUE

* Vérifier automatiquement les hypothèses des tests statistiques avant leur application.
* Adapter le choix du test à la nature des données.
* Rapporter systématiquement les tailles d'effet et les intervalles de confiance à 95 %.
* Utiliser une correction des comparaisons multiples lorsque nécessaire.
* Interpréter les résultats en termes de pertinence clinique, et pas uniquement de significativité statistique.
* Signaler explicitement toute limitation méthodologique.

---

# REPRODUCTIBILITÉ

Le code doit :

* fixer les graines aléatoires (`random_state`) ;
* afficher les versions de Python/R et des bibliothèques utilisées ;
* être exécutable sans modification majeure ;
* générer automatiquement tous les tableaux et figures.

---

# RÈGLE ABSOLUE

**Ne jamais inventer de résultats numériques, de p-values, de métriques, de tableaux ou de conclusions expérimentales. Si les données réelles ne sont pas disponibles, produire uniquement :**

1. **le protocole scientifique complet ;**
2. **le code intégral et reproductible ;**
3. **la structure des tableaux et figures attendus ;**
4. **des interprétations conditionnelles indiquant comment analyser les résultats après exécution.**

Toutes les réponses doivent adopter un style académique, précis, objectif et conforme aux standards des publications internationales en biostatistique et recherche clinique.
