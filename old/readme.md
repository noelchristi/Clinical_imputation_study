Rôle : Tu es un Data Scientist Senior spécialisé en Biostatistique et Recherche Clinique, expert en gestion des données manquantes.

Contexte : 
J'ai accès à une base de données cliniques complète present dans le workspace. Je souhaite mener une étude de simulation de perte de données (amputation contrôlée) pour évaluer l'efficacité de différentes méthodes d'imputation sur nos variables cliniques spécifiques (continues, binaires et ordinales).

Objectif :
Rédige un brouillon complet d'article scientifique (section Méthodes, Résultats et Discussion), incluant l'implémentation de tous les tests statistiques requis et l'analyse critique associée.

Instructions méthodologiques à exécuter :
1. Analyse de la base : Identifie et classe les variables de la base (ex: Continu/Normal, Continu/Asymétrique, Catégoriel, Ordinal).
2. Simulation de perte (Amputation Multi-Scénarios) : Génère artificiellement trois scénarios distincts de données manquantes de type MCAR (Missing Completely At Random) aux taux stricts de : 10%, 20%, et 40%. Conserve la base originale comme référence absolue pour le calcul des erreurs.
3. Imputation : Applique et compare les méthodes suivantes :
   - Imputation par la Médiane/Mode (Baseline)
   - MICE (Multiple Imputation by Chained Equations)
   - KNN Imputer (k-Nearest Neighbors)
   - MissForest (Random Forest Imputation)
4. Évaluation & Tests Statistiques :
   - Pour les variables continues : Calcule le RMSE (Root Mean Squared Error) et l'MAE (Mean Absolute Error). Applique un test de Kolmogorov-Smirnov ou de Wilcoxon pour comparer la distribution originale vs imputée.
   - Pour les variables catégorielles/ordinales : Calcule le score F1-pondéré et le Taux d'erreur de classification. Applique un test du Chi-deux ou le test exact de Fisher pour évaluer la distorsion des proportions.

Format attendu pour le rendu du brouillon :

# SECTION 1 : MATÉRIEL ET MÉTHODES
- Description du protocole de simulation multi-scénarios (MCAR 10%, 20%, 40%).
- Présentation mathématique et logique succincte des 4 algorithmes d'imputation retenus.
- Justification du choix des métriques (RMSE, F1-Score) et des tests de distribution (Wilcoxon, Chi-deux).

# SECTION 2 : RÉSULTATS (Code et tableaux multi-scénarios)
- Fournis le script d'implémentation complet (en Python avec scikit-learn/miceforest OU en R avec mice/missForest). Le code doit inclure une boucle propre, commentée et modulaire pour automatiser l'amputation, l'imputation et l'évaluation aux trois taux de perte (10%, 20%, 40%).
- Génère un tableau comparatif synthétique structuré ainsi :
  | Taux de Perte | Variable Clinique | Type de Variable | Méthode | Métrique d'Erreur | P-value (Test de Distorsion) |
- Rédige une description textuelle rigoureuse analysant la dégradation progressive des performances à mesure que le taux de perte augmente.

# SECTION 3 : DISCUSSION ET RECOMMANDATIONS CRITIQUES
- Analyse variable par variable et selon le taux de perte : Explique POURQUOI telle méthode a surperformé ou échoué sur tel type de donnée clinique. Analyse en détail la résistance des algorithmes face au scénario extrême (40% de perte).
- Biais induits et limites de tolérance : Analyse comment chaque méthode affecte la variance des données et détermine le seuil critique où l'imputation modifie de façon inacceptable la distribution originale, augmentant le risque d'erreur de type I (faux positifs) dans les analyses cliniques ultérieures.
- Recommandation finale : Conclus par une règle de décision pratique et un arbre de choix pour les cliniciens confrontés à des données manquantes selon le volume et la nature des variables.

Règles de style : Utilise un ton académique, précis et scientifique. Ne fais aucune généralité vide. Sois spécifique aux contraintes biologiques et médicales des variables cliniques.