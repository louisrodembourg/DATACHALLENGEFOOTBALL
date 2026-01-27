# Explication détaillée du fichier `football.py`

Ce document détaille le fonctionnement de votre pipeline de prédiction de matchs de football. Le script est structuré en plusieurs étapes claires : chargement, ingénierie des fonctionnalités (feature engineering), sélection de features, entraînement de modèles et soumission.

## 1. Imports et Configuration
Le script commence par importer les bibliothèques nécessaires :
- **Manipulation de données** : `pandas`, `numpy`, `json`.
- **Machine Learning** : `scikit-learn` pour les outils de base (pipeline, imputation, validation croisée, métriques), et les bibliothèques de boosting performantes (`lightgbm`, `xgboost`, `catboost`).
- **Optimisation** : `optuna` (bien que la partie exécution soit commentée).
- **Gestion système** : `os`, `argparse`, `warnings` (pour supprimer les avertissements inutiles).

## 2. Utilitaires & Logging (`save_experiment`)
Cette fonction gère la sauvegarde des résultats de vos expériences.
- Elle sauvegarde la **configuration** de l'expérience (score, paramètres, description) dans un fichier JSON.
- Si le run est validé (score suffisant), elle sauvegarde le **fichier de soumission** (CSV) dans le dossier `experiments/` et crée une version incrémentale (`submission_V{N}.csv`) dans le dossier `submission/` pour faciliter l'envoi sur la plateforme de data challenge.

## 3. Chargement des données (`load_data`)
Cette fonction charge tous les fichiers CSV nécessaires :
- **Données d'équipe** (Home/Away) pour le Train et le Test.
- **Données joueurs** (Home/Away) pour le Train et le Test.
- **Cibles (Targets)** : Résultats des matchs (`Y_train.csv`) et données supplémentaires (`Y_train_supp.csv` contenant par exemple la différence de buts).

## 4. Feature Engineering (Construction des variables)

C'est le cœur de votre analyse de données.

### 4.1 Agrégation des joueurs (`aggregate_players_by_position`)
Les données joueurs sont très détaillées (plusieurs lignes par match). Cette fonction les résume en une ligne par match et par équipe :
- Elle regroupe les joueurs par **Position** (Gardien, Défenseur, Milieu, Attaquant).
- Elle calcule la **Moyenne** et la **Somme** des statistiques pour chaque groupe.
- Résultat : Des colonnes comme `P_HOME_FORWARD_goals_sum` (Somme des buts des attaquants à domicile).

### 4.2 Construction avancée (`build_features_v2`)
Cette fonction combine toutes les données et crée de nouvelles variables intelligentes :
1.  **Fusions** : Assemble les données équipes (Home/Away) et les données joueurs agrégées.
2.  **Ratios et Efficacités** : Calcule des indicateurs de performance relative :
    - *Possession Share* : Part de possession de l'équipe à domicile.
    - *Shot Accuracy* : Ratio tirs cadrés / tirs totaux.
    - *Conversion* : Ratio buts / tirs totaux.
    - *Delta* : Différence de ces ratios entre l'équipe Domicile et Extérieur (ex: `DELTA_CONVERSION`).
3.  **Smart Deltas** : Variables inspirées de la connaissance métier :
    - *Duels Gardien vs Attaquant* : Compare les tirs cadrés adverses aux arrêts du gardien.
    - *Dominance physique* : Différence de duels gagnés.
    - *Créativité* : Différence de passes clés.
4.  **Deltas Classiques** : Pour chaque statistique numérique, calcule simplement `Valeur_Domicile - Valeur_Extérieur`.
5.  **Nettoyage ciblé** : Supprime certaines variables (comme les deltas de victoires passées) identifiées comme nuisibles lors de vos tests précédents.
6.  **Nettoyage basique** : Retire les colonnes constantes ou quasi-constantes et les informations administratives (Noms d'équipes, ID des ligues pour éviter le sur-apprentissage sur des équipes spécifiques).

## 5. Pipeline Principal (`if __name__ == "__main__":`)

C'est le chef d'orchestre qui exécute les étapes dans l'ordre.

### A. Préparation des données
- Charge les données brutes.
- Applique le `build_features_v2` sur les jeux d'entraînement et de test.

### B. Nettoyage et Sélection de Features
1.  **Suppression des corrélations** : Calcule la matrice de corrélation sur le jeu d'entraînement et supprime l'une des deux variables si elles sont corrélées à plus de **95%**. Cela évite la redondance.
2.  **Alignement** : S'assure que le Train et le Test ont exactement les mêmes colonnes.
3.  **Encodage de la cible** : Transforme le résultat (Away Wins, Draw, Home Wins) en chiffres (0, 1, 2) pour les modèles.
4.  **Sélection Top-K** : 
    - Entraîne un modèle LightGBM rapide.
    - Calcule l'importance de chaque variable ("gain").
    - Ne garde que les **800 meilleures features**. C'est une étape cruciale pour réduire le bruit et améliorer la performance.

### C. Feature Auxiliaire (Goal Diff)
- Vous utilisez un modèle de régression (`CatBoostRegressor`) pour prédire la **différence de buts** (`GOAL_DIFF_HOME_AWAY`).
- Cette prédiction est ajoutée comme une nouvelle variable (`PRED_GOAL_DIFF`) dans le jeu de données. C'est une technique de "Stacking" de features très efficace.

### D. Construction du Modèle
- **Choix du modèle** : L'utilisateur peut choisir (via arguments ou menu) entre LightGBM, XGBoost, CatBoost ou un Stacking (combinaison des trois).
- **Hyperparamètres** : Le code contient des dictionnaires de paramètres (`params_lgb`, `params_xgb`, `params_cat`) déjà optimisés (probablement via Optuna dans des expériences précédentes).
- Si "Stacking" est choisi, il entraîne les 3 modèles et utilise une `LogisticRegression` pour combiner leurs prédictions.

### E. Validation et Entraînement
1.  **Estimation Rapide (Validation)** :
    - Coupe le jeu d'entraînement en deux (80% train, 20% validation).
    - Entraîne le modèle sur les 80%.
    - Calcule le score sur les 20% restants.
2.  **Décision (Gate)** :
    - Si le score estimé est inférieur à `0.4850`, le script s'arrête (pour ne pas perdre de temps ou soumettre un mauvais modèle).
    - Sinon, il continue.
3.  **Entraînement Final** :
    - Ré-entraîne le modèle choisi sur **100% des données d'entraînement**.

### F. Soumission
- Prédit les probabilités pour le jeu de Test.
- Convertit les probabilités en prédiction "dure" (0 ou 1 strict) : la classe avec la plus haute probabilité prend tout.
- Génère le fichier CSV final prêt à être soumis.
