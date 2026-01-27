# Football Match Prediction Project ⚽

Ce repository contient le code source d'un pipeline de Machine Learning complet pour la prédiction de résultats de matchs de football (Victoire Domicile, Nul, Victoire Extérieur). 

Le projet se distingue par une approche **Data-Centric**, privilégiant une ingénierie des features métier avancée et une architecture modulaire robuste, plutôt que la complexification inutile des modèles.

## 📋 Table des Matières
- [Contexte](#contexte)
- [Architecture du Code](#architecture-du-code)
- [Méthodologie D détaillée](#méthodologie-détaillée)
  - [1. Feature Engineering Avancé](#1-feature-engineering-avancé)
  - [2. Stratégie de Sélection (Signal vs Bruit)](#2-stratégie-de-sélection-signal-vs-bruit)
  - [3. Feature Stacking](#3-feature-stacking)
  - [4. Modélisation](#4-modélisation)
- [Installation & Utilisation](#installation--utilisation)

## 📖 Contexte
L'objectif est de prédire l'issue d'un match de football (`HOME_WINS`, `DRAW`, `AWAY_WINS`) à partir de données historiques hétérogènes : statistiques d'équipes et statistiques individuelles de joueurs.

**Contrainte majeure** : Les données joueurs sont granulaires et variables (le nombre de joueurs et leurs positions changent). Le défi technique principal est donc de synthétiser cette information dans un format tabulaire fixe sans perte de signal.

## 📂 Architecture du Code

Le projet suit les standards de **Clean Architecture** pour garantir la maintenabilité et la reproductibilité :

```bash
.
├── src/                   # 🧠 Cœur du projet (Modules réutilisables)
│   ├── data.py            # Chargement robuste et centralisé des données
│   ├── features.py        # Logique métier : Création et transformation des variables
│   └── utils.py           # Outils transverses (Logging, Sauvegarde, Reproductibilité)
├── scripts/               # ⚡ Exécutables (CLI)
│   ├── run_pipeline.py    # Pipeline Principal (End-to-End : Data -> Train -> Predict)
│   ├── optimize_features.py # Script de sélection de variables (Optuna/LGBM)
│   └── optimize_hyperparameters.py # Tuning des modèles
├── experiments/           # 🧪 Laboratoire : Logs JSON et résultats de chaque run
├── docs/                  # Documentation technique
└── requirements.txt       # Environnement d'exécution
```

## 🧠 Méthodologie Détaillée

### 1. Feature Engineering Avancé
Le code dans `src/features.py` implémente une logique métier footballistique :

*   **Agrégation Positionnelle** : Plutôt que de traiter les joueurs individuellement, nous les regroupons par ligne tactique (Gardiens, Défenseurs, Milieux, Attaquants). Nous calculons ainsi des métriques agrégées (ex: "Puissance offensive des attaquants à domicile").
*   **Smart Matchups (Interactions)** : Création de variables de confrontation directe.
    *   *Exemple* : `DUEL_ATT_H_GK_A` compare statistiquement les tirs des attaquants à domicile vs les arrêts du gardien extérieur.
*   **Dominance Physique & Créative** : Calcul de deltas sur des métriques clés comme les duels aériens gagnés ou les passes clés réussies.
*   **Métriques d'Efficacité (Ratios)** : Pour éviter le biais de volume (une équipe qui a beaucoup la balle tire plus mais n'est pas forcément dangereuse), nous calculons des ratios :
    *   *Conversion Rate* : Buts / Tirs.
    *   *Dangerosity* : Attaques Dangereuses / Attaques Totales.

### 2. Stratégie de Sélection (Signal vs Bruit)
Avec plus de 1500 features générées, le risque de sur-apprentissage (overfitting) est élevé. Le pipeline (`scripts/run_pipeline.py`) applique une double filtration :

1.  **Filtre de Collinéarité** : Suppression des variables corrélées à plus de **95%** (Matrice de corrélation Pearson) pour réduire la redondance.
2.  **Embedded Feature Selection** : Utilisation d'un modèle LightGBM rapide pour calculer l'importance des features ("Gain"). Nous ne conservons que le **Top-K (ex: 800)** variables les plus prédictives. Cela permet d'éliminer le bruit statistique qui perturbe l'apprentissage.

### 3. Feature Stacking
Nous utilisons une technique de **Stacking de Features** :
*   Un modèle auxiliaire (CatBoost Regressor) est entraîné pour prédire non pas le résultat du match, mais la **différence de buts** (`GOAL_DIFF`).
*   Cette prédiction (`PRED_GOAL_DIFF`) est réinjectée comme une "super-feature" dans le modèle principal. Elle agit comme un résumé dense de la dynamique de force entre les deux équipes.

### 4. Modélisation
L'architecture finale repose sur l'état de l'art en données tabulaires :
*   **Modèles** : XGBoost, LightGBM et CatBoost.
*   **Ensemble (Stacking)** : Une option permet de combiner ces trois modèles via une Régression Logistique (Meta-Learner) pour stabiliser la variance des prédictions.
*   **Validation** : Validation Croisée Stratifiée (Stratified K-Fold) pour assurer que la distribution des classes (Victoire/Nul/Défaite) est respectée dans chaque pli de validation.

## ⚙️ Installation & Utilisation

1.  **Installation** :
    ```bash
    pip install -r requirements.txt
    ```

2.  **Lancer le Pipeline** :
    ```bash
    python scripts/run_pipeline.py --model stack
    ```
    *Options de modèles : `lgb` (LightGBM), `xgb` (XGBoost), `cat` (CatBoost), `stack` (Ensemble).*

3.  **Résultats** :
    Les fichiers de soumission sont générés automatiquement dans `submission/` si le score de validation dépasse le seuil de qualité défini.

---
*Projet réalisé pour le Data Challenge "Football : Qui va gagner ?" par QRT. Codé en Python sans génération automatique de code.*