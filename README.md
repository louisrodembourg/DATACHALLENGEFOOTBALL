# Football Match Prediction Project ⚽

Ce repository contient le code source d'un pipeline de Machine Learning complet pour la prédiction de résultats de matchs de football. Le projet se concentre sur une ingénierie des données (Feature Engineering) robuste et l'utilisation de modèles de boosting avancés (CatBoost, XGBoost, LightGBM) pour maximiser la précision des prédictions.

## 📋 Table des Matières
- [Contexte](#contexte)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Méthodologie](#méthodologie)
- [Résultats](#résultats)

## 📖 Contexte
L'objectif est de prédire l'issue d'un match (Victoire Domicile, Nul, Victoire Extérieur) à partir de données historiques détaillées sur les équipes et les joueurs. Ce projet a été développé dans le cadre d'un Data Challenge, avec un focus particulier sur la généralisation et l'évitement du sur-apprentissage (overfitting).

## 📂 Structure du Projet

L'arborescence est organisée pour séparer clairement les données, le code, les expériences et les résultats :

```bash
.
├── experiments/       # Logs des configurations et résultats d'entraînement
├── features_engineering/ # Scripts dédiés à la création et analyse de features
├── mlp_test/          # Expérimentations avec des réseaux de neurones (PyTorch)
├── src/               # Code source principal (à venir après refactoring)
├── submission/        # Fichiers de soumission générés (ignorés par git)
├── scripts/           # Scripts principaux d'exécution (train, optimize)
├── requirements.txt   # Dépendances Python
└── README.md          # Documentation du projet
```

*(Note : Les dossiers `data/`, `catboost_info/` et les fichiers temporaires sont exclus du versionning pour garder le dépôt léger.)*

## ⚙️ Installation

1. **Cloner le repository** :
   ```bash
   git clone https://github.com/votre-username/football-prediction.git
   cd football-prediction
   ```

2. **Créer un environnement virtuel** (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Utilisation

Pour lancer le pipeline d'entraînement principal :

```bash
python football_clean.py
```

Arguments optionnels (si implémentés) :
- `--model` : Choix du modèle (`catboost`, `xgb`, `lgb`, `stacking`).
- `--optimize` : Lancer une recherche d'hyperparamètres Optuna.

## 🧠 Méthodologie

### 1. Feature Engineering
Le cœur de la performance réside dans la transformation des données brutes :
- **Agrégation Joueurs** : Regroupement des statistiques individuelles par position (Attaquants, Défenseurs, etc.).
- **Smart Deltas** : Comparaison contextuelle (ex: Duels aériens gagnés par la défense vs perdus par l'attaque adverse).
- **Ratios** : Efficacité de tir, possession relative, conversion.

### 2. Sélection de Variables
Pour éviter le bruit, une sélection rigoureuse est appliquée :
- Suppression des variables corrélées (>95%).
- Sélection basée sur l'importance des features (Feature Importance via LightGBM) pour ne garder que le Top-K (ex: 800 meilleures).

### 3. Modélisation
Utilisation de l'état de l'art en données tabulaires :
- **Modèles de base** : XGBoost, LightGBM, CatBoost.
- **Stacking** : Combinaison des prédictions via une Régression Logistique pour stabiliser les résultats.
- **Optimisation** : Recherche d'hyperparamètres via Optuna (voir `optuna_lab.py`).

## 📊 Résultats
Les performances sont évaluées via une validation croisée (Cross-Validation) stratifiée pour assurer la robustesse du score final. Les logs détaillés de chaque expérience se trouvent dans le dossier `experiments/`.

---
*Projet réalisé pour le Data Challenge [Nom de l'année/promo]. Codé en Python sans génération automatique de code.*
