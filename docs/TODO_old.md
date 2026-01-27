# TODO List - Football Data Challenge

# 🔴 PHASE 1 : Architecture & Validation (Fondations)
*Ces étapes sont indispensables. Si elles sont bancales, le reste ne sert à rien.*

- [ ] **Mettre en place le Workflow de Sauvegarde**
    - [x] Intégrer la fonction `save_experiment()` (CSV + JSON) dans le code.
    - [x] Règle d'or : Ne plus jamais écraser un fichier. Chaque run doit être loggé.
    - [x] Gestion des versions automatique (V1, V2...).

- [ ] **Valider la stratégie "Cross-Validation" (CV)**
    - [x] Activer `cross_val_score(cv=5)` avant l'entraînement final.
    - [x] **Objectif** : Ne soumettre sur le site que si `cv_score_local > 0.4820`. Faire confiance au CV local plus qu'au Leaderboard public.

- [ ] **Upgrade vers le Stacking (vs Voting)**
    - [x] Remplacer le `VotingClassifier` par un `StackingClassifier`.
    - [x] Utiliser `LogisticRegression` comme `final_estimator`.
    - [x] Utiliser `cross_val_predict` pour générer les méta-features proprement (éviter le data leakage).
    - [x] *Impact attendu : +0.5% à +1% (Le stacking comprend quel modèle est fort sur les matchs nuls vs victoires).*

# 🟠 PHASE 2 : Feature Engineering (La clé de la victoire)
*C'est ici que se joue la 1ère place. Les modèles sont les mêmes pour tout le monde, les données font la différence.*

- [ ] **Test A/B : V1 vs V2**
    - [x] Lancer un run avec **V1** (Agrégation globale Joueurs). Noter le CV Score.
    - [x] Lancer un run avec **V2** (Agrégation par Position FORWARD, DEFENDER...). Noter le CV Score.
    - [x] **Décision** : Si V2 est inférieur à V1, c'est qu'il y a trop de bruit -> Tester V2 + Feature Selection.
-> V2 un peu mieux que V1 sans features selection
- [ ] **Feature Selection "Intelligente"**
    - [x] Si V2 est retenu (~4000 colonnes), réactiver `SelectFromModel`.
    - [ ] **Ajustement** : Tester un seuil plus doux (`threshold="mean"` au lieu de `1.25*mean`) pour ne pas perdre d'info cruciale.
    - [ ] Analyse de l'importance des features (SHAP ou `feature_importances_`) pour identifier les variables inutiles.
    - [ ] Création de nouvelles features (Dynamique d'équipe, Stats avancées, Interactions Attaque/Défense).

## TO DO Engineering features 
### 🧱 NIVEAU 1 : Physique du Match (Attaque vs Défense)
    *Objectif : Confronter les forces opposées (Attaque A vs Défense B) plutôt que de simples soustractions.*
    - [ ] **Cross-Deltas :** Créer des variables croisées (ex: `HOME_Shots_Target` - `AWAY_Goalkeeper_Saves`).
    - [ ] **Ratios de Domination :** Utiliser des divisions pour capturer l'intensité (ex: `Possession_Ratio = Home / (Home + Away)`).

### 📊 NIVEAU 2 : Agrégation Intelligente ("Smart Aggregation")
    *Objectif : Mieux capturer les talents individuels et la structure d'équipe.*
    - [ ] **Facteur Star (Max vs Mean) :** Pour les attaquants, calculer le `MAX` et la `SUM` des buts/tirs (un buteur vedette pèse plus qu'une moyenne d'équipe).
    - [ ] **Solidité (Standard Deviation) :** Ajouter l'`écart-type` des notes/stats. Un écart-type faible = équipe homogène et solide.
    - [ ] **Filtrage par Poste :** Ne calculer les `Tackles` que pour les Défenseurs/Milieux, et les `Saves` que pour les Gardiens (réduction du bruit).

### 🎯 NIVEAU 3 : Métriques de Qualité (Efficacité)
    *Objectif : Donner au modèle la notion de réalisme (Conversion).*
    - [ ] **Conversion Rate (Offensif) :** `Goals / Shots`. Distinguer domination stérile et efficacité chirurgicale.
    - [ ] **Mur Défensif :** `Saves / Shots_Received`. Capacité du gardien à subir sans rompre.

### 🧹 NIVEAU 4 : Nettoyage & Sélection
    *Objectif : Réduire la dimensionnalité (>5000 colonnes) pour éviter l'overfitting.*
    - [ ] **Correlation Threshold :** Supprimer les features corrélées à > 0.95 (garder une seule des deux).
    - [ ] **Zero Variance :** Supprimer les colonnes remplies à >99% de 0.
### Niveau 5 : Autre target auxiliaire (Goal Diff & Total Goals)**
    - [x] Vérifier que la feature `PRED_GOAL_DIFF` (issue du CatBoostRegressor) est bien présente dans X_train et X_test.
    - [ ] **Idée Bonus** : Créer une 2ème target auxiliaire `PRED_TOTAL_GOALS` (Somme des buts) si `Y_train_supp` le permet, et l'ajouter en feature.




# 🟡 PHASE 3 : Optimisation Fine (Les derniers 0.5%)
*À faire uniquement quand les features sont figées (Phase 2 terminée).*

- [ ] **Tuning Optuna (Nuit de calcul)**
    - [ ] Lancer le script Optuna sur **LightGBM** (30-50 trials).
    - [ ] Lancer le script Optuna sur **XGBoost** (30-50 trials).
    - [ ] Mettre à jour les dictionnaires `params` avec les valeurs exactes trouvées.

- [ ] **Format de Soumission**
    - [x] S'assurer que le script de conversion en format strict (0 ou 1) utilise bien la logique `argmax` et non un seuil arbitraire.
