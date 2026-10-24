# Méthode 1 : Pipeline GBDT

Script : `scripts/run_pipeline.py`. Code des features : `src/features.py`.

## Données
Une ligne = un match. Les statistiques d'équipes et de joueurs sont déjà agrégées (saison et 5 derniers matchs : somme, moyenne, écart-type), normalisées sur une échelle de 0 à 10, sans date ni nom d'équipe dans le test.
La difficulté : les joueurs sont en nombre variable par match. Il faut les résumer dans un format tabulaire fixe.

## 1. Feature engineering
- **Agrégation par poste** : les joueurs sont regroupés par ligne (gardiens, défenseurs, milieux, attaquants), avec somme et moyenne par ligne.
- **Deltas domicile – extérieur** : calculés pour chaque statistique.
- **Ratios d'efficacité** : conversion (buts / tirs), précision des tirs, part des attaques dangereuses. Ils corrigent le biais de volume.
- **Duels métier** : par exemple `DUEL_ATT_H_GK_A`, tirs des attaquants à domicile contre arrêts du gardien adverse. On y ajoute la dominance physique (duels) et la créativité (passes clés).
- **Momentum** : forme récente (5 derniers matchs) rapportée à la moyenne de la saison.

Au total, environ 4 500 variables, ramenées à environ 2 800 après suppression des colonnes corrélées à plus de 0.95.

## 2. Sélection sans fuite de données
Un LightGBM rapide classe les features par gain et ne garde que les **250 meilleures**. Cette sélection est faite **dans chaque fold de CV** (pipeline sklearn) : la faire sur tout le jeu d'entraînement gonflait artificiellement le score de CV.

## 3. Feature auxiliaire
Un CatBoost Regressor prédit la **différence de buts**, une cible fournie en plus. Il est entraîné en `cross_val_predict`, et sa prédiction `PRED_GOAL_DIFF` devient une feature dense du modèle principal.
Comme ce calcul est fait avant le découpage en folds, il reste une petite fuite dans la CV. Recalculer la feature dans chaque fold coûte environ 0.002 d'accuracy ([détail](validation_robustesse.md)).

## 4. Modèle
**Stacking** de LightGBM, XGBoost et CatBoost, combinés par une régression logistique régularisée. Les hyperparamètres sont volontairement contraints (faible profondeur, forte régularisation L1/L2) pour réduire la variance.

## 5. Validation
`StratifiedKFold` 5 folds (seed 42). Un run n'est soumis que si la CV dépasse un seuil.
Tentatives abandonnées après comparaison avec le score en ligne : clustering de joueurs, MLP et transformer maison.

**Résultat** : CV entre 0.4935 et 0.4956, **0.4913 en ligne**. C'est le meilleur score en ligne du projet et la soumission finale.
