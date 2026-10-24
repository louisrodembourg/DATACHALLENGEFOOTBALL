# Validation et robustesse

Scripts : `scripts/ensemble_search.py`, `scripts/adversarial_validation.py`

Point de départ : les gains obtenus en CV (jusqu'à 0.4957) ne se retrouvent pas en ligne (0.487–0.491). Cette fiche explique pourquoi et ce qui en découle.

## 1. Fuite de données dans la feature auxiliaire
`PRED_GOAL_DIFF` était calculée en `cross_val_predict` sur tout le jeu d'entraînement, **avant** le découpage en folds : la ligne de validation d'un fold a donc servi à entraîner le régresseur qui produit la feature de ses voisines.
Correction : la feature est recalculée **dans chaque fold**, out-of-fold sur le train du fold et par un fit sur ce train pour la validation.
Effet mesuré : environ −0.002 d'accuracy. La fuite était réelle mais faible.

## 2. Ensemble validé en CV imbriquée
Une bibliothèque de 16 modèles (CatBoost, LightGBM et XGBoost avec 100 à 1 350 features et plusieurs seeds, régressions logistiques, modèle ordinal sur la différence de buts, TabICL) est combinée par **sélection gloutonne** (Caruana, sur la log-loss).
Choisir les poids sur les mêmes données que l'évaluation surestime le gain. Ils sont donc choisis sur 4 folds et évalués sur le 5ᵉ :

| | Accuracy (CV imbriquée, sans fuite) |
|---|---|
| CatBoost 250 features | 0.4918 |
| Ensemble glouton | **0.4945** (+0.0027, meilleur sur 4 folds sur 5) |

En ligne : 0.4873. Le gain ne se transfère pas.

## 3. Décalage entre train et test (validation adversariale)
Un LightGBM entraîné à distinguer les matchs du train de ceux du test y parvient avec une **AUC de 0.98**.
- Ce sont surtout les statistiques des joueurs qui ont changé (AUC 0.97, contre 0.89 pour les équipes). Par exemple, les arrêts au poing des gardiens ont un KS de 0.36.
- La moitié des features a un KS supérieur à 0.10.

Une **validation proxy** est construite avec les 25 % de matchs du train qui ressemblent le plus au test, puis plusieurs jeux de features y sont comparés :

| Features | Accuracy proxy |
|---|---|
| Toutes (top 250) | 0.4945 |
| Toutes sauf les plus décalées (KS < 0.10) | 0.4971 |
| **Équipes seules** (top 250) | **0.4990** |

En ligne, la version « équipes seules » obtient 0.4888, en dessous du stacking complet (0.4913).

## Conclusion
- Les cinq soumissions tiennent dans 0.004, ce qui est comparable à l'incertitude du score en ligne (±0.003 sur 25 000 matchs). Elles sont **statistiquement indiscernables**.
- Ni la CV (biaisée par le décalage) ni le classement (trop bruité) ne permettent plus de départager des variantes à ce niveau.
- Le **stacking GBDT reste la soumission finale**. Continuer à soumettre des variantes reviendrait à surapprendre le classement.
