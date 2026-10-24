# Football Match Prediction ⚽

Prédire l'issue d'un match (`HOME_WINS` / `DRAW` / `AWAY_WINS`) à partir de statistiques agrégées d'équipes et de joueurs. Data Challenge **QRT « Football : Qui va gagner ? »**.

Approche **data-centric** : l'effort porte sur les features et la rigueur de validation, pas sur la complexité des modèles.

## Résultats

| Méthode | Validation | Score en ligne |
|---|---|---|
| **Stacking LightGBM + XGBoost + CatBoost** (retenu) | 0.4935 – 0.4956 (CV) | **0.4913** |
| CatBoost features équipe seules (robuste au shift) | 0.4990 (validation proxy) | 0.4888 |
| Blend 50 % TabICL + 50 % CatBoost | 0.4947 (CV) | 0.4880 |
| Ensemble glouton (CatBoost + TabICL + LogReg) | 0.4957 (CV imbriquée) | 0.4873 |
| TabICL seul (foundation model tabulaire) | 0.4952 (CV) | — |

Métrique : accuracy. Le problème est très bruité : 0.49–0.50 correspond au haut du classement.

**Conclusion.** Tous les scores en ligne tiennent dans 0.004, soit l'ordre de grandeur de l'incertitude du classement (±0.003 sur 25 000 matchs). Les variantes sont donc statistiquement équivalentes. Le stacking est conservé comme soumission finale, et j'ai arrêté d'optimiser sur le classement pour ne pas surapprendre le leaderboard.

## Méthodes

1. **Pipeline GBDT** : feature engineering métier (environ 2 800 variables), sélection dans chaque fold, feature auxiliaire « différence de buts prédite », stacking de boosters.
   → [docs/methodology_fr.md](docs/methodology_fr.md)
2. **Foundation model tabulaire (TabICL)** : transformer pré-entraîné qui prédit par *in-context learning*, sans entraînement ni tuning, comparé au GBDT sur les mêmes folds.
   → [docs/tabular_foundation_model.md](docs/tabular_foundation_model.md)
3. **Validation et robustesse** : correction d'une fuite, ensemble validé en CV imbriquée, diagnostic du décalage train/test.
   → [docs/validation_robustesse.md](docs/validation_robustesse.md)

**Constats clés**
- Un modèle pré-entraîné sans réglage atteint le niveau de la pipeline optimisée, et les deux sont d'accord sur 92 % des prédictions : le plafond vient des données, pas du modèle.
- Train et test suivent des distributions différentes : un classifieur les distingue avec une AUC de 0.98. C'est pour cette raison que les gains de CV ne se retrouvent pas en ligne.

## Utilisation

```bash
pip install -r requirements.txt

python scripts/run_pipeline.py --model stack               # Pipeline GBDT -> submission/
python scripts/test_tabular_fm.py --device cpu             # Benchmark TabICL vs CatBoost (CV)
python scripts/ensemble_search.py --leakfree --submit      # Ensemble glouton validé en CV imbriquée
python scripts/adversarial_validation.py                   # Shift train/test + validation proxy
```

Les données QRT sont à placer dans `data/` (non versionnées).

## Structure

```
src/          data.py (chargement) · features.py (feature engineering) · utils.py (logging des runs)
scripts/      run_pipeline.py · test_tabular_fm.py · ensemble_search.py · adversarial_validation.py
              optimize_*.py (Optuna : features, hyperparamètres, seuils)
docs/         une fiche par méthode
experiments/  config JSON + score + logs de chaque run
```
