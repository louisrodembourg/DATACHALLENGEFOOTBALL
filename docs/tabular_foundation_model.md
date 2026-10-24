# Méthode 2 : Foundation model tabulaire (TabICL)

Script : `scripts/test_tabular_fm.py`

## Pourquoi un modèle tabulaire, et pas Chronos ?
Les foundation models de séries temporelles (Chronos-2, TimesFM…) prolongent un historique ordonné. Or ici chaque match est un instantané déjà agrégé, sans date ni nom d'équipe dans le test : il n'y a aucune série à prolonger.
Pour ce type de données, l'équivalent est un **foundation model tabulaire**. [TabICL](https://github.com/soda-inria/tabicl) est un transformer pré-entraîné sur des millions de tableaux synthétiques. Il reçoit le jeu d'entraînement comme contexte et prédit en une passe, par *in-context learning* : pas d'entraînement, pas de tuning.

## Protocole
Pour une comparaison équitable, tout est identique à la pipeline GBDT : mêmes features, même feature auxiliaire, mêmes 5 folds, même sélection LightGBM faite dans chaque fold.
- **CatBoost** (référence) utilise les 250 meilleures features.
- **TabICL** utilise les 100 meilleures du même classement (sa mémoire croît avec le nombre de lignes × colonnes).
- Les probabilités out-of-fold sont sauvegardées, ce qui permet de tester des blends sans réentraîner.

## Résultats (CV 5 folds)

| Modèle | Accuracy | Log-loss |
|---|---|---|
| CatBoost | 0.4935 ± 0.009 | 1.0167 |
| TabICL | **0.4952** ± 0.007 | 1.0165 |
| Blend 50/50 | 0.4947 ± 0.009 | **1.0149** |

- **Sans aucun réglage, TabICL égale la pipeline optimisée.** L'écart (+0.002) reste dans le bruit d'un fold à l'autre.
- **Le blend apporte peu** : les deux modèles prédisent la même classe dans 92 % des cas. Ils exploitent le même signal.
- **Pondérer les classes** pour forcer des nuls a été testé en CV imbriquée : poids instables d'un fold à l'autre et pas de gain. Ce n'est donc pas retenu.

**Soumission** : blend 50/50 TabICL + CatBoost (meilleure log-loss, variance réduite). Score en ligne : **0.4880**, en dessous du stacking GBDT (0.4913) mais dans la marge d'incertitude du classement. Le gain de CV ne se retrouve pas en ligne, pour une raison expliquée dans [validation_robustesse.md](validation_robustesse.md) : train et test suivent des distributions différentes.

## Contraintes techniques (Mac M-series, 18 Go)
- **GPU (MPS) inutilisable** : la mémoire est partagée avec le système et TabICL la surestime, ce qui provoque du swap ou des erreurs « out of memory ». L'inférence se fait donc sur CPU, en environ 3 minutes par fold et avec un pic d'environ 10 Go.
- **torch doit être importé avant LightGBM/CatBoost** : dans l'ordre inverse, les deux runtimes OpenMP se bloquent mutuellement.
- **Le test est prédit par paquets de 2 500 lignes** pour tenir en mémoire. La prédiction des 25 368 matchs a tout de même pris environ 8 h sur CPU, contre 6 s pour CatBoost : c'est le principal coût de la méthode.
- **Les features sont calculées une seule fois et mises en cache** dans `cache/`, ce qui évite environ 10 minutes de préparation à chaque run.
