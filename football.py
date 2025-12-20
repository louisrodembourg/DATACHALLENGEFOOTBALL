import pandas as pd
import numpy as np
import os
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================

def load_data(base_path='data/'):
    # Chemins à adapter selon vos noms de fichiers réels
    print("Chargement des données...")
    
    # Train
    x_train_team_home = pd.read_csv(f'{base_path}Train_Data/train_home_team_statistics_df.csv')
    x_train_team_away = pd.read_csv(f'{base_path}Train_Data/train_away_team_statistics_df.csv')
    x_train_player_home = pd.read_csv(f'{base_path}Train_Data/train_home_player_statistics_df.csv')
    x_train_player_away = pd.read_csv(f'{base_path}Train_Data/train_away_player_statistics_df.csv')
    y_train = pd.read_csv(f'{base_path}Y_train.csv')
    y_train_supp = pd.read_csv(f'{base_path}benchmark_and_extras/Y_train_supp.csv') # Goal diff
    
    # Test
    x_test_team_home = pd.read_csv(f'{base_path}Test_Data/test_home_team_statistics_df.csv')
    x_test_team_away = pd.read_csv(f'{base_path}Test_Data/test_away_team_statistics_df.csv')
    x_test_player_home = pd.read_csv(f'{base_path}Test_Data/test_home_player_statistics_df.csv')
    x_test_player_away = pd.read_csv(f'{base_path}Test_Data/test_away_player_statistics_df.csv')
    
    return (x_train_team_home, x_train_team_away, x_train_player_home, x_train_player_away, y_train, y_train_supp,
            x_test_team_home, x_test_team_away, x_test_player_home, x_test_player_away)

# =============================================================================
# 2. FEATURE ENGINEERING (LE CŒUR DU SYSTÈME)
# =============================================================================

def aggregate_players_by_position(player_df, prefix):
    """
    Agrège les stats en séparant par POSITION (G, D, M, F).
    Cela capture la structure tactique de l'équipe.
    """
    # 1. On sépare les colonnes numériques et la position
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' not in numeric_cols: numeric_cols.append('ID')
    
    # On garde ID et POSITION pour le pivot
    cols_to_use = numeric_cols + ['POSITION']
    
    # 2. Pivot Table : C'est la magie. 
    # On groupe par ID et POSITION, puis on calcule la moyenne (mean) et la somme (sum)
    # On évite l'écart-type (std) ici pour ne pas exploser le nombre de colonnes (déjà x4 à cause des positions)
    pivot_df = player_df[cols_to_use].groupby(['ID', 'POSITION']).agg(['mean', 'sum'])
    
    # 3. Aplatir le MultiIndex (ex: ('GOALS', 'mean') -> 'GOALS_mean')
    pivot_df.columns = [f'{c[0]}_{c[1]}' for c in pivot_df.columns]
    
    # 4. Dé-empiler les positions (ex: une ligne par match, colonnes: DEFENDER_GOALS_mean, FORWARD_GOALS_mean...)
    flat_df = pivot_df.unstack(level='POSITION')
    
    # 5. Aplatir à nouveau les noms de colonnes
    flat_df.columns = [f'{prefix}_{pos}_{col}' for col, pos in flat_df.columns]
    flat_df.reset_index(inplace=True)
    
    # 6. Gestion des valeurs manquantes (ex: si une équipe joue sans attaquants... improbable mais possible en data sale)
    flat_df.fillna(0, inplace=True)
    
    return flat_df

def build_features_v2(team_home, team_away, player_home, player_away):
    print("Construction des features V2 (Position-Aware)...")
    
    # 1. Agrégation fine par position
    p_home_agg = aggregate_players_by_position(player_home, 'P_HOME')
    p_away_agg = aggregate_players_by_position(player_away, 'P_AWAY')
    
    # 2. Merge Team + Players
    df = team_home.merge(team_away, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    
    # 3. Delta Features (Uniquement sur les colonnes les plus importantes pour ne pas exploser la RAM)
    # On se concentre sur les totaux d'équipe et les moyennes par position
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # On cherche les paires HOME/AWAY
    base_features = set([c.replace('_HOME', '') for c in numeric_cols if '_HOME' in c])
    
    print(f"Création des deltas sur {len(base_features)} variables...")
    for col in base_features:
        col_h = f"{col}_HOME"
        col_a = f"{col}_AWAY"
        if col_h in df.columns and col_a in df.columns:
            df[f'DELTA_{col}'] = df[col_h] - df[col_a]
    
    # 4. Nettoyage
    drop_cols = ['LEAGUE_HOME', 'TEAM_NAME_HOME', 'LEAGUE_AWAY', 'TEAM_NAME_AWAY', 
                 'LEAGUE', 'TEAM_NAME']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.fillna(0, inplace=True)
    
    return df
# =============================================================================
# 3. OPTIMISATION HYPERPARAMÈTRES (OPTUNA)
# =============================================================================

def objective_lgb(trial, X, y):
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_class': 3,
        'n_estimators': 1000, # Fixed high number with early stopping usually
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42
    }
    
    # 3-Fold CV pour aller vite lors de l'optimisation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    estimator = cast(BaseEstimator, lgb.LGBMClassifier(**param))
    scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# =============================================================================
# 4. PIPELINE PRINCIPAL
# =============================================================================

# --- A. Chargement ---
(xt_h, xt_a, xp_h, xp_a, y_train_raw, y_supp, 
 xtest_h, xtest_a, xpt_h, xpt_a) = load_data()

# --- B. Construction du Dataset ---
X_train = build_features_v2(xt_h, xt_a, xp_h, xp_a)
X_test = build_features_v2(xtest_h, xtest_a, xpt_h, xpt_a)

# Alignement des colonnes (au cas où certaines manquent dans le test)
X_train, X_test = X_train.align(X_test, join='inner', axis=1)

# Préparation de la Target (Classification)
# Conversion de [HOME_WINS, DRAW, AWAY_WINS] en [2, 1, 0]
# Attention l'ordre des colonnes est important. 
# Si Y_train colonnes sont: ID, HOME_WINS, DRAW, AWAY_WINS
# idxmax va donner le nom de la colonne.
y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
le = LabelEncoder()
y_train_cls = le.fit_transform(y_classes) # Mappe: AWAY_WINS->0, DRAW->1, HOME_WINS->2

# Préparation de la Target (Régression Auxiliaire - Différence de buts)
y_train_reg = y_supp['GOAL_DIFF_HOME_AWAY']

ID_test = X_test['ID'] # Sauvegarde pour la soumission
X_train = X_train.drop(columns=['ID'])
X_test = X_test.drop(columns=['ID'])

print(f"Shape finale Train: {X_train.shape}")

# --- C. ÉTAPE CLÉ : Prédiction Auxiliaire (Goal Diff) ---
# On entraîne un modèle pour prédire la différence de buts, et on utilise 
# cette prédiction comme une feature très forte pour savoir qui gagne.
print("Entraînement du modèle auxiliaire (Goal Difference)...")

regressor = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)

# Pour éviter le data leakage, on devrait faire cela en Out-of-Fold (OOF) sur le train
# mais pour simplifier ici, on entraîne sur tout le train (risque léger d'overfit)
# Idéalement : utiliser cross_val_predict pour X_train
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
from typing import cast

# Feature 'PRED_GOAL_DIFF' pour le Train (via Cross Validation pour éviter fuite)
X_train['PRED_GOAL_DIFF'] = cross_val_predict(cast(BaseEstimator, regressor), X_train, y_train_reg, cv=5, n_jobs=-1)

# Feature 'PRED_GOAL_DIFF' pour le Test (via entraînement complet)
regressor.fit(X_train.drop(columns=['PRED_GOAL_DIFF']), y_train_reg)
X_test['PRED_GOAL_DIFF'] = regressor.predict(X_test)

# --- E. Optimisation (Optionnelle - Commentée pour exécution rapide) ---
# study = optuna.create_study(direction='maximize')
# study.optimize(lambda trial: objective_lgb(trial, X_train, y_train_cls), n_trials=50)
# best_params_lgb = study.best_params
# print("Best Params:", best_params_lgb)

# Paramètres solides par défaut (si on saute Optuna)
params_lgb = {
    'n_estimators': 2000, 'learning_rate': 0.03, 'num_leaves': 20, 
    'colsample_bytree': 0.6, 'subsample': 0.8, 'random_state': 42,
    'n_jobs' :-1,'verbose': -1
}
params_xgb = {
    'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 5, 
    'colsample_bytree': 0.8, 'subsample': 0.8, 'random_state': 42,
    'eval_metric': 'mlogloss','tree_method':'hist','n_jobs': -1
}
params_cat = {
    'iterations': 2000, 'learning_rate': 0.03, 'depth': 6, 
    'verbose': 0, 'random_state': 42,'thread_count': -1,
    'rsm':0.6
}

# --- E. Modélisation Finale (Ensemble Voting) ---
print("Entraînement de l'ensemble final...")

clf1 = lgb.LGBMClassifier(**params_lgb)
clf2 = xgb.XGBClassifier(**params_xgb)
clf3 = CatBoostClassifier(**params_cat)

eclf = VotingClassifier(
    estimators=[
        ('lgb', cast(BaseEstimator, clf1)), 
        ('xgb', cast(BaseEstimator, clf2)), 
        ('cat', cast(BaseEstimator, clf3))
    ],
    voting='soft',
    verbose=True,
    n_jobs=1
)

# Validation Score final
#cv_scores = cross_val_score(eclf, X_train, y_train_cls, cv=5, scoring='accuracy')
#print(f"Accuracy CV Moyenne : {cv_scores.mean():.5f} (+/- {cv_scores.std():.5f})")

# Entraînement sur tout le dataset
eclf.fit(X_train, y_train_cls)

# --- F. Prédiction et Soumission ---
print("Génération des prédictions...")
probs = eclf.predict_proba(X_test)

# Reconstruction du format de soumission (HOME_WINS, DRAW, AWAY_WINS)
# Rappel mapping LE: 0=AWAY, 1=DRAW, 2=HOME
submission = pd.DataFrame({
    'ID': ID_test,
    'HOME_WINS': probs[:, 2], # Proba classe 2
    'DRAW': probs[:, 1],      # Proba classe 1
    'AWAY_WINS': probs[:, 0]  # Proba classe 0
})

# Vérification (la somme doit faire 1)
# submission[['HOME_WINS', 'DRAW', 'AWAY_WINS']].sum(axis=1)

# Création du dossier de soumission s'il n'existe pas
os.makedirs('submission', exist_ok=True)
submission_path = os.path.join('submission', 'submission_qrt_optimized.csv')

submission.to_csv(submission_path, index=False)
print(f"Fichier '{submission_path}' généré avec succès.")

