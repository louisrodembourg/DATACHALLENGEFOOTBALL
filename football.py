import pandas as pd
import numpy as np
import os
import re
import glob
import json
import datetime
import warnings
from typing import cast

# ML Imports
import lightgbm as lgb
import xgboost as xgb
import optuna
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel

# Configuration
warnings.filterwarnings('ignore')

# =============================================================================
# 1. UTILITAIRES & LOGGING (Ta demande spécifique)
# =============================================================================

def save_experiment(cv_score, params_dict, description, submission_df=None, folder='experiments'):
    """
    Sauvegarde la config de l'expérience (JSON).
    Si un submission_df est fourni, sauvegarde aussi le CSV.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"cv{cv_score:.4f}_{timestamp}"
    
    # 1. Sauvegarde de la Config (JSON) - TOUJOURS
    config_filename = f"{folder}/conf_{base_name}.json"
    log_data = {
        "timestamp": timestamp,
        "cv_score": cv_score,
        "description": description,
        "parameters": params_dict,
        "status": "ABORTED" if submission_df is None else "COMPLETED"
    }
    
    with open(config_filename, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    print(f"\n[LOG] Config sauvegardée : {config_filename}")

    # 2. Sauvegarde du CSV (Seulement si le run est validé)
    if submission_df is not None:
        # Sauvegarde version brute (experiments)
        csv_filename = f"{folder}/sub_{base_name}.csv"
        submission_df.to_csv(csv_filename, index=False)
        print(f"[LOG] CSV Expérience sauvegardé : {csv_filename}")
        
        # Sauvegarde version "Submission" incrémentale (pour upload facile)
        os.makedirs('submission', exist_ok=True)
        existing_files = glob.glob('submission/submission_V*.csv')
        version = 1
        if existing_files:
            versions = []
            for f in existing_files:
                match = re.search(r'submission_V(\d+)\.csv', f)
                if match:
                    versions.append(int(match.group(1)))
            if versions:
                version = max(versions) + 1
        
        final_filename = f'submission/submission_V{version}.csv'
        submission_df.to_csv(final_filename, index=False)
        print(f"🚀 PRÊT À SOUMETTRE : '{final_filename}'")

# =============================================================================
# 2. CHARGEMENT DES DONNÉES
# =============================================================================

def load_data(base_path='data/'):
    print("--- Chargement des données ---")
    # Train
    x_train_team_home = pd.read_csv(f'{base_path}Train_Data/train_home_team_statistics_df.csv')
    x_train_team_away = pd.read_csv(f'{base_path}Train_Data/train_away_team_statistics_df.csv')
    x_train_player_home = pd.read_csv(f'{base_path}Train_Data/train_home_player_statistics_df.csv')
    x_train_player_away = pd.read_csv(f'{base_path}Train_Data/train_away_player_statistics_df.csv')
    y_train = pd.read_csv(f'{base_path}Y_train.csv')
    y_train_supp = pd.read_csv(f'{base_path}benchmark_and_extras/Y_train_supp.csv')
    
    # Test
    x_test_team_home = pd.read_csv(f'{base_path}Test_Data/test_home_team_statistics_df.csv')
    x_test_team_away = pd.read_csv(f'{base_path}Test_Data/test_away_team_statistics_df.csv')
    x_test_player_home = pd.read_csv(f'{base_path}Test_Data/test_home_player_statistics_df.csv')
    x_test_player_away = pd.read_csv(f'{base_path}Test_Data/test_away_player_statistics_df.csv')
    
    return (x_train_team_home, x_train_team_away, x_train_player_home, x_train_player_away, y_train, y_train_supp,
            x_test_team_home, x_test_team_away, x_test_player_home, x_test_player_away)

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def aggregate_players_by_position(player_df, prefix):
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' not in numeric_cols: numeric_cols.append('ID')
    
    cols_to_use = numeric_cols + ['POSITION']
    pivot_df = player_df[cols_to_use].groupby(['ID', 'POSITION']).agg(['mean', 'sum'])
    pivot_df.columns = [f'{c[0]}_{c[1]}' for c in pivot_df.columns]
    flat_df = pivot_df.unstack(level='POSITION')
    flat_df.columns = [f'{prefix}_{pos}_{col}' for col, pos in flat_df.columns]
    flat_df.reset_index(inplace=True)
    flat_df.fillna(0, inplace=True)
    return flat_df

def build_features_v2(team_home, team_away, player_home, player_away):
    print("--- Construction des features V2 (Position-Aware) ---")
    p_home_agg = aggregate_players_by_position(player_home, 'P_HOME')
    p_away_agg = aggregate_players_by_position(player_away, 'P_AWAY')
    
    df = team_home.merge(team_away, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_features = set([c.replace('_HOME', '') for c in numeric_cols if '_HOME' in c])
    
    # print(f"Création des deltas sur {len(base_features)} variables...")
    for col in base_features:
        col_h = f"{col}_HOME"
        col_a = f"{col}_AWAY"
        if col_h in df.columns and col_a in df.columns:
            df[f'DELTA_{col}'] = df[col_h] - df[col_a]
    
    drop_cols = ['LEAGUE_HOME', 'TEAM_NAME_HOME', 'LEAGUE_AWAY', 'TEAM_NAME_AWAY', 'LEAGUE', 'TEAM_NAME']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.fillna(0, inplace=True)
    return df

# =============================================================================
# 4. FONCTIONS OPTIMISATION (Stockées pour usage futur)
# =============================================================================

def objective_lgb(trial, X, y):
    # Fonction prête à l'emploi si besoin de relancer Optuna
    param = {
        'objective': 'multiclass', 'metric': 'multi_logloss', 'verbosity': -1,
        'boosting_type': 'gbdt', 'num_class': 3, 'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'n_jobs': -1
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    estimator = cast(BaseEstimator, lgb.LGBMClassifier(**param))

    scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# =============================================================================
# 5. PIPELINE D'EXÉCUTION PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    
    # --- A. Chargement ---
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, y_supp, 
     xtest_h, xtest_a, xpt_h, xpt_a) = load_data()

    # --- B. Construction Features ---
    X_train = build_features_v2(xt_h, xt_a, xp_h, xp_a)
    X_test = build_features_v2(xtest_h, xtest_a, xpt_h, xpt_a)
    
    # Alignement et nettoyage ID
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)
    ID_test = X_test['ID']
    X_train = X_train.drop(columns=['ID'])
    X_test = X_test.drop(columns=['ID'])

    # Targets
    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y_train_cls = le.fit_transform(y_classes) # 0:AWAY, 1:DRAW, 2:HOME
    y_train_reg = y_supp['GOAL_DIFF_HOME_AWAY']

    print(f"Shape finale Train: {X_train.shape}")

    # --- C. Target Auxiliaire (Goal Diff) ---
    print("--- Ajout Feature Auxiliaire (Goal Diff) ---")
    regressor = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    X_train['PRED_GOAL_DIFF'] = cross_val_predict(cast(BaseEstimator, regressor), X_train, y_train_reg, cv=5, n_jobs=-1)
    
    regressor.fit(X_train.drop(columns=['PRED_GOAL_DIFF']), y_train_reg)
    X_test['PRED_GOAL_DIFF'] = regressor.predict(X_test)

    # --- D. Feature Selection (Optionnel - Désactivé par défaut) ---
    if False: # Mettre True pour activer
        print("Démarrage de la sélection des features...")
        selector = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1, verbose=-1)
        selector.fit(X_train, y_train_cls)
        model_selector = SelectFromModel(selector, prefit=True, threshold="1.25*mean")
        X_train_selected = model_selector.transform(X_train)
        X_test_selected = model_selector.transform(X_test)
        # Remise en DataFrame... (Code gardé en réserve)

    # --- E. Configuration des Modèles ---
    # Si tu veux relancer Optuna, décommente les lignes ci-dessous :
    # study = optuna.create_study(direction='maximize')
    # study.optimize(lambda trial: objective_lgb(trial, X_train, y_train_cls), n_trials=30)
    
    params_lgb = {
        'n_estimators': 2000, 'learning_rate': 0.03, 'num_leaves': 20, 
        'colsample_bytree': 0.6, 'subsample': 0.8, 'random_state': 42,
        'n_jobs': -1, 'verbose': -1
    }
    params_xgb = {
        'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 5, 
        'colsample_bytree': 0.8, 'subsample': 0.8, 'random_state': 42,
        'eval_metric': 'mlogloss', 'tree_method': 'hist', 'n_jobs': -1
    }
    params_cat = {
        'iterations': 2000, 'learning_rate': 0.03, 'depth': 6, 
        'rsm': 0.6, 'verbose': 0, 'random_state': 42, 'thread_count': -1
    }
    
    # Description pour le log
    run_description = "V2 Features - Full (No Selection) - Soft Voting - Params Anti-Overfit"
    experiment_params = {
        "lgb": params_lgb, "xgb": params_xgb, "cat": params_cat,
        "desc": run_description
    }

    # --- F. Validation Croisée (CRITIQUE) ---
    print("\n--- ÉTAPE 1 : Validation Croisée (Estimation du Score) ---")
    
    clf1 = lgb.LGBMClassifier(**params_lgb)
    clf2 = xgb.XGBClassifier(**params_xgb)
    clf3 = CatBoostClassifier(**params_cat)

    eclf = VotingClassifier(
        estimators=[
            ('lgb', cast(BaseEstimator, clf1)), 
            ('xgb', cast(BaseEstimator, clf2)), 
            ('cat', cast(BaseEstimator, clf3))
        ],
        voting='soft', verbose=True, n_jobs=1 
    )

    # Calcul du score
    cv_scores = cross_val_score(eclf, X_train, y_train_cls, cv=5, scoring='accuracy', n_jobs=1)
    mon_score_estime = cv_scores.mean()
    
    print(f"\n📊 SCORES CV: {cv_scores}")
    print(f"🏆 MOYENNE : {mon_score_estime:.5f} (+/- {cv_scores.std():.5f})")

    # --- G. Décision et Sauvegarde ---
    
    # Seuil pour passer en prod (à ajuster selon tes ambitions)
    THRESHOLD_SCORE = 0.4815 

    if mon_score_estime < THRESHOLD_SCORE:
        print(f"\n❌ Score insuffisant (< {THRESHOLD_SCORE}). Arrêt du script.")
        # ON SAUVEGARDE QUAND MÊME LA TRACE (JSON uniquement)
        save_experiment(
            cv_score=mon_score_estime, 
            params_dict=experiment_params, 
            description=run_description,
            submission_df=None # Pas de CSV généré
        )
        exit() # Fin du script
    else:
        print(f"\n✅ Score validé ! Lancement de l'entraînement final.")

    # --- H. Entraînement Final & Prédictions ---
    print("--- ÉTAPE 2 : Entraînement Final (100% Data) ---")
    eclf.fit(X_train, y_train_cls)
    probs = eclf.predict_proba(X_test)

    # --- I. Formatage Soumission ---
    submission = pd.DataFrame({
        'ID': ID_test,
        'HOME_WINS': probs[:, 2],
        'DRAW': probs[:, 1],
        'AWAY_WINS': probs[:, 0]
    })
    
    # Conversion Binaire (Strict 0/1)
    print("Conversion binaire...")
    cols_submit = ['HOME_WINS', 'DRAW', 'AWAY_WINS']
    max_indices = submission[cols_submit].values.argmax(axis=1)
    hard_preds = np.zeros(submission[cols_submit].shape, dtype=int)
    hard_preds[np.arange(len(submission)), max_indices] = 1
    submission[cols_submit] = hard_preds

    # --- J. Sauvegarde Finale (CSV + JSON) ---
    save_experiment(
        cv_score=mon_score_estime, 
        params_dict=experiment_params, 
        description=run_description,
        submission_df=submission # Le CSV sera généré
    )