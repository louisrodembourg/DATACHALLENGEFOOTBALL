import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
import warnings
import sys
from sklearn.base import BaseEstimator
from typing import cast

import sys
import os

# Proper path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Gestion des imports
try:
    from src.data import load_data
    from src.features import build_features
except ImportError:
    print("❌ Impossible d'importer load_data/build_features depuis src/")
    sys.exit()

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# =============================================================================
# 🧬 DATA PREP (Aligné football.py)
# =============================================================================

def prepare_training_matrix(
    use_corr_removal: bool = True,
    corr_threshold: float = 0.95,
    use_topk: bool = True,
    topk_features: int = 800,
    add_pred_goal_diff: bool = True,
    random_state: int = 42,
):
    print("--- 1. Chargement et Feature Construction (Train + Test pour Alignement) ---")
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, y_supp, 
     xtest_h, xtest_a, xpt_h, xpt_a) = load_data()

    # Construction via football_clean.build_features
    X = build_features(xt_h, xt_a, xp_h, xp_a)
    X_test = build_features(xtest_h, xtest_a, xpt_h, xpt_a)

    # Target classification
    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y = le.fit_transform(y_classes)

    # 2. Nettoyage Corrélation
    if use_corr_removal:
        print("Correlation Analysis (>0.95)...")
        features_for_corr = X.drop(columns=['ID'], errors='ignore').select_dtypes(include=[np.number])
        corr_matrix = features_for_corr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        X.drop(columns=to_drop, errors='ignore', inplace=True)
        X_test.drop(columns=to_drop, errors='ignore', inplace=True)
        print(f"📉 Removing {len(to_drop)} correlated columns")

    # 3. Alignement (CRUCIAL pour matcher football_clean)
    print("Alignement Train/Test...")
    X, X_test = X.align(X_test, join='inner', axis=1)
    
    # Retrait ID après alignement
    X = X.drop(columns=['ID'], errors='ignore')
    
    # 4. Top-K Selection (DÉSACTIVÉ ICI POUR ÉVITER LEAKAGE, DÉPLACÉ DANS LA CV)
    #if use_topk:
    #    print(f"--- Sélection Top-{topk_features} features (LightGBM gain) ---")
        # LEAKAGE ALERT: Fitting on whole X, y makes CV scores overly optimistic!
        # Moved to Pipeline inside cross_val_score
    #    pass
    
    # 5. PRED_GOAL_DIFF
    if add_pred_goal_diff:
        print("--- Ajout Feature Auxiliaire (PRED_GOAL_DIFF) ---")
        y_train_reg = y_supp['GOAL_DIFF_HOME_AWAY']
        regressor = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_state=random_state,
        )
        X = X.copy()
        X['PRED_GOAL_DIFF'] = cross_val_predict(
            cast(BaseEstimator, regressor),
            X,
            y_train_reg,
            cv=5,
            n_jobs=-1,
        )

    return X, y

# =============================================================================
# 🎯 OPTIMISATION LOG LOSS (Meilleur pour les probas du Stacking)
# =============================================================================

def get_selector(random_state=42):
    """
    Returns a Feature Selector based on LightGBM gain.
    Included in Pipeline to prevent Data Leakage.
    """
    lgb_selector = lgb.LGBMClassifier(
        n_estimators=100, # Lightweight for selection
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
        importance_type='gain'
    )
    return SelectFromModel(estimator=lgb_selector, max_features=800, threshold=-np.inf) # Take top 800

def optimize_lightgbm(trial, X, y):
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = lgb.LGBMClassifier(**params, n_jobs=1)
    
    # Pipeline: Selection (Train only) -> Model
    pipeline = make_pipeline(get_selector(), model)
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_log_loss')
    return scores.mean()

def optimize_xgboost(trial, X, y):
    params = {
        'objective': 'multi:softprob', # Optimise les probas directement
        'eval_metric': 'mlogloss',
        'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        # XGB n'a pas class_weight='balanced' direct, on peut le gérer autrement ou laisser faire
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = xgb.XGBClassifier(**params, n_jobs=1)
    
    # Pipeline: Selection (Train only) -> Model
    pipeline = make_pipeline(get_selector(), model)
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_log_loss')
    return scores.mean()

def optimize_catboost(trial, X, y):
    params = {
        'loss_function': 'MultiClass', # LogLoss par défaut
        'iterations': trial.suggest_int('iterations', 300, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bootstrap_type': 'Bernoulli',  # <--- OBLIGATOIRE pour utiliser subsample
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0, 5),
        'verbose': False,
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', [None, 'Balanced']),
        'allow_writing_files': False # Évite de créer des dossiers catboost_info partout
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = CatBoostClassifier(**params, thread_count=1)
    
    # Pipeline: Selection (Train only) -> Model
    pipeline = make_pipeline(get_selector(), model)
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_log_loss')
    return scores.mean()

# =============================================================================
# 🚀 MAIN RUN
# =============================================================================

def run_optimization(name, optimize_func, n_trials, X, y):
    print(f"\n🚀 Optimisation {name} en cours...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_func(trial, X, y), n_trials=n_trials)
    print(f"\n✅ {name} TERMINÉ !")
    print(f"🏆 Meilleur Score (neg_log_loss) : {study.best_value:.5f}")
    print("📝 Meilleurs Paramètres :")
    print(study.best_params)
    return study.best_params

if __name__ == "__main__":
    print("--- ⚙️ OPTUNA LABO V2 (LogLoss Optimization) ---")
    
    # 1. Chargement & Préparation (identique football.py)
    X, y = prepare_training_matrix(
        use_corr_removal=True,
        corr_threshold=0.95,
        use_topk=True,
        topk_features=800,
        add_pred_goal_diff=True,
        random_state=42,
    )
    print(f"Features Finales : {X.shape[1]}")

    # 2. MENU DE CHOIX
    print("\nQuel modèle veux-tu optimiser (Metrics: LogLoss) ?")
    print("1. LightGBM (Très Rapide)")
    print("2. XGBoost (Moyen)")
    print("3. CatBoost (Lent mais puissant)")
    print("4. TOUS (Séquentiel : LGB -> XGB -> CAT)")
    choice = input("Ton choix (1/2/3/4) : ")

    if choice == '1':
        run_optimization("LightGBM", optimize_lightgbm, 50, X, y)
    elif choice == '2':
        run_optimization("XGBoost", optimize_xgboost, 50, X, y)
    elif choice == '3':
        run_optimization("CatBoost", optimize_catboost, 30, X, y)
    elif choice == '4':
        print("\n🔄 Lancement complet séquentiel...")
        res_lgb = run_optimization("LightGBM", optimize_lightgbm, 50, X, y)
        res_xgb = run_optimization("XGBoost", optimize_xgboost, 50, X, y)
        res_cat = run_optimization("CatBoost", optimize_catboost, 30, X, y)
        
        print("\n\n🎉🎉 TOUT EST FINI ! RÉCAPITULATIF :")
        print(f"👉 LightGBM: {res_lgb}")
        print(f"👉 XGBoost: {res_xgb}")
        print(f"👉 CatBoost: {res_cat}")
    else:
        print("Choix invalide.")
        sys.exit()