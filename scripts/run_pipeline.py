"""
Football Match Prediction Pipeline.
Includes: Feature Engineering, Feature Selection, and Model Stacking/Boosting.

This file is structured to be read alongside `docs/methodology_fr.md`.
"""

# =============================================================================
# 1. Imports and Configuration
# =============================================================================

import pandas as pd
import numpy as np
import os
import re
import glob
import json
import datetime
import warnings
import argparse
import sys
from typing import cast, Protocol, Any

# Machine Learning & Stats
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel

# Boosting Libraries
import lightgbm as lgb
import xgboost as xgb
import optuna
from catboost import CatBoostClassifier, CatBoostRegressor

# Add project root to sys.path to allow imports from src/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Custom Modules
try:
    from src.data import load_data
    from src.features import build_features, fit_player_clustering
    from src.utils import save_experiment
except ImportError:
    # If running from root without package structure
    sys.path.append(os.path.join(project_root, 'src'))
    from data import load_data
    from features import build_features, fit_player_clustering
    from utils import save_experiment

# Configuration
warnings.filterwarnings('ignore')

# To clean output and avoid warnings
class _ProbClassifier(Protocol):
    classes_: Any

    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...
    def predict_proba(self, X: Any) -> Any: ...


def _parse_args():
    parser = argparse.ArgumentParser(description="Football Pipeline: Training and Submission")
    parser.add_argument(
        "--model",
        choices=["lgb", "xgb", "cat", "stack"],
        default=None,
        help="Model to use: lgb, xgb, cat, stack.",
    )
    return parser.parse_args()


def _prompt_model_choice() -> str:
    print("\n=== Model Choice ===")
    print("1. LightGBM (lgb)")
    print("2. XGBoost (xgb)")
    print("3. CatBoost (cat)")
    print("4. Stacking (stack) [default]")
    choice = input("Your choice (1/2/3/4): ").strip()
    mapping = {"1": "lgb", "2": "xgb", "3": "cat", "4": "stack"}
    return mapping.get(choice, "stack")

# =============================================================================
# 5. Main Pipeline (`if __name__ == "__main__":`)
# =============================================================================

if __name__ == "__main__":

    args = _parse_args()
    
    # --- 5.A Data Preparation ---
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, y_supp, 
     xtest_h, xtest_a, xpt_h, xpt_a) = load_data()

    # --- 5.A.1 Player Clustering (Train on Train Set) ---
    # Combine home and away players from train set to learn roles
    print("Training Player Roles (Clustering)...")
    # Only use common columns just in case
    cols_h = set(xp_h.columns)
    cols_a = set(xp_a.columns)
    common_cols = list(cols_h.intersection(cols_a))
    
    all_train_players = pd.concat([xp_h[common_cols], xp_a[common_cols]], axis=0, ignore_index=True)
    cluster_bundle = fit_player_clustering(all_train_players, k=6)

    # --- 5.A.2 Data Preparation (Feature Construction) ---
    X_train = build_features(xt_h, xt_a, xp_h, xp_a, cluster_bundle=cluster_bundle)
    X_test = build_features(xtest_h, xtest_a, xpt_h, xpt_a, cluster_bundle=cluster_bundle)
    print(f"Initial Dimensions : Train={X_train.shape}, Test={X_test.shape}")
    
    # --- 5.B Cleaning and Feature Selection ---
    
    # 5.B.1 Correlation Removal
    print("Correlation Analysis (Train set)...")
    
    features_for_corr = X_train.drop(columns=['ID'], errors='ignore').select_dtypes(include=[np.number])
    corr_matrix = features_for_corr.corr().abs()
    #  Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(f"📉 Removing {len(to_drop)} correlated columns (>0.95)...")
    
    X_train.drop(columns=to_drop, errors='ignore', inplace=True)
    X_test.drop(columns=to_drop, errors='ignore', inplace=True)

    # 5.B.2 Alignment Train/Test
    print("Column Alignment...")
    
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)
    print(f"✅ Synchronized Dimensions : Train={X_train.shape}, Test={X_test.shape}")

    # Handling ID
    if 'ID' in X_test.columns:
        ID_test = X_test['ID']
    else:
        ID_test = X_test.index 
    
    # Drop ID for training
    X_train_final = X_train.drop(columns=['ID'], errors='ignore')
    X_test_final = X_test.drop(columns=['ID'], errors='ignore')

    # Target Reconstruction
    # Y_train.csv is OHE (HOME_WINS, DRAW, AWAY_WINS), we need a single column for classification
    if 'TARGET' in y_train_raw.columns:
        y_target = y_train_raw['TARGET'].replace({'AWAY_WINS': 0, 'DRAW': 1, 'HOME_WINS': 2})
    else:
        # Reconstruct target from OHE columns
        # 0: AWAY_WINS, 1: DRAW, 2: HOME_WINS
        y_target = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1).replace({
            'AWAY_WINS': 0, 'DRAW': 1, 'HOME_WINS': 2
        })

    # 5.B.3 Feature Selection (MOVED TO PIPELINE TO AVOID LEAKAGE)
    # The selection led to Overfitting/Optimistic bias in CV.
    # Now handled inside the model pipeline.
    print("Feature Selection will be handled inside the Model Pipeline.")
    
    # --- 5.C Auxiliary Feature (Goal Diff) ---
    print("Training Auxiliary Model (Goal Diff)...")
    y_goal_diff = y_supp['GOAL_DIFF_HOME_AWAY']
    
    # Removing extreme outliers if any (optional, standard scaling is robust enough)
    
    cat_reg = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, loss_function='RMSE', verbose=0, random_seed=42)
    
    # Cross-Val Predictions for Train (to avoid leak)
    # We predict goal diff for each fold, then use it as feature
    goal_diff_preds_train = cross_val_predict(cat_reg, X_train_final, y_goal_diff, cv=5, n_jobs=-1)
    
    # Fit on full train for Test prediction
    cat_reg.fit(X_train_final, y_goal_diff)
    goal_diff_preds_test = cat_reg.predict(X_test_final)
    
    # Add as feature
    X_train_final['PRED_GOAL_DIFF'] = goal_diff_preds_train
    X_test_final['PRED_GOAL_DIFF'] = goal_diff_preds_test
    print("✅ Auxiliary Feature added.")

    # =============================================================================
    # 6. Model Training & Prediction
    # =============================================================================
    
    model_choice = args.model if args.model else _prompt_model_choice()
    
    # Hyperparameters (Optimized previously)
    # Added L1/L2 regularization to prevent overfitting on 800+ features
    params_lgb = {
        'n_estimators': 675, 'learning_rate': 0.006921430104787609, 'num_leaves': 96, 
        'subsample': 0.9451510441002412, 'colsample_bytree':  0.7349031047131501,'max_depth': 18,'min_child_samples': 53,
        'reg_alpha': 0.018217537238705006, 'reg_lambda': 8.02527438958144,  # L1/L2
        'random_state': 42, 'verbose': -1,'n_jobs': -1
    }
    
    params_xgb = {
        'n_estimators': 463, 'learning_rate': 0.013621254775271107, 'max_depth': 3, 
        'subsample': 0.5465452013828958, 'colsample_bytree': 0.5113398671313254, 
        'eval_metric': 'mlogloss','tree_method': 'hist',
        'reg_alpha': 0.7854236780369659, 'reg_lambda': 3.5901935424101845,  # L1/L2
        'random_state': 42, 'gamma': 2.3773484745997786,'min_child_weight': 2,
        'n_jobs': -1
    }
    
    params_cat = {
        'iterations': 1041, 'learning_rate': 0.018811897562003008, 'depth': 10, 
        'l2_leaf_reg': 2.2707122819052272,'border_count': 101, 'subsample': 0.7912439842621549,
        'random_strength': 2.2811358187843416,'rsm': 0.6,
        'verbose': 0, 'random_seed': 42,'bootstrap_type': 'Bernoulli',
        'allow_writing_files': False
    }

    # Model Definition
    
    # Feature Selector for Pipeline
    selector = SelectFromModel(
        estimator=lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, importance_type='gain', random_state=42, verbose=-1, n_jobs=1),
        max_features=1350,
        threshold=-np.inf
    )

    base_model = None
    if model_choice == 'lgb':
        base_model = lgb.LGBMClassifier(**params_lgb)
    elif model_choice == 'xgb':
        base_model = xgb.XGBClassifier(**params_xgb)
    elif model_choice == 'cat':
        base_model = CatBoostClassifier(**params_cat)
    elif model_choice == 'stack':
        # Stacking
        estimators = [
            ('lgb', lgb.LGBMClassifier(**params_lgb)),
            ('xgb', xgb.XGBClassifier(**params_xgb)),
            ('cat', CatBoostClassifier(**params_cat))
        ]
        base_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )
        
        # Calibration of the final Stacking output
        # This improves the reliability of the probabilities predicted by the stack
        base_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        
    # Wrap in Pipeline to prevent leakage during Feature Selection
    model = make_pipeline(selector, base_model)

    # 6.1 Validation (Robust Estimate with CV)
    print(f"--- Training {model_choice.upper()} (Cross-Validation) ---")
    
    # Utilisation de la validation croisée (5 folds) pour un score robuste
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_final, y_target, cv=cv, scoring='accuracy', n_jobs=-1)
    
    score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"📊 CV Score (5-Folds) : {score:.4f} (+/- {std_score:.4f})")
    
    if score < 0.4870:
        print(f"⚠️ Score too low (< 0.4850). Aborting full training.")
        save_experiment(score, {"model": model_choice}, "Score insufficient", None)
    else:
        # 6.2 Full Training
        print(f"🚀 Score valid! Retraining on FULL dataset...")
        model.fit(X_train_final, y_target)
        
        # 6.3 Prediction
        print("🔮 Predicting Test set...")
        test_probs = model.predict_proba(X_test_final)
        test_preds = np.argmax(test_probs, axis=1) # 0, 1, 2
        
        # Export in OHE format (HOME_WINS, DRAW, AWAY_WINS)
        submission = pd.DataFrame({'ID': ID_test})
        submission['HOME_WINS'] = (test_preds == 2).astype(int)
        submission['DRAW'] = (test_preds == 1).astype(int)
        submission['AWAY_WINS'] = (test_preds == 0).astype(int)
        
        save_experiment(score, {"model": model_choice}, f"Full Run {model_choice}", submission)
