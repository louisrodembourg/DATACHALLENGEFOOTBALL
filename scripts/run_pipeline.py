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
    from src.features import build_features
    from src.utils import save_experiment
except ImportError:
    # If running from root without package structure
    sys.path.append(os.path.join(project_root, 'src'))
    from data import load_data
    from features import build_features
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

    # --- 5.A Data Preparation (Feature Construction) ---
    X_train = build_features(xt_h, xt_a, xp_h, xp_a)
    X_test = build_features(xtest_h, xtest_a, xpt_h, xpt_a)
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

    # Target
    y_target = y_train_raw['TARGET'].replace({'AWAY_WINS': 0, 'DRAW': 1, 'HOME_WINS': 2})

    # 5.B.3 Feature Selection (Top-K with LightGBM)
    # Using a fast LGBM to select best features
    print("Feature Selection (LGBM)...")
    lgb_sel = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    # Simple imputation for selection (LGBM handles nans but good practice)
    lgb_sel.fit(X_train_final, y_target)
    
    # Get importance
    importances = pd.Series(lgb_sel.feature_importances_, index=X_train_final.columns)
    # Keep Top 800
    top_k = 800
    if len(importances) > top_k:
        top_features = importances.nlargest(top_k).index.tolist()
        print(f"📉 Reducing to Top {top_k} features.")
        X_train_final = X_train_final[top_features]
        X_test_final = X_test_final[top_features]
    else:
        print(f"✨ Kept all {len(importances)} features (<= {top_k}).")

    # --- 5.C Auxiliary Feature (Goal Diff) ---
    print("Training Auxiliary Model (Goal Diff)...")
    y_goal_diff = y_supp['GOAL_DIFF_HOME_AWAY']
    
    # Removing extreme outliers if any (optional, standard scaling is robust enough)
    
    cat_reg = CatBoostRegressor(iterations=300, depth=4, learning_rate=0.05, loss_function='RMSE', verbose=0, random_seed=42)
    
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
    params_lgb = {
        'n_estimators': 1500, 'learning_rate': 0.015, 'num_leaves': 31, 
        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1
    }
    
    params_xgb = {
        'n_estimators': 1200, 'learning_rate': 0.02, 'max_depth': 5, 
        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 
        'enable_categorical': False, 'n_jobs': -1
    }
    
    params_cat = {
        'iterations': 1500, 'learning_rate': 0.02, 'depth': 6, 
        'l2_leaf_reg': 5, 'loss_function': 'MultiClass', 'verbose': 0, 'random_seed': 42
    }

    # Model Definition
    model = None
    if model_choice == 'lgb':
        model = lgb.LGBMClassifier(**params_lgb)
    elif model_choice == 'xgb':
        model = xgb.XGBClassifier(**params_xgb)
    elif model_choice == 'cat':
        model = CatBoostClassifier(**params_cat)
    elif model_choice == 'stack':
        # Stacking
        estimators = [
            ('lgb', lgb.LGBMClassifier(**params_lgb)),
            ('xgb', xgb.XGBClassifier(**params_xgb)),
            ('cat', CatBoostClassifier(**params_cat))
        ]
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )

    # 6.1 Validation (Fast Estimate)
    print(f"--- Training {model_choice.upper()} (Validation) ---")
    
    # Split 80/20 for fast validation check
    XT_train, XT_val, yt_train, yt_val = train_test_split(X_train_final, y_target, test_size=0.2, random_state=42, stratify=y_target)
    
    model.fit(XT_train, yt_train)
    val_preds = model.predict(XT_val)
    score = accuracy_score(yt_val, val_preds)
    
    print(f"📊 Validation Score (20% holdout) : {score:.4f}")
    
    if score < 0.4850:
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
        
        # Map back to strings
        mapping = {0: 'AWAY_WINS', 1: 'DRAW', 2: 'HOME_WINS'}
        test_preds_str = [mapping[p] for p in test_preds]
        
        # Export
        submission = pd.DataFrame({'ID': ID_test, 'TARGET': test_preds_str})
        
        save_experiment(score, {"model": model_choice}, f"Full Run {model_choice}", submission)
