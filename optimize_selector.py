"""
Optimization script for the Selector Model (LGBMClassifier) used in feature selection.
"""

import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator
from typing import cast

# Import data loading and feature engineering from the main script
# Ensure football_clean.py is in the same directory
try:
    from football_clean import load_data, build_features
except ImportError:
    # If football_clean.py was renamed to football.py
    try:
        from football import load_data, build_features_v2 as build_features
    except ImportError:
        raise ImportError("Could not import load_data and build_features from football_clean.py or football.py")

def prepare_data_for_optimization():
    print("--- Preparing Data for Optimization ---")
    # 1. Load Data
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, y_supp, 
     xtest_h, xtest_a, xpt_h, xpt_a) = load_data()

    # 2. Build Features
    X_train = build_features(xt_h, xt_a, xp_h, xp_a)
    X_test = build_features(xtest_h, xtest_a, xpt_h, xpt_a)
    
    # 3. Correlation Removal (Same as main pipeline)
    print("Cleaning correlations...")
    features_for_corr = X_train.drop(columns=['ID'], errors='ignore').select_dtypes(include=[np.number])
    corr_matrix = features_for_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_train.drop(columns=to_drop, errors='ignore', inplace=True)
    X_test.drop(columns=to_drop, errors='ignore', inplace=True)

    # 4. Alignment
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)
    
    # 5. Drop ID
    X_train = X_train.drop(columns=['ID'], errors='ignore')
    
    # 6. Prepare Target
    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y_train_cls = le.fit_transform(y_classes)
    
    print(f"Data Prepared. Shape: {X_train.shape}")
    return X_train, y_train_cls

def objective(trial, X, y):
    """
    Optuna objective function for the Selector Model.
    Optimizes finding the best hyperparameters for a LightGBM model
    that will later be used for feature selection (importance).
    """
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_class': 3,
        'n_jobs': -1,
        'random_state': 42,
        
        # Hyperparameters to search
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    estimator = cast(BaseEstimator, lgb.LGBMClassifier(**param))
    
    # We optimize for accuracy as a proxy for "good model structure" 
    # which implies good feature importance ranking.
    scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

if __name__ == "__main__":
    # 1. Get Data
    X_train, y_train = prepare_data_for_optimization()
    
    # 2. Setup Optuna
    print("\n--- Starting Optuna Optimization for Selector Model ---")
    study = optuna.create_study(direction='maximize')
    
    # Use functools.partial to pass X and y to the objective if needed, 
    # or just use a lambda/wrapper
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50) # 50 trials for demo, can be increased

    print("\n--- Optimization Finished ---")
    print(f"Best Trial Score: {study.best_value}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
        
    print("\nCopy these parameters into your main script for the 'selector_model'!")
