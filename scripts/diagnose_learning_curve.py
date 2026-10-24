"""
Diagnose Learning Curve (Bias vs Variance).
Reproduces the exact pipeline from run_pipeline.py (Stacking + Calibration)
to analyze if the model suffers from Underfitting (High Bias) or Overfitting (High Variance).
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings

# Configuration
warnings.filterwarnings('ignore')

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from src.data import load_data
    from src.features import build_features
except ImportError:
    sys.path.append(os.path.join(project_root, 'src'))
    from data import load_data
    from features import build_features


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    print(f"Computing learning curve for {title}...")
    print(f"Train sizes: {train_sizes}")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy', verbose=1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print("\nResults:")
    print(f"Train Sizes: {train_sizes}")
    print(f"Train Scores: {train_scores_mean}")
    print(f"Validation Scores: {test_scores_mean}")

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    output_path = os.path.join(project_root, 'learning_curve.png')
    plt.savefig(output_path)
    print(f"\n✅ Learning Curve saved to: {output_path}")
    
    return train_scores_mean, test_scores_mean


if __name__ == "__main__":
    print("=== 1. Data Loading & Preprocessing (Identical to run_pipeline.py) ===")
    
    # --- Data Preparation ---
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, y_supp, 
     xtest_h, xtest_a, xpt_h, xpt_a) = load_data()

    X_train = build_features(xt_h, xt_a, xp_h, xp_a)
    X_test = build_features(xtest_h, xtest_a, xpt_h, xpt_a)
    
    # --- Cleaning ---
    print("Removing Correlations...")
    features_for_corr = X_train.drop(columns=['ID'], errors='ignore').select_dtypes(include=[np.number])
    corr_matrix = features_for_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    X_train.drop(columns=to_drop, errors='ignore', inplace=True)
    
    # Alignment (Simulated with Test just to be consistent, though not strictly needed for Learning Curve itself, 
    # but needed to match the exact feature set produced by pipeline)
    X_test.drop(columns=to_drop, errors='ignore', inplace=True)
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)

    X_train_final = X_train.drop(columns=['ID'], errors='ignore')

    # Target
    if 'TARGET' in y_train_raw.columns:
        y_target = y_train_raw['TARGET'].replace({'AWAY_WINS': 0, 'DRAW': 1, 'HOME_WINS': 2})
    else:
        y_target = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1).replace({
            'AWAY_WINS': 0, 'DRAW': 1, 'HOME_WINS': 2
        })

    # --- Auxiliary Feature ---
    print("Generating Auxiliary Feature (Goal Diff)...")
    y_goal_diff = y_supp['GOAL_DIFF_HOME_AWAY']
    cat_reg = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, loss_function='RMSE', verbose=0, random_seed=42)
    
    # Cross-Val Predictions for Train
    goal_diff_preds_train = cross_val_predict(cat_reg, X_train_final, y_goal_diff, cv=5, n_jobs=-1)
    X_train_final['PRED_GOAL_DIFF'] = goal_diff_preds_train
    
    print(f"Feature set ready: {X_train_final.shape}")


    # === 2. Model Definition (Same as run_pipeline.py - High Bias / Low Variance Config) ===
    print("=== 2. Defining Stacking Pipeline ===")

    params_lgb = {
        'n_estimators': 300, 
        'learning_rate': 0.015, 
        'num_leaves': 31, 
        'subsample': 0.8, 
        'colsample_bytree': 0.6,
        'max_depth': 6,  # Reduced from 18
        'min_child_samples': 100, # Increased from 53
        'reg_alpha': 10.0, # Increased massiveley (L1)
        'reg_lambda': 20.0,  # Increased massiveley (L2)
        'random_state': 42, 'verbose': -1,'n_jobs': -1
    }
    
    params_xgb = {
        'n_estimators': 300, 
        'learning_rate': 0.02, 
        'max_depth': 4, # Reduced
        'subsample': 0.8, 
        'colsample_bytree': 0.6, 
        'eval_metric': 'mlogloss','tree_method': 'hist',
        'reg_alpha': 5.0, # Increased
        'reg_lambda': 10.0, # Increased
        'random_state': 42, 'gamma': 1.0, 'min_child_weight': 5,
        'n_jobs': -1
    }
    
    params_cat = {
        'iterations': 800, 
        'learning_rate': 0.02, 
        'depth': 6, # Reduced from 10
        'l2_leaf_reg': 10.0, # Increased
        'border_count': 101, 'subsample': 0.8,
        'random_strength': 5.0,
        'rsm': 0.6,
        'verbose': 0, 'random_seed': 42,'bootstrap_type': 'Bernoulli',
        'allow_writing_files': False
    }

    # Selector
    selector = SelectFromModel(
        estimator=lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, importance_type='gain', random_state=42, verbose=-1, n_jobs=1),
        max_features=250,
        threshold=-np.inf
    )

    # Stacking
    estimators = [
        ('lgb', lgb.LGBMClassifier(**params_lgb)),
        ('xgb', xgb.XGBClassifier(**params_xgb)),
        ('cat', CatBoostClassifier(**params_cat))
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.1),
        cv=5,
        n_jobs=-1
    )
    
    # Calibration REMOVED to match pipeline
    # calibrated_stack = CalibratedClassifierCV(stacking_clf, method='isotonic', cv=3)
    
    # Full Pipeline
    model = make_pipeline(selector, stacking_clf)

    # === 3. Compute Learning Curve ===
    # Using 5 splits instead of standard CV to match project robustness
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n=== 3. Starting Learning Curve Computation ===")
    print("NOTE: This process is computationally expensive.")
    print("It trains the Full Stacking+Correction pipeline ~25 times.")
    
    plot_learning_curve(
        model, 
        "Learning Curve (Stacked + Calibrated)", 
        X_train_final, 
        y_target, 
        cv=cv, 
        n_jobs=-1, # Parallelize
        train_sizes=np.linspace(0.1, 1.0, 5) # 5 points: 10%, 32.5%, 55%, 77.5%, 100%
    )
    
    print("Done.")
