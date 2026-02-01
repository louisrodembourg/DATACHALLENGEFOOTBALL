"""
Script to test model performance with varying numbers of features (SelectFromModel).
Based on the logic of `run_pipeline.py`.
"""

# =============================================================================
# 1. Imports and Configuration
# =============================================================================

import pandas as pd
import numpy as np
import os
import sys
import argparse
import warnings
import matplotlib.pyplot as plt

# Machine Learning & Stats
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# Boosting Libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

# Custom Modules Setup
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

# Configuration
warnings.filterwarnings('ignore')

def _parse_args():
    parser = argparse.ArgumentParser(description="Feature Count Optimization Loop")
    parser.add_argument(
        "--model",
        choices=["lgb", "xgb", "cat", "stack"],
        default="stack", # Default to stack as requested
        help="Model to use for the loop (default: stack).",
    )
    return parser.parse_args()

# =============================================================================
# Main Loop 
# =============================================================================

if __name__ == "__main__":

    args = _parse_args()
    print(f"--- Starting Feature Loop Optimization using {args.model.upper()} ---")

    # --- 1. Data Preparation (Same as run_pipeline.py) ---
    print("Loading and Building Data (This may take a minute)...")
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, y_supp, 
     xtest_h, xtest_a, xpt_h, xpt_a) = load_data()

    X_train = build_features(xt_h, xt_a, xp_h, xp_a)
    X_test = build_features(xtest_h, xtest_a, xpt_h, xpt_a) # Built to ensure consistency

    # --- 2. Cleaning and Feature Selection (Section 5.B) ---
    print("Correlation Analysis...")
    features_for_corr = X_train.drop(columns=['ID'], errors='ignore').select_dtypes(include=[np.number])
    corr_matrix = features_for_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(f"📉 Removing {len(to_drop)} correlated columns (>0.95)...")
    
    X_train.drop(columns=to_drop, errors='ignore', inplace=True)
    
    # Align
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)

    # Prepare Target
    if 'TARGET' in y_train_raw.columns:
        y_target = y_train_raw['TARGET'].replace({'AWAY_WINS': 0, 'DRAW': 1, 'HOME_WINS': 2})
    else:
        y_target = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1).replace({
            'AWAY_WINS': 0, 'DRAW': 1, 'HOME_WINS': 2
        })
    
    X_train_final = X_train.drop(columns=['ID'], errors='ignore')

    # --- 3. Auxiliary Feature (Goal Diff) (Section 5.C) ---
    print("Training Auxiliary Model (Goal Diff)...")
    y_goal_diff = y_supp['GOAL_DIFF_HOME_AWAY']
    
    cat_reg = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, loss_function='RMSE', verbose=0, random_seed=42)
    
    # Cross-Val Predictions for Train
    goal_diff_preds_train = cross_val_predict(cat_reg, X_train_final, y_goal_diff, cv=5, n_jobs=-1)
    X_train_final['PRED_GOAL_DIFF'] = goal_diff_preds_train
    print("✅ Auxiliary Feature added.")

    # --- 4. Define Models and Params (Exact copy from run_pipeline.py) ---
    params_lgb = {
        'n_estimators': 675, 'learning_rate': 0.006921430104787609, 'num_leaves': 96, 
        'subsample': 0.9451510441002412, 'colsample_bytree':  0.7349031047131501,'max_depth': 18,'min_child_samples': 53,
        'reg_alpha': 0.018217537238705006, 'reg_lambda': 8.02527438958144,
        'random_state': 42, 'verbose': -1,'n_jobs': -1
    }
    
    params_xgb = {
        'n_estimators': 463, 'learning_rate': 0.013621254775271107, 'max_depth': 3, 
        'subsample': 0.5465452013828958, 'colsample_bytree': 0.5113398671313254, 
        'eval_metric': 'mlogloss','tree_method': 'hist',
        'reg_alpha': 0.7854236780369659, 'reg_lambda': 3.5901935424101845,
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

    base_model = None
    if args.model == 'lgb':
        base_model = lgb.LGBMClassifier(**params_lgb)
    elif args.model == 'xgb':
        base_model = xgb.XGBClassifier(**params_xgb)
    elif args.model == 'cat':
        base_model = CatBoostClassifier(**params_cat)
    elif args.model == 'stack':
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

    # --- 5. The Loop: 50 to 1000 features ---
    
    results = []
    # Test from 50 to 1000 by steps of 50
    feature_counts = list(range(1000, 1500, 50)) 
    
    print("\n--- Starting Feature Count Iteration ---")
    print(f"Testing counts: {feature_counts}")
    
    best_score = 0
    best_n = 0

    for n_features in feature_counts:
        print(f"\nProcessing max_features = {n_features} ...")
        
        # Selector (Same definition as run_pipeline)
        selector = SelectFromModel(
            estimator=lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, importance_type='gain', random_state=42, verbose=-1, n_jobs=1),
            max_features=n_features,
            threshold=-np.inf
        )
        
        # Pipeline
        model = make_pipeline(selector, base_model)
        
        # Validation using StratifiedKFold (same as pipeline)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_final, y_target, cv=cv, scoring='accuracy', n_jobs=-1)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"   -> Score: {mean_score:.4f} (+/- {std_score:.4f})")
        results.append({'n_features': n_features, 'score': mean_score, 'std': std_score})
        
        if mean_score > best_score:
            best_score = mean_score
            best_n = n_features

    # --- 6. Results & Plot ---
    print("\n========================================")
    print("           RESULTS SUMMARY")
    print("========================================")
    print(f"Best Configuration: {best_n} features with Score: {best_score:.4f}")
    
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(project_root, f"experiments/feature_count_test_{args.model}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Plotting
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['n_features'], df_results['score'], marker='o', linestyle='-')
        
        # Highlight best point
        best_row = df_results.loc[df_results['score'].idxmax()]
        plt.plot(best_row['n_features'], best_row['score'], 'r*', markersize=15, label=f'Best: {best_row["score"]:.4f}')
        
        plt.title(f'CV Accuracy vs Number of Features ({args.model.upper()})')
        plt.xlabel('Number of Features Selected')
        plt.ylabel('CV Accuracy (5-fold)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        img_path = os.path.join(project_root, f"experiments/feature_count_plot_{args.model}.png")
        plt.savefig(img_path)
        print(f"Plot saved to {img_path}")
    except Exception as e:
        print("Could not save plot:", e)
