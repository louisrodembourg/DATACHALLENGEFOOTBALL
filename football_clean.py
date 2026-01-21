"""
Football Match Prediction Pipeline.
Includes: Feature Engineering, Feature Selection, and Model Stacking/Boosting.

This file is structured to be read alongside `EXPLICATION_FOOTBALL_PY.md`.
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
# 2. Utilities & Logging (`save_experiment`)
# =============================================================================

def save_experiment(cv_score, params_dict, description, submission_df=None, folder='experiments'):
    """
    Saves the experiment configuration (JSON) and the submission file if validated.
    See Section 2 of `EXPLICATION_FOOTBALL_PY.md`.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"cv{cv_score:.4f}_{timestamp}"
    
    # 1. Save Config (JSON)
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
    
    print(f"\n[INFO] Config saved: {config_filename}")

    # 2. Save CSV (If validated)
    if submission_df is not None:
        # Raw version
        csv_filename = f"{folder}/sub_{base_name}.csv"
        submission_df.to_csv(csv_filename, index=False)
        print(f"[INFO] Experiment CSV saved: {csv_filename}")
        
        # Incremental version for submission
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
        print(f"[SUCCESS] Ready for submission: '{final_filename}'")

# =============================================================================
# 3. Data Loading (`load_data`)
# =============================================================================

def load_data(base_path='data/'):
    print("--- Loading Data ---")
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
# 4. Feature Engineering
# =============================================================================

def aggregate_players_by_position(player_df, prefix):
    """
    4.1 Player Aggregation
    Aggregates player statistics by position (Mean and Sum).
    """
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' not in numeric_cols: numeric_cols.append('ID')
    
    cols_to_use = numeric_cols + ['POSITION']
    #For every match and position, compute mean and sum of numeric stats
    pivot_df = player_df[cols_to_use].groupby(['ID', 'POSITION']).agg(['mean', 'sum'])
    # Flatten MultiIndex columns
    pivot_df.columns = [f'{c[0]}_{c[1]}' for c in pivot_df.columns]
    flat_df = pivot_df.unstack(level='POSITION')
    # Rename columns to include prefix and position
    flat_df.columns = [f'{prefix}_{pos}_{col}' for col, pos in flat_df.columns]
    flat_df.reset_index(inplace=True)
    flat_df.fillna(0, inplace=True)
    return flat_df

def build_features(team_home, team_away, player_home, player_away):
    """
    4.2 Advanced Construction (`build_features_v2`)
    Combines data, calculates ratios, smart deltas, and performs cleaning.
    """
    print("--- Feature Construction ---")
    
    # 4.2.1 Fusions
    p_home_agg = aggregate_players_by_position(player_home, 'P_HOME')
    p_away_agg = aggregate_players_by_position(player_away, 'P_AWAY')
    # Merge team stats with aggregated player stats
    df = team_home.merge(team_away, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    df.fillna(0, inplace=True)

    # 4.2.2 Ratios and Efficiencies
    EPS = 1e-6
    try:
        # Safe division 
        def _safe_div(a, b):
            return a / (b + EPS)
        # Helper to find scopes for a given metric prefix
        def _scopes(metric_prefix: str):
            scopes = set()
            for c in df.columns:
                if not c.startswith(metric_prefix + '_'):
                    continue
                if not c.endswith('_HOME'):
                    continue
                scopes.add(c[len(metric_prefix) + 1 : -len('_HOME')])
            return sorted(scopes)

        def _col(base: str, side: str):
            return f"{base}_{side}"

        # Simple Ratios Home/Away
        ratio_metrics = [
            'TEAM_BALL_POSSESSION', 'TEAM_ATTACKS', 'TEAM_DANGEROUS_ATTACKS',
            'TEAM_SHOTS_TOTAL', 'TEAM_SHOTS_ON_TARGET', 'TEAM_SHOTS_INSIDEBOX',
            'TEAM_PASSES', 'TEAM_SUCCESSFUL_PASSES', 'TEAM_CORNERS'
        ]
        # Compute ratios
        for m in ratio_metrics:
            for sc in _scopes(m):
                base = f"{m}_{sc}"
                h, a = _col(base, 'HOME'), _col(base, 'AWAY')
                if h in df.columns and a in df.columns:
                    df[f'RATIO_{base}'] = _safe_div(df[h], df[a])

        # Possession Share
        for sc in _scopes('TEAM_BALL_POSSESSION'):
            base = f"TEAM_BALL_POSSESSION_{sc}"
            h, a = _col(base, 'HOME'), _col(base, 'AWAY')
            if h in df.columns and a in df.columns:
                df[f'HOME_POSSESSION_SHARE_{sc}'] = _safe_div(df[h], df[h] + df[a])

        # Advanced Metrics (Conversion, Accuracy, Danger)
        all_metrics = ['TEAM_SHOTS_TOTAL', 'TEAM_SHOTS_ON_TARGET', 'TEAM_SHOTS_INSIDEBOX', 'TEAM_GOALS',
                  'TEAM_ATTACKS', 'TEAM_DANGEROUS_ATTACKS', 'TEAM_PASSES', 'TEAM_SUCCESSFUL_PASSES', 'TEAM_SAVES']
        # Gather all scopes available in the data for these metrics (_season_sum, _last_5_games_avg)
        all_scopes = set()
        for m in all_metrics:
            all_scopes |= set(_scopes(m))
        all_scopes = sorted(all_scopes)

        for sc in all_scopes:
            # Column retrieval
            cols = {m: (f"{m}_{sc}_HOME", f"{m}_{sc}_AWAY") for m in all_metrics}
            
            # Shot Accuracy
            if cols['TEAM_SHOTS_ON_TARGET'][0] in df:
                df[f'HOME_SHOT_ACCURACY_{sc}'] = _safe_div(df[cols['TEAM_SHOTS_ON_TARGET'][0]], df[cols['TEAM_SHOTS_TOTAL'][0]])
            if cols['TEAM_SHOTS_ON_TARGET'][1] in df:
                df[f'AWAY_SHOT_ACCURACY_{sc}'] = _safe_div(df[cols['TEAM_SHOTS_ON_TARGET'][1]], df[cols['TEAM_SHOTS_TOTAL'][1]])
            if f'HOME_SHOT_ACCURACY_{sc}' in df:
                df[f'DELTA_SHOT_ACCURACY_{sc}'] = df[f'HOME_SHOT_ACCURACY_{sc}'] - df[f'AWAY_SHOT_ACCURACY_{sc}']

            # Conversion Rate
            if cols['TEAM_GOALS'][0] in df:
                df[f'HOME_CONVERSION_{sc}'] = _safe_div(df[cols['TEAM_GOALS'][0]], df[cols['TEAM_SHOTS_TOTAL'][0]])
            if cols['TEAM_GOALS'][1] in df:
                df[f'AWAY_CONVERSION_{sc}'] = _safe_div(df[cols['TEAM_GOALS'][1]], df[cols['TEAM_SHOTS_TOTAL'][1]])
            if f'HOME_CONVERSION_{sc}' in df:
                df[f'DELTA_CONVERSION_{sc}'] = df[f'HOME_CONVERSION_{sc}'] - df[f'AWAY_CONVERSION_{sc}']

            # Inside Box Share
            if cols['TEAM_SHOTS_INSIDEBOX'][0] in df:
                df[f'HOME_INSIDEBOX_SHARE_{sc}'] = _safe_div(df[cols['TEAM_SHOTS_INSIDEBOX'][0]], df[cols['TEAM_SHOTS_TOTAL'][0]])
            if cols['TEAM_SHOTS_INSIDEBOX'][1] in df:
                df[f'AWAY_INSIDEBOX_SHARE_{sc}'] = _safe_div(df[cols['TEAM_SHOTS_INSIDEBOX'][1]], df[cols['TEAM_SHOTS_TOTAL'][1]])
            if f'HOME_INSIDEBOX_SHARE_{sc}' in df:
                df[f'DELTA_INSIDEBOX_SHARE_{sc}'] = df[f'HOME_INSIDEBOX_SHARE_{sc}'] - df[f'AWAY_INSIDEBOX_SHARE_{sc}']

            # Danger Ratio
            if cols['TEAM_DANGEROUS_ATTACKS'][0] in df:
                df[f'HOME_DANGER_RATIO_{sc}'] = _safe_div(df[cols['TEAM_DANGEROUS_ATTACKS'][0]], df[cols['TEAM_ATTACKS'][0]])
            if cols['TEAM_DANGEROUS_ATTACKS'][1] in df:
                df[f'AWAY_DANGER_RATIO_{sc}'] = _safe_div(df[cols['TEAM_DANGEROUS_ATTACKS'][1]], df[cols['TEAM_ATTACKS'][1]])
            if f'HOME_DANGER_RATIO_{sc}' in df:
                df[f'DELTA_DANGER_RATIO_{sc}'] = df[f'HOME_DANGER_RATIO_{sc}'] - df[f'AWAY_DANGER_RATIO_{sc}']

            # Pass Completion
            if cols['TEAM_SUCCESSFUL_PASSES'][0] in df:
                df[f'HOME_PASS_COMPLETION_{sc}'] = _safe_div(df[cols['TEAM_SUCCESSFUL_PASSES'][0]], df[cols['TEAM_PASSES'][0]])
            if cols['TEAM_SUCCESSFUL_PASSES'][1] in df:
                df[f'AWAY_PASS_COMPLETION_{sc}'] = _safe_div(df[cols['TEAM_SUCCESSFUL_PASSES'][1]], df[cols['TEAM_PASSES'][1]])
            if f'HOME_PASS_COMPLETION_{sc}' in df:
                df[f'DELTA_PASS_COMPLETION_{sc}'] = df[f'HOME_PASS_COMPLETION_{sc}'] - df[f'AWAY_PASS_COMPLETION_{sc}']

            # Attack vs Defense (Shots on Target vs Saves)
            if cols['TEAM_SHOTS_ON_TARGET'][0] in df and cols['TEAM_SAVES'][1] in df:
                df[f'CROSS_HOME_SOT_minus_AWAY_SAVES_{sc}'] = df[cols['TEAM_SHOTS_ON_TARGET'][0]] - df[cols['TEAM_SAVES'][1]]
            if cols['TEAM_SHOTS_ON_TARGET'][1] in df and cols['TEAM_SAVES'][0] in df:
                df[f'CROSS_AWAY_SOT_minus_HOME_SAVES_{sc}'] = df[cols['TEAM_SHOTS_ON_TARGET'][1]] - df[cols['TEAM_SAVES'][0]]

            # Saves per SOT faced
            if cols['TEAM_SAVES'][0] in df and cols['TEAM_SHOTS_ON_TARGET'][1] in df:
                df[f'HOME_SAVES_PER_SOT_FACED_{sc}'] = _safe_div(df[cols['TEAM_SAVES'][0]], df[cols['TEAM_SHOTS_ON_TARGET'][1]])
            if cols['TEAM_SAVES'][1] in df and cols['TEAM_SHOTS_ON_TARGET'][0] in df:
                df[f'AWAY_SAVES_PER_SOT_FACED_{sc}'] = _safe_div(df[cols['TEAM_SAVES'][1]], df[cols['TEAM_SHOTS_ON_TARGET'][0]])
            if f'HOME_SAVES_PER_SOT_FACED_{sc}' in df:
                df[f'DELTA_SAVES_PER_SOT_FACED_{sc}'] = df[f'HOME_SAVES_PER_SOT_FACED_{sc}'] - df[f'AWAY_SAVES_PER_SOT_FACED_{sc}']

    except Exception:
        pass

    # 4.2.3 Smart Deltas (Physics & Creativity)
    try:
        # Duel Striker vs Goalkeeper
        c_shot_h = [c for c in df.columns if 'P_HOME' in c and 'FORWARD' in c and 'SHOTS' in c and 'TARGET' in c and 'sum' in c]
        c_save_a = [c for c in df.columns if 'P_AWAY' in c and 'GOALKEEPER' in c and 'SAVES' in c and 'sum' in c]
        if c_shot_h and c_save_a: df['DUEL_ATT_H_GK_A'] = df[c_shot_h[0]] - df[c_save_a[0]]
            
        c_shot_a = [c for c in df.columns if 'P_AWAY' in c and 'FORWARD' in c and 'SHOTS' in c and 'TARGET' in c and 'sum' in c]
        c_save_h = [c for c in df.columns if 'P_HOME' in c and 'GOALKEEPER' in c and 'SAVES' in c and 'sum' in c]
        if c_shot_a and c_save_h: df['DUEL_ATT_A_GK_H'] = df[c_shot_a[0]] - df[c_save_h[0]]

        # Physical Dominance (Duels)
        col_duel = 'PLAYER_DUELS_WON'
        cols_duels_h = [c for c in df.columns if 'P_HOME' in c and col_duel in c and 'sum' in c]
        cols_duels_a = [c for c in df.columns if 'P_AWAY' in c and col_duel in c and 'sum' in c]
        if cols_duels_h: df['PHYSICAL_DOMINANCE'] = df[cols_duels_h].sum(axis=1) - df[cols_duels_a].sum(axis=1)

        # Creativity (Key Passes)
        col_key = 'PLAYER_KEY_PASSES'
        cols_key_h = [c for c in df.columns if 'P_HOME' in c and col_key in c and 'sum' in c]
        cols_key_a = [c for c in df.columns if 'P_AWAY' in c and col_key in c and 'sum' in c]
        if cols_key_h: df['CREATIVITY_DIFF'] = df[cols_key_h].sum(axis=1) - df[cols_key_a].sum(axis=1)
    except Exception: pass

    # 4.2.4 Classic Deltas (Home - Away)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_features = set([c.replace('_HOME', '') for c in numeric_cols if '_HOME' in c])
    for col in base_features:
        col_h, col_a = f"{col}_HOME", f"{col}_AWAY"
        if col_h in df.columns and col_a in df.columns: df[f'DELTA_{col}'] = df[col_h] - df[col_a]

    # 4.2.5 Targeted Removal
    deltas_to_drop = [
        'DELTA_TEAM_GAME_WON_season_sum',
        'DELTA_TEAM_GAME_LOST_season_sum',
    ]
    df.drop(columns=[c for c in deltas_to_drop if c in df.columns], inplace=True)

    # 4.2.6 Basic Cleaning
    df = df.loc[:, (df != 0).any(axis=0)] # Remove columns with all 0s
    zeros = (df == 0).mean()
    df = df.loc[:, zeros < 0.995] # Remove quasi-empty columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    cols_to_drop = df.select_dtypes(include=['object']).columns
    if len(cols_to_drop) > 0: df.drop(columns=cols_to_drop, inplace=True)

    drop_cols = ['LEAGUE_HOME', 'TEAM_NAME_HOME', 'LEAGUE_AWAY', 'TEAM_NAME_AWAY', 'LEAGUE', 'TEAM_NAME']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    return df

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
        
    X_train = X_train.drop(columns=['ID'], errors='ignore')
    X_test = X_test.drop(columns=['ID'], errors='ignore')

    # 5.B.3 Target Encoding
    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    # Encode target classes for classification
    y_train_cls = le.fit_transform(y_classes) # 0:AWAY, 1:DRAW, 2:HOME
    y_train_reg = y_supp['GOAL_DIFF_HOME_AWAY']

    # 5.B.4 Top-K Selection
    USE_TOPK_FEATURES = True
    TOPK_FEATURES = 800
    if USE_TOPK_FEATURES:
        print(f"--- Top-{TOPK_FEATURES} Features Selection (LightGBM Importance) ---")
        selector_model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        selector_model.fit(X_train, y_train_cls)
        booster = selector_model.booster_
        importances = booster.feature_importance(importance_type='gain')
        feature_names = booster.feature_name()
        imp_df = pd.DataFrame({'feature': feature_names, 'importance_gain': importances})
        imp_df.sort_values('importance_gain', ascending=False, inplace=True)
        keep = imp_df['feature'].head(min(TOPK_FEATURES, imp_df.shape[0])).tolist()

        X_train = X_train[keep].copy()
        X_test = X_test[keep].copy()
        print(f"✅ After Selection : Train={X_train.shape}, Test={X_test.shape}")

    # --- 5.C Auxiliary Feature (Goal Diff) ---
    print("--- Adding Auxiliary Feature (Stacking Goal Diff) ---")
    regressor = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    
    # Train: Cross Val predict to avoid leakage
    X_train['PRED_GOAL_DIFF'] = cross_val_predict(cast(BaseEstimator, regressor), X_train, y_train_reg, cv=5, n_jobs=-1)
    
    # Test: Fit on full train and predict
    regressor.fit(X_train.drop(columns=['PRED_GOAL_DIFF']), y_train_reg)
    X_test['PRED_GOAL_DIFF'] = regressor.predict(X_test)

    # --- 5.D Model Construction ---
    model_choice = args.model or _prompt_model_choice()
    print(f"\n✅ Selected Model : {model_choice}")

    # Optimized Parameters
    params_lgb = {
        'n_estimators': 675, 'learning_rate': 0.006921430104787609, 'num_leaves': 96, 
        'colsample_bytree': 0.7349031047131501, 'subsample': 0.9451510441002412, 'random_state': 42,
        'max_depth': 18, 'min_child_samples': 53,'n_jobs': -1, 'verbose': -1,'reg_alpha': 0.018217537238705006,
        'reg_lambda': 8.02527438958144}
    params_xgb = {
        'n_estimators': 463, 'learning_rate': 0.013621254775271107, 'max_depth': 3, 
        'colsample_bytree': 0.5113398671313254, 'subsample': 0.5465452013828958, 'random_state': 42,
        'eval_metric': 'mlogloss', 'tree_method': 'hist', 'n_jobs': -1,'gamma':2.3773484745997786,
        'min_child_weight': 2,'reg_alpha': 0.7854236780369659, 'reg_lambda': 3.5901935424101845
    }
    params_cat = {
        'iterations': 1041, 'learning_rate': 0.018811897562003008, 'depth': 10,
        'l2_leaf_reg': 2.2707122819052272,'border_count': 101,'subsample': 0.7912439842621549,
        'bootstrap_type': 'Bernoulli',
        'random_strength': 2.2811358187843416,
        'rsm': 0.6, 'verbose': 0, 'random_state': 42, 'thread_count': -1,
        'allow_writing_files': False
    }
    
    run_description = f"V2 Features - TopK=800 - PRED_GOAL_DIFF - model={model_choice}"
    experiment_params = {
        "model": model_choice,
        "lgb": params_lgb,
        "xgb": params_xgb,
        "cat": params_cat,
        "meta_features": "PRED_GOAL_DIFF"
    }

    # Model Initialization
    print("\n--- Model Initialization ---")

    if model_choice == 'lgb':
        model = cast(_ProbClassifier, lgb.LGBMClassifier(**params_lgb))
    elif model_choice == 'xgb':
        model = cast(_ProbClassifier, xgb.XGBClassifier(**params_xgb))
    elif model_choice == 'cat':
        model = cast(_ProbClassifier, CatBoostClassifier(**params_cat))
    else:
        # Stacking
        clf1 = lgb.LGBMClassifier(**params_lgb)
        clf2 = xgb.XGBClassifier(**params_xgb)
        clf3 = CatBoostClassifier(**params_cat)
        estimators_list = [
            ('lgb', cast(BaseEstimator, clf1)),
            ('xgb', cast(BaseEstimator, clf2)),
            ('cat', cast(BaseEstimator, clf3)),
        ]
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        model = cast(
            _ProbClassifier,
            StackingClassifier(
                estimators=estimators_list,
                final_estimator=meta_learner,
                cv=5,
                stack_method='auto',
                n_jobs=1,
                passthrough=False,
                verbose=1,
            ),
        )

    # --- 5.E Validation and Training ---
    
    # 5.E.1 Quick Score Estimation (Hold-Out 80/20)
    print("\n--- 5.E.1 Score Estimation (Validation 20%) ---")
    
    X_tr_part, X_val_part, y_tr_part, y_val_part = train_test_split(
        X_train, y_train_cls, test_size=0.2, random_state=42, stratify=y_train_cls
    )
    
    model.fit(X_tr_part, y_tr_part)
    preds_val = model.predict(X_val_part)
    score_estime = accuracy_score(y_val_part, preds_val)
    
    print(f"📊 Validation Score : {score_estime:.5f}")

    # 5.E.2 Decision (Gate)
    THRESHOLD_SCORE = 0.4850 

    if score_estime < THRESHOLD_SCORE:
        print(f"\n[INFO] Insufficient Score (< {THRESHOLD_SCORE}). Stopping.")
        save_experiment(
            cv_score=score_estime, 
            params_dict=experiment_params, 
            description=run_description,
            submission_df=None
        )
        exit()
    else:
        print(f"\n✅ Score Validated. Starting Final Training on 100% Data.")

    # 5.E.3 Final Training
    model.fit(X_train, y_train_cls)
    probs = np.asarray(model.predict_proba(X_test))

    # --- 5.F Submission ---
    classes = getattr(model, 'classes_', np.array([0, 1, 2]))
    class_to_col = {int(c): i for i, c in enumerate(classes)}
    submission = pd.DataFrame({
        'ID': ID_test,
        'HOME_WINS': probs[:, class_to_col[2]],
        'DRAW': probs[:, class_to_col[1]],
        'AWAY_WINS': probs[:, class_to_col[0]],
    })

    # Binary Conversion
    print("Binary Conversion (Hard Voting)...")
    cols_submit = ['HOME_WINS', 'DRAW', 'AWAY_WINS']
    max_indices = submission[cols_submit].values.argmax(axis=1)
    hard_preds = np.zeros(submission[cols_submit].shape, dtype=int)
    hard_preds[np.arange(len(submission)), max_indices] = 1
    submission[cols_submit] = hard_preds

    save_experiment(
        cv_score=score_estime, 
        params_dict=experiment_params, 
        description=run_description,
        submission_df=submission
    )
