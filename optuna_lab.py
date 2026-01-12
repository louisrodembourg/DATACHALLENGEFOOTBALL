import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import warnings
import sys

# Gestion des imports
try:
    from football import load_data
except ImportError:
    try:
        from football import load_data
    except ImportError:
        print("❌ Impossible de trouver load_data.")
        sys.exit()

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# =============================================================================
# 🧬 FEATURES CHAMPIONNES (0.499 Configuration)
# =============================================================================
def aggregate_players_by_position(player_df, prefix):
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' not in numeric_cols: numeric_cols.append('ID')
    cols_to_use = numeric_cols + ['POSITION']
    # Sécurité doublons colonnes
    player_df = player_df.loc[:, ~player_df.columns.duplicated()]
    pivot_df = player_df[cols_to_use].groupby(['ID', 'POSITION']).agg(['mean', 'sum'])
    pivot_df.columns = [f'{c[0]}_{c[1]}' for c in pivot_df.columns]
    flat_df = pivot_df.unstack(level='POSITION')
    flat_df.columns = [f'{prefix}_{pos}_{col}' for col, pos in flat_df.columns]
    flat_df.reset_index(inplace=True)
    flat_df.fillna(0, inplace=True)
    return flat_df

def build_features_v2(team_home, team_away, player_home, player_away):
    print("   -> Construction features...")
    p_home_agg = aggregate_players_by_position(player_home, 'P_HOME')
    p_away_agg = aggregate_players_by_position(player_away, 'P_AWAY')
    df = team_home.merge(team_away, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    df.fillna(0, inplace=True)

    # Deep Quality
    try:
        df['HOME_EFFICIENCY'] = df['TEAM_GOALS_season_sum_HOME'] / (df['TEAM_SHOTS_TOTAL_season_sum_HOME'] + 1)
        df['AWAY_EFFICIENCY'] = df['TEAM_GOALS_season_sum_AWAY'] / (df['TEAM_SHOTS_TOTAL_season_sum_AWAY'] + 1)
        s_on_h = [c for c in df.columns if 'SHOTS_ON_TARGET' in c and '_HOME' in c and 'TEAM' in c][0]
        s_on_a = [c for c in df.columns if 'SHOTS_ON_TARGET' in c and '_AWAY' in c and 'TEAM' in c][0]
        s_tot_h = 'TEAM_SHOTS_TOTAL_season_sum_HOME'
        s_tot_a = 'TEAM_SHOTS_TOTAL_season_sum_AWAY'
        df['HOME_ACCURACY'] = df[s_on_h] / (df[s_tot_h] + 1)
        df['AWAY_ACCURACY'] = df[s_on_a] / (df[s_tot_a] + 1)
        df['HOME_GK_RESISTANCE'] = 1 - (df['TEAM_GOALS_season_sum_AWAY'] / (df[s_on_a] + 1))
        df['AWAY_GK_RESISTANCE'] = 1 - (df['TEAM_GOALS_season_sum_HOME'] / (df[s_on_h] + 1))
    except: pass

    # Smart Deltas
    try:
        c_shot_h = [c for c in df.columns if 'P_HOME' in c and 'FORWARD' in c and 'SHOTS' in c and 'TARGET' in c and 'sum' in c]
        c_save_a = [c for c in df.columns if 'P_AWAY' in c and 'GOALKEEPER' in c and 'SAVES' in c and 'sum' in c]
        if c_shot_h and c_save_a: df['DUEL_ATT_H_GK_A'] = df[c_shot_h[0]] - df[c_save_a[0]]
        c_shot_a = [c for c in df.columns if 'P_AWAY' in c and 'FORWARD' in c and 'SHOTS' in c and 'TARGET' in c and 'sum' in c]
        c_save_h = [c for c in df.columns if 'P_HOME' in c and 'GOALKEEPER' in c and 'SAVES' in c and 'sum' in c]
        if c_shot_a and c_save_h: df['DUEL_ATT_A_GK_H'] = df[c_shot_a[0]] - df[c_save_h[0]]
        col_duel = 'PLAYER_DUELS_WON'
        cols_duels_h = [c for c in df.columns if 'P_HOME' in c and col_duel in c and 'sum' in c]
        cols_duels_a = [c for c in df.columns if 'P_AWAY' in c and col_duel in c and 'sum' in c]
        if cols_duels_h: df['PHYSICAL_DOMINANCE'] = df[cols_duels_h].sum(axis=1) - df[cols_duels_a].sum(axis=1)
        col_key = 'PLAYER_KEY_PASSES'
        cols_key_h = [c for c in df.columns if 'P_HOME' in c and col_key in c and 'sum' in c]
        cols_key_a = [c for c in df.columns if 'P_AWAY' in c and col_key in c and 'sum' in c]
        if cols_key_h: df['CREATIVITY_DIFF'] = df[cols_key_h].sum(axis=1) - df[cols_key_a].sum(axis=1)
    except: pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_features = set([c.replace('_HOME', '') for c in numeric_cols if '_HOME' in c])
    for col in base_features:
        col_h, col_a = f"{col}_HOME", f"{col}_AWAY"
        if col_h in df.columns and col_a in df.columns: df[f'DELTA_{col}'] = df[col_h] - df[col_a]

    # Nettoyage
    df = df.loc[:, (df != 0).any(axis=0)]
    zeros = (df == 0).mean()
    df = df.loc[:, zeros < 0.995]
    df = df.loc[:, ~df.columns.duplicated()]
    cols_to_drop = df.select_dtypes(include=['object']).columns
    if len(cols_to_drop) > 0: df.drop(columns=cols_to_drop, inplace=True)
    drop_cols = ['LEAGUE_HOME', 'TEAM_NAME_HOME', 'LEAGUE_AWAY', 'TEAM_NAME_AWAY', 'LEAGUE', 'TEAM_NAME']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

# =============================================================================
# 🎯 OPTIMISATION LOG LOSS (Meilleur pour les probas du Stacking)
# =============================================================================

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
        'class_weight': 'balanced' # TRES IMPORTANT POUR LES NULS
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # On optimise le neg_log_loss pour avoir de meilleures probabilités
    scores = cross_val_score(lgb.LGBMClassifier(**params, n_jobs=1), X, y, cv=cv, scoring='neg_log_loss')
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
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(xgb.XGBClassifier(**params, n_jobs=1), X, y, cv=cv, scoring='neg_log_loss')
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
        'auto_class_weights': 'Balanced', # TRES IMPORTANT
        'allow_writing_files': False # Évite de créer des dossiers catboost_info partout
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(CatBoostClassifier(**params, thread_count=1), X, y, cv=cv, scoring='neg_log_loss')
    return scores.mean()

# =============================================================================
# 🚀 MAIN RUN
# =============================================================================

if __name__ == "__main__":
    print("--- ⚙️ OPTUNA LABO V2 (LogLoss Optimization) ---")
    
    # 1. Chargement & Préparation
    data = load_data()
    # Tri intelligent des données (comme vu précédemment)
    train_team = []
    for item in data:
        if hasattr(item, 'shape') and item.shape[0] < 15000:
             if hasattr(item, 'columns') and 'HOME_WINS' in item.columns:
                 y_target = item
             elif len(item.shape) == 2:
                 train_team.append(item)
        elif hasattr(item, 'shape') and 200000 <= item.shape[0] < 400000:
             xp_h_raw = item # On assume que le premier trouvé est le bon
    
    # On fait simple pour le chargement, on reprend la méthode standard si ça marche
    # Sinon on applique le tri
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, _, _, _, _, _) = load_data()

    # Nettoyage majuscules
    def clean_df(df):
        df.columns = [c.upper() for c in df.columns]
        return df.loc[:, ~df.columns.duplicated()]
    
    xt_h, xt_a = clean_df(xt_h), clean_df(xt_a)
    xp_h, xp_a = clean_df(xp_h), clean_df(xp_a)

    print("Construction X_train...")
    X = build_features_v2(xt_h, xt_a, xp_h, xp_a)
    
    # Target
    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y = le.fit_transform(y_classes)
    
    # Nettoyage Corrélation
    print("Nettoyage Corrélation...")
    if 'ID' in X.columns: X.drop(columns=['ID'], inplace=True)
    corr_matrix = X.select_dtypes(include=np.number).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X.drop(to_drop, axis=1, inplace=True)
    print(f"Features Finales : {X.shape[1]}")

    # 2. MENU DE CHOIX
    print("\nQuel modèle veux-tu optimiser (Metrics: LogLoss) ?")
    print("1. LightGBM (Très Rapide)")
    print("2. XGBoost (Moyen)")
    print("3. CatBoost (Lent mais puissant)")
    choice = input("Ton choix (1/2/3) : ")

    study = optuna.create_study(direction='maximize') # Maximize car neg_log_loss est négatif (ex: -0.9 est mieux que -1.1)
    
    if choice == '1':
        print("\n🚀 Optimisation LightGBM en cours...")
        study.optimize(lambda trial: optimize_lightgbm(trial, X, y), n_trials=30)
    elif choice == '2':
        print("\n🚀 Optimisation XGBoost en cours...")
        study.optimize(lambda trial: optimize_xgboost(trial, X, y), n_trials=30)
    elif choice == '3':
        print("\n🚀 Optimisation CatBoost en cours...")
        study.optimize(lambda trial: optimize_catboost(trial, X, y), n_trials=30)
    else:
        print("Choix invalide.")
        sys.exit()

    print("\n✅ OPTIMISATION TERMINÉE !")
    print(f"🏆 Meilleur Score (LogLoss - proche de 0 est mieux) : {study.best_value:.5f}")
    print("📝 Meilleurs Paramètres à copier :")
    print(study.best_params)