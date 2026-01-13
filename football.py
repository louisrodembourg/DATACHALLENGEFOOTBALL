import pandas as pd
import numpy as np
import os
import re
import glob
import json
import datetime
import warnings
from typing import cast
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# ML Imports
import lightgbm as lgb
import xgboost as xgb
import optuna
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel

# Configuration
warnings.filterwarnings('ignore')

# =============================================================================
# 1. UTILITAIRES & LOGGING
# =============================================================================

def save_experiment(cv_score, params_dict, description, submission_df=None, folder='experiments'):
    """
    Sauvegarde la config de l'expérience (JSON).
    Si un submission_df est fourni, sauvegarde aussi le CSV.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"cv{cv_score:.4f}_{timestamp}"
    
    # 1. Sauvegarde de la Config (JSON)
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
    """
    VERSION SIMPLE (V2) : Mean et Sum uniquement.
     Agrégation par position des joueurs."""
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' not in numeric_cols: numeric_cols.append('ID')
    
    # Agrégation standard (Mean + Sum)
    cols_to_use = numeric_cols + ['POSITION']
    pivot_df = player_df[cols_to_use].groupby(['ID', 'POSITION']).agg(['mean', 'sum'])
    
    pivot_df.columns = [f'{c[0]}_{c[1]}' for c in pivot_df.columns]
    flat_df = pivot_df.unstack(level='POSITION')
    flat_df.columns = [f'{prefix}_{pos}_{col}' for col, pos in flat_df.columns]
    flat_df.reset_index(inplace=True)
    flat_df.fillna(0, inplace=True)
    return flat_df

def build_features_v2(team_home, team_away, player_home, player_away):
    print("--- Construction des features (Raw) ---")
    
    # 1. Agrégation SIMPLE
    p_home_agg = aggregate_players_by_position(player_home, 'P_HOME')
    p_away_agg = aggregate_players_by_position(player_away, 'P_AWAY')
    
    df = team_home.merge(team_away, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    df.fillna(0, inplace=True)

    # 1.5 TEAM ADVANCED RATIOS / EFFICIENCIES (scopes: season_* and 5_last_match_*)
    # Objectif: compléter les DELTA_ déjà générés avec des ratios et des métriques de qualité.
    EPS = 1e-6
    try:
        def _safe_div(a, b):
            return a / (b + EPS)

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

        # --- Ratios simples (Home/Away) sur quelques signaux robustes ---
        ratio_metrics = [
            'TEAM_BALL_POSSESSION',
            'TEAM_ATTACKS',
            'TEAM_DANGEROUS_ATTACKS',
            'TEAM_SHOTS_TOTAL',
            'TEAM_SHOTS_ON_TARGET',
            'TEAM_SHOTS_INSIDEBOX',
            'TEAM_PASSES',
            'TEAM_SUCCESSFUL_PASSES',
            'TEAM_CORNERS',
        ]

        for m in ratio_metrics:
            for sc in _scopes(m):
                base = f"{m}_{sc}"
                h, a = _col(base, 'HOME'), _col(base, 'AWAY')
                if h in df.columns and a in df.columns:
                    df[f'RATIO_{base}'] = _safe_div(df[h], df[a])

        # --- Possession share (Home / (Home + Away)) ---
        for sc in _scopes('TEAM_BALL_POSSESSION'):
            base = f"TEAM_BALL_POSSESSION_{sc}"
            h, a = _col(base, 'HOME'), _col(base, 'AWAY')
            if h in df.columns and a in df.columns:
                df[f'HOME_POSSESSION_SHARE_{sc}'] = _safe_div(df[h], df[h] + df[a])

        # --- Efficiences / Qualité par scope ---
        all_scopes = set()
        for m in ['TEAM_SHOTS_TOTAL', 'TEAM_SHOTS_ON_TARGET', 'TEAM_SHOTS_INSIDEBOX', 'TEAM_GOALS',
                  'TEAM_ATTACKS', 'TEAM_DANGEROUS_ATTACKS', 'TEAM_PASSES', 'TEAM_SUCCESSFUL_PASSES', 'TEAM_SAVES']:
            all_scopes |= set(_scopes(m))
        all_scopes = sorted(all_scopes)

        for sc in all_scopes:
            shots = f"TEAM_SHOTS_TOTAL_{sc}"
            sot = f"TEAM_SHOTS_ON_TARGET_{sc}"
            inside = f"TEAM_SHOTS_INSIDEBOX_{sc}"
            goals = f"TEAM_GOALS_{sc}"
            attacks = f"TEAM_ATTACKS_{sc}"
            dang = f"TEAM_DANGEROUS_ATTACKS_{sc}"
            passes = f"TEAM_PASSES_{sc}"
            succ = f"TEAM_SUCCESSFUL_PASSES_{sc}"
            saves = f"TEAM_SAVES_{sc}"

            h_shots, a_shots = _col(shots, 'HOME'), _col(shots, 'AWAY')
            h_sot, a_sot = _col(sot, 'HOME'), _col(sot, 'AWAY')
            h_inside, a_inside = _col(inside, 'HOME'), _col(inside, 'AWAY')
            h_goals, a_goals = _col(goals, 'HOME'), _col(goals, 'AWAY')
            h_att, a_att = _col(attacks, 'HOME'), _col(attacks, 'AWAY')
            h_dang, a_dang = _col(dang, 'HOME'), _col(dang, 'AWAY')
            h_pass, a_pass = _col(passes, 'HOME'), _col(passes, 'AWAY')
            h_succ, a_succ = _col(succ, 'HOME'), _col(succ, 'AWAY')
            h_saves, a_saves = _col(saves, 'HOME'), _col(saves, 'AWAY')

            # Shot accuracy: SOT / total
            if h_sot in df.columns and h_shots in df.columns:
                df[f'HOME_SHOT_ACCURACY_{sc}'] = _safe_div(df[h_sot], df[h_shots])
            if a_sot in df.columns and a_shots in df.columns:
                df[f'AWAY_SHOT_ACCURACY_{sc}'] = _safe_div(df[a_sot], df[a_shots])
            if f'HOME_SHOT_ACCURACY_{sc}' in df.columns and f'AWAY_SHOT_ACCURACY_{sc}' in df.columns:
                df[f'DELTA_SHOT_ACCURACY_{sc}'] = df[f'HOME_SHOT_ACCURACY_{sc}'] - df[f'AWAY_SHOT_ACCURACY_{sc}']

            # Conversion: goals / total
            if h_goals in df.columns and h_shots in df.columns:
                df[f'HOME_CONVERSION_{sc}'] = _safe_div(df[h_goals], df[h_shots])
            if a_goals in df.columns and a_shots in df.columns:
                df[f'AWAY_CONVERSION_{sc}'] = _safe_div(df[a_goals], df[a_shots])
            if f'HOME_CONVERSION_{sc}' in df.columns and f'AWAY_CONVERSION_{sc}' in df.columns:
                df[f'DELTA_CONVERSION_{sc}'] = df[f'HOME_CONVERSION_{sc}'] - df[f'AWAY_CONVERSION_{sc}']

            # Inside box share
            if h_inside in df.columns and h_shots in df.columns:
                df[f'HOME_INSIDEBOX_SHARE_{sc}'] = _safe_div(df[h_inside], df[h_shots])
            if a_inside in df.columns and a_shots in df.columns:
                df[f'AWAY_INSIDEBOX_SHARE_{sc}'] = _safe_div(df[a_inside], df[a_shots])
            if f'HOME_INSIDEBOX_SHARE_{sc}' in df.columns and f'AWAY_INSIDEBOX_SHARE_{sc}' in df.columns:
                df[f'DELTA_INSIDEBOX_SHARE_{sc}'] = df[f'HOME_INSIDEBOX_SHARE_{sc}'] - df[f'AWAY_INSIDEBOX_SHARE_{sc}']

            # Danger ratio: dangerous attacks / attacks
            if h_dang in df.columns and h_att in df.columns:
                df[f'HOME_DANGER_RATIO_{sc}'] = _safe_div(df[h_dang], df[h_att])
            if a_dang in df.columns and a_att in df.columns:
                df[f'AWAY_DANGER_RATIO_{sc}'] = _safe_div(df[a_dang], df[a_att])
            if f'HOME_DANGER_RATIO_{sc}' in df.columns and f'AWAY_DANGER_RATIO_{sc}' in df.columns:
                df[f'DELTA_DANGER_RATIO_{sc}'] = df[f'HOME_DANGER_RATIO_{sc}'] - df[f'AWAY_DANGER_RATIO_{sc}']

            # Pass completion: successful passes / passes
            if h_succ in df.columns and h_pass in df.columns:
                df[f'HOME_PASS_COMPLETION_{sc}'] = _safe_div(df[h_succ], df[h_pass])
            if a_succ in df.columns and a_pass in df.columns:
                df[f'AWAY_PASS_COMPLETION_{sc}'] = _safe_div(df[a_succ], df[a_pass])
            if f'HOME_PASS_COMPLETION_{sc}' in df.columns and f'AWAY_PASS_COMPLETION_{sc}' in df.columns:
                df[f'DELTA_PASS_COMPLETION_{sc}'] = df[f'HOME_PASS_COMPLETION_{sc}'] - df[f'AWAY_PASS_COMPLETION_{sc}']

            # Cross attaque vs défense (shots on target vs saves)
            if h_sot in df.columns and a_saves in df.columns:
                df[f'CROSS_HOME_SOT_minus_AWAY_SAVES_{sc}'] = df[h_sot] - df[a_saves]
            if a_sot in df.columns and h_saves in df.columns:
                df[f'CROSS_AWAY_SOT_minus_HOME_SAVES_{sc}'] = df[a_sot] - df[h_saves]

            # Saves per SOT faced (approx)
            if h_saves in df.columns and a_sot in df.columns:
                df[f'HOME_SAVES_PER_SOT_FACED_{sc}'] = _safe_div(df[h_saves], df[a_sot])
            if a_saves in df.columns and h_sot in df.columns:
                df[f'AWAY_SAVES_PER_SOT_FACED_{sc}'] = _safe_div(df[a_saves], df[h_sot])
            if f'HOME_SAVES_PER_SOT_FACED_{sc}' in df.columns and f'AWAY_SAVES_PER_SOT_FACED_{sc}' in df.columns:
                df[f'DELTA_SAVES_PER_SOT_FACED_{sc}'] = df[f'HOME_SAVES_PER_SOT_FACED_{sc}'] - df[f'AWAY_SAVES_PER_SOT_FACED_{sc}']
    except:
        pass


    # 3. SMART DELTAS (Physique & Créativité)
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

    # 4. Deltas Classiques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_features = set([c.replace('_HOME', '') for c in numeric_cols if '_HOME' in c])
    for col in base_features:
        col_h, col_a = f"{col}_HOME", f"{col}_AWAY"
        if col_h in df.columns and col_a in df.columns: df[f'DELTA_{col}'] = df[col_h] - df[col_a]

    # 4.1 Nettoyage ciblé de deltas (data-driven)
    # D'après l'ablation (CV 5-fold x 3) sur ton dataset: certains deltas "season_sum" sur W/L
    # semblent ajouter du bruit / sur-apprentissage.
    deltas_to_drop = [
        'DELTA_TEAM_GAME_WON_season_sum',
        'DELTA_TEAM_GAME_LOST_season_sum',
    ]
    df.drop(columns=[c for c in deltas_to_drop if c in df.columns], inplace=True)

    # 5. NETTOYAGE BASIQUE (Pas de corrélation ici !)
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
# 4. FONCTIONS OPTIMISATION ((Non) utilisées dans le pipeline principal pour le moment)
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
    print(f"Dimensions avant alignement : Train={X_train.shape}, Test={X_test.shape}")
    
    # --- 1. NETTOYAGE CORRÉLATION CENTRALISÉ (La correction critique) ---
    # On calcule uniquement sur le TRAIN pour ne pas tricher
    print("Construction de la matrice de corrélation sur le TRAIN...")
    
    # On exclut 'ID' du calcul pour ne pas le virer par erreur
    features_for_corr = X_train.drop(columns=['ID'], errors='ignore').select_dtypes(include=[np.number])
    
    corr_matrix = features_for_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Liste des colonnes à supprimer (Corr > 0.95)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(f"📉 Suppression de {len(to_drop)} colonnes corrélées (Train & Test)...")
    
    # ON APPLIQUE LA SUPPRESSION SUR LES DEUX
    X_train.drop(columns=to_drop, errors='ignore', inplace=True)
    X_test.drop(columns=to_drop, errors='ignore', inplace=True)

    # --- 2. ALIGNEMENT ET NETTOYAGE ID (Ton code adapté) ---
    # L'alignement va maintenant gérer proprement les petits écarts restants 
    # (ex: une colonne vide dans Test mais pas dans Train qui aurait survécu au nettoyage basique)
    print("Alignement final des colonnes...")
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)
    
    print(f"✅ Dimensions SYNCHRONISÉES : Train={X_train.shape}, Test={X_test.shape}")

    # Sauvegarde et suppression ID
    if 'ID' in X_test.columns:
        ID_test = X_test['ID']
    else:
        # Cas rare où l'ID serait passé en index lors de l'align
        ID_test = X_test.index 
        
    X_train = X_train.drop(columns=['ID'], errors='ignore')
    X_test = X_test.drop(columns=['ID'], errors='ignore')

    # Targets
    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y_train_cls = le.fit_transform(y_classes) # 0:AWAY, 1:DRAW, 2:HOME
    y_train_reg = y_supp['GOAL_DIFF_HOME_AWAY']

    print(f"Shape finale Train: {X_train.shape}")

    # --- B.3 Feature Selection Top-K (data-driven, K=800) ---
    # Motivation: ton benchmark delta_tree_eval (5 folds x 3 repeats) montre qu'un topK-only
    # basé sur les importances améliore nettement le score tout en réduisant la dimension.
    USE_TOPK_FEATURES = True
    TOPK_FEATURES = 800
    if USE_TOPK_FEATURES:
        print(f"--- Sélection Top-{TOPK_FEATURES} features (LightGBM gain importance) ---")
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

        # Filtre train/test
        X_train = X_train[keep].copy()
        X_test = X_test[keep].copy()
        print(f"✅ Après TopK: Train={X_train.shape}, Test={X_test.shape}")

    # --- C. Target Auxiliaire (Goal Diff) ---
    print("--- Ajout Feature Auxiliaire (Goal Diff) ---")
    regressor = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    # Feature 'PRED_GOAL_DIFF' pour le Train (via Cross Validation pour éviter fuite)
    X_train['PRED_GOAL_DIFF'] = cross_val_predict(cast(BaseEstimator, regressor), X_train, y_train_reg, cv=5, n_jobs=-1)
    # Feature 'PRED_GOAL_DIFF' pour le Test (via entraînement complet)
    regressor.fit(X_train.drop(columns=['PRED_GOAL_DIFF']), y_train_reg)
    X_test['PRED_GOAL_DIFF'] = regressor.predict(X_test)

    # # --- D. Feature Selection (Optionnel - Désactivé par défaut) ---
    # if True: # Mettre True pour activer
    #     print("Démarrage de la sélection des features...")
    #     selector = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1, verbose=-1)
    #     selector.fit(X_train, y_train_cls)
    #     model_selector = SelectFromModel(selector, prefit=True, threshold="1.25*mean")
    #     # 3. On sauvegarde les noms des colonnes avant transformation (pour info)
    #     original_cols = X_train.columns
    #     n_original = X_train.shape[1]
    #     X_train_selected = model_selector.transform(X_train)
    #     X_test_selected = model_selector.transform(X_test)
    #     selected_mask = model_selector.get_support()
    #     selected_columns = original_cols[selected_mask]
    #     X_train = pd.DataFrame(cast(np.ndarray, X_train_selected), columns=selected_columns)
    #     X_test = pd.DataFrame(cast(np.ndarray, X_test_selected), columns=selected_columns)

    #     print(f"✅ Nettoyage terminé : Passage de {n_original} à {X_train.shape[1]} features.")
    #     print(f"Les features conservées sont les plus pertinentes pour la victoire.")

    # --- E. Configuration des Modèles ---
    # Si tu veux relancer Optuna, décommente les lignes ci-dessous :
    # study = optuna.create_study(direction='maximize')
    # study.optimize(lambda trial: objective_lgb(trial, X_train, y_train_cls), n_trials=30)
    #best_params_lgb = study.best_params
    #print("Best Params:", best_params_lgb
    
    params_lgb = {
        'n_estimators': 680, 'learning_rate': 0.006047203533566928, 'num_leaves': 87, 
        'colsample_bytree': 0.6044442848527449, 'subsample': 0.800557500971823, 'random_state': 42,
        'max_depth': 13, 'min_child_samples': 63,'n_jobs': -1, 'verbose': -1,'reg_alpha': 4.614173334936142,
        'reg_lambda': 2.049690280922791
    }
    params_xgb = {
        'n_estimators': 444, 'learning_rate': 0.020188642966900427, 'max_depth': 10, 
        'colsample_bytree': 0.6367834808496551, 'subsample': 0.7473194384456971, 'random_state': 42,
        'eval_metric': 'mlogloss', 'tree_method': 'hist', 'n_jobs': -1,'gamma':4.967662956642563,
        'min_child_weight': 6,'reg_alpha': 8.129955132262442, 'reg_lambda': 5.813142771285806
    }
    params_cat = {
        'iterations': 2000, 'learning_rate': 0.03, 'depth': 6, 
        'rsm': 0.6, 'verbose': 0, 'random_state': 42, 'thread_count': -1
    }
    
    # Description pour le log
    run_description = "V2 Features - Full (No Selection) - Stacking (LR meta) - Params Anti-Overfit"
    experiment_params = {
        "lgb": params_lgb, "xgb": params_xgb, "cat": params_cat,
        "stacking": True,
        "meta_learner": "LogisticRegression",
        "meta_features": "OOF predict_proba + PRED_GOAL_DIFF",
    }

    # --- F. Construction du Stacking & Estimation ---
    print("\n--- ÉTAPE 1 : Configuration du Stacking ---")
    
    # 1. Définition des modèles de base
    clf1 = lgb.LGBMClassifier(**params_lgb)
    clf2 = xgb.XGBClassifier(**params_xgb)
    clf3 = CatBoostClassifier(**params_cat)
    
    estimators_list = [
        ('lgb', cast(BaseEstimator, clf1)), 
        ('xgb', cast(BaseEstimator, clf2)), 
        ('cat', cast(BaseEstimator, clf3)),
    ]

    # 2. Définition du "Chef" (Méta-modèle)
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)

    # 3. Création du StackingClassifier (C'est ici qu'on crée le modèle 'eclf')
    # n_jobs=1 est crucial pour éviter les conflits avec le parallélisme interne
    eclf = StackingClassifier(
        estimators=estimators_list,
        final_estimator=meta_learner,
        cv=5,               # Le Stacking garde sa CV interne de 5 folds (indispensable)
        stack_method='auto',
        n_jobs=1,
        passthrough=False,
        verbose=1
    )

    # --- Estimation Rapide (Hold-Out) ---
    print("\n--- ÉTAPE 1 : Estimation Rapide du Score (1 seul run) ---")
    
    # On coupe : 80% pour entraîner, 20% pour vérifier le score
    X_tr_part, X_val_part, y_tr_part, y_val_part = train_test_split(
        X_train, y_train_cls, test_size=0.2, random_state=42, stratify=y_train_cls
    )
    
    print("Entraînement sur 80% des données pour estimation...")
    # On entraîne le Stacking sur la partie 'Train' (il fera sa cuisine interne CV=5 là-dessus)
    eclf.fit(X_tr_part, y_tr_part)
    
    # On teste sur la partie 'Validation'
    preds_val = eclf.predict(X_val_part)
    mon_score_estime = accuracy_score(y_val_part, preds_val)
    
    print(f"📊 Score estimé (Validation set 20%) : {mon_score_estime:.5f}")

    # --- G. Décision et Sauvegarde ---
    
    # Seuil pour passer en prod (à ajuster selon tes ambitions)
    THRESHOLD_SCORE = 0.4850 

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