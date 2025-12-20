import pandas as pd
import numpy as np
import os
import re
import glob
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import warnings

warnings.filterwarnings('ignore')
import json
import datetime
import os

def save_experiment(submission_df, cv_score, params_dict, description, folder='experiments'):
    """
    Sauvegarde la soumission ET la config associée.
    
    Args:
        submission_df: Le DataFrame de soumission (ID, HOME, DRAW, AWAY)
        cv_score: Ton score moyen de cross-validation (ex: 0.4813)
        params_dict: Dictionnaire contenant tes hyperparamètres (lgb, xgb, cat...)
        description: Une phrase courte pour décrire l'essai (ex: "V2 features sans selection")
        folder: Dossier de sauvegarde
    """
    # Création du dossier si inexistant
    os.makedirs(folder, exist_ok=True)
    
    # Timestamp pour rendre le nom unique
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Nom de base : score_date
    # ex: 0.4813_20231220_234512
    base_name = f"cv{cv_score:.4f}_{timestamp}"
    
    # 1. Sauvegarde du CSV
    csv_filename = f"{folder}/sub_{base_name}.csv"
    submission_df.to_csv(csv_filename, index=False)
    
    # 2. Sauvegarde de la Config (JSON)
    config_filename = f"{folder}/conf_{base_name}.json"
    
    log_data = {
        "timestamp": timestamp,
        "cv_score": cv_score,
        "description": description,
        "parameters": params_dict
    }
    
    with open(config_filename, 'w') as f:
        json.dump(log_data, f, indent=4)
        
    print(f"\n[SUCCÈS] Expérience sauvegardée dans '{folder}/'")
    print(f"📄 CSV: {csv_filename}")
    print(f"⚙️ Config: {config_filename}")
# =============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================

def load_data(base_path='data/'):
    # Chemins à adapter selon vos noms de fichiers réels
    print("Chargement des données...")
    
    # Train
    x_train_team_home = pd.read_csv(f'{base_path}Train_Data/train_home_team_statistics_df.csv')
    x_train_team_away = pd.read_csv(f'{base_path}Train_Data/train_away_team_statistics_df.csv')
    x_train_player_home = pd.read_csv(f'{base_path}Train_Data/train_home_player_statistics_df.csv')
    x_train_player_away = pd.read_csv(f'{base_path}Train_Data/train_away_player_statistics_df.csv')
    y_train = pd.read_csv(f'{base_path}Y_train.csv')
    y_train_supp = pd.read_csv(f'{base_path}benchmark_and_extras/Y_train_supp.csv') # Goal diff
    
    # Test
    x_test_team_home = pd.read_csv(f'{base_path}Test_Data/test_home_team_statistics_df.csv')
    x_test_team_away = pd.read_csv(f'{base_path}Test_Data/test_away_team_statistics_df.csv')
    x_test_player_home = pd.read_csv(f'{base_path}Test_Data/test_home_player_statistics_df.csv')
    x_test_player_away = pd.read_csv(f'{base_path}Test_Data/test_away_player_statistics_df.csv')
    
    return (x_train_team_home, x_train_team_away, x_train_player_home, x_train_player_away, y_train, y_train_supp,
            x_test_team_home, x_test_team_away, x_test_player_home, x_test_player_away)

# =============================================================================
# 2. FEATURE ENGINEERING (LE CŒUR DU SYSTÈME)
# =============================================================================

def aggregate_players_by_position(player_df, prefix):
    """
    Agrège les stats en séparant par POSITION (G, D, M, F).
    Cela capture la structure tactique de l'équipe.
    """
    # 1. On sépare les colonnes numériques et la position
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' not in numeric_cols: numeric_cols.append('ID')
    
    # On garde ID et POSITION pour le pivot
    cols_to_use = numeric_cols + ['POSITION']
    
    # 2. Pivot Table : C'est la magie. 
    # On groupe par ID et POSITION, puis on calcule la moyenne (mean) et la somme (sum)
    # On évite l'écart-type (std) ici pour ne pas exploser le nombre de colonnes (déjà x4 à cause des positions)
    pivot_df = player_df[cols_to_use].groupby(['ID', 'POSITION']).agg(['mean', 'sum'])
    
    # 3. Aplatir le MultiIndex (ex: ('GOALS', 'mean') -> 'GOALS_mean')
    pivot_df.columns = [f'{c[0]}_{c[1]}' for c in pivot_df.columns]
    
    # 4. Dé-empiler les positions (ex: une ligne par match, colonnes: DEFENDER_GOALS_mean, FORWARD_GOALS_mean...)
    flat_df = pivot_df.unstack(level='POSITION')
    
    # 5. Aplatir à nouveau les noms de colonnes
    flat_df.columns = [f'{prefix}_{pos}_{col}' for col, pos in flat_df.columns]
    flat_df.reset_index(inplace=True)
    
    # 6. Gestion des valeurs manquantes (ex: si une équipe joue sans attaquants... improbable mais possible en data sale)
    flat_df.fillna(0, inplace=True)
    
    return flat_df

def build_features_v2(team_home, team_away, player_home, player_away):
    print("Construction des features V2 (Position-Aware)...")
    
    # 1. Agrégation fine par position
    p_home_agg = aggregate_players_by_position(player_home, 'P_HOME')
    p_away_agg = aggregate_players_by_position(player_away, 'P_AWAY')
    
    # 2. Merge Team + Players
    df = team_home.merge(team_away, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    
    # 3. Delta Features (Uniquement sur les colonnes les plus importantes pour ne pas exploser la RAM)
    # On se concentre sur les totaux d'équipe et les moyennes par position
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # On cherche les paires HOME/AWAY
    base_features = set([c.replace('_HOME', '') for c in numeric_cols if '_HOME' in c])
    
    print(f"Création des deltas sur {len(base_features)} variables...")
    for col in base_features:
        col_h = f"{col}_HOME"
        col_a = f"{col}_AWAY"
        if col_h in df.columns and col_a in df.columns:
            df[f'DELTA_{col}'] = df[col_h] - df[col_a]
    
    # 4. Nettoyage
    drop_cols = ['LEAGUE_HOME', 'TEAM_NAME_HOME', 'LEAGUE_AWAY', 'TEAM_NAME_AWAY', 
                 'LEAGUE', 'TEAM_NAME']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.fillna(0, inplace=True)
    
    return df
# =============================================================================
# 3. OPTIMISATION HYPERPARAMÈTRES (OPTUNA)
# =============================================================================

def objective_lgb(trial, X, y):
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_class': 3,
        'n_estimators': 1000, # Fixed high number with early stopping usually
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42
    }
    
    # 3-Fold CV pour aller vite lors de l'optimisation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    estimator = cast(BaseEstimator, lgb.LGBMClassifier(**param))
    scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# =============================================================================
# 4. PIPELINE PRINCIPAL
# =============================================================================

# --- A. Chargement ---
(xt_h, xt_a, xp_h, xp_a, y_train_raw, y_supp, 
 xtest_h, xtest_a, xpt_h, xpt_a) = load_data()

# --- B. Construction du Dataset ---
X_train = build_features_v2(xt_h, xt_a, xp_h, xp_a)
X_test = build_features_v2(xtest_h, xtest_a, xpt_h, xpt_a)

# Alignement des colonnes (au cas où certaines manquent dans le test)
X_train, X_test = X_train.align(X_test, join='inner', axis=1)

# Préparation de la Target (Classification)
# Conversion de [HOME_WINS, DRAW, AWAY_WINS] en [2, 1, 0]
# Attention l'ordre des colonnes est important. 
# Si Y_train colonnes sont: ID, HOME_WINS, DRAW, AWAY_WINS
# idxmax va donner le nom de la colonne.
y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
le = LabelEncoder()
y_train_cls = le.fit_transform(y_classes) # Mappe: AWAY_WINS->0, DRAW->1, HOME_WINS->2

# Préparation de la Target (Régression Auxiliaire - Différence de buts)
y_train_reg = y_supp['GOAL_DIFF_HOME_AWAY']

ID_test = X_test['ID'] # Sauvegarde pour la soumission
X_train = X_train.drop(columns=['ID'])
X_test = X_test.drop(columns=['ID'])

print(f"Shape finale Train: {X_train.shape}")

# --- C. ÉTAPE CLÉ : Prédiction Auxiliaire (Goal Diff) ---
# On entraîne un modèle pour prédire la différence de buts, et on utilise 
# cette prédiction comme une feature très forte pour savoir qui gagne.
print("Entraînement du modèle auxiliaire (Goal Difference)...")

regressor = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)

# Pour éviter le data leakage, on devrait faire cela en Out-of-Fold (OOF) sur le train
# mais pour simplifier ici, on entraîne sur tout le train (risque léger d'overfit)
# Idéalement : utiliser cross_val_predict pour X_train
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
from typing import cast

# Feature 'PRED_GOAL_DIFF' pour le Train (via Cross Validation pour éviter fuite)
X_train['PRED_GOAL_DIFF'] = cross_val_predict(cast(BaseEstimator, regressor), X_train, y_train_reg, cv=5, n_jobs=-1)

# Feature 'PRED_GOAL_DIFF' pour le Test (via entraînement complet)
regressor.fit(X_train.drop(columns=['PRED_GOAL_DIFF']), y_train_reg)
X_test['PRED_GOAL_DIFF'] = regressor.predict(X_test)

# --- E. Optimisation (Optionnelle - Commentée pour exécution rapide) ---
# study = optuna.create_study(direction='maximize')
# study.optimize(lambda trial: objective_lgb(trial, X_train, y_train_cls), n_trials=50)
# best_params_lgb = study.best_params
# print("Best Params:", best_params_lgb)

# Paramètres solides par défaut (si on saute Optuna)
params_lgb = {
    'n_estimators': 2000, 'learning_rate': 0.03, 'num_leaves': 20, 
    'colsample_bytree': 0.6, 'subsample': 0.8, 'random_state': 42,
    'n_jobs' :-1,'verbose': -1
}
params_xgb = {
    'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 5, 
    'colsample_bytree': 0.8, 'subsample': 0.8, 'random_state': 42,
    'eval_metric': 'mlogloss','tree_method':'hist','n_jobs': -1
}
params_cat = {
    'iterations': 2000, 'learning_rate': 0.03, 'depth': 6, 
    'verbose': 0, 'random_state': 42,'thread_count': -1,
    'rsm':0.6
}

# --- E. Modélisation Finale (Ensemble Voting) ---
print("Entraînement de l'ensemble final...")

clf1 = lgb.LGBMClassifier(**params_lgb)
clf2 = xgb.XGBClassifier(**params_xgb)
clf3 = CatBoostClassifier(**params_cat)

eclf = VotingClassifier(
    estimators=[
        ('lgb', cast(BaseEstimator, clf1)), 
        ('xgb', cast(BaseEstimator, clf2)), 
        ('cat', cast(BaseEstimator, clf3))
    ],
    voting='soft',
    verbose=True,
    n_jobs=1
)

# Validation Score final
cv_scores = cross_val_score(eclf, X_train, y_train_cls, cv=5, scoring='accuracy')
# --- E.1. ESTIMATION (Est-ce que je suis bon ?) ---
print("--- ÉTAPE 1 : Validation Croisée (Simulation) ---")
# Cela prend du temps, mais ça te dit la vérité sur ton niveau
cv_scores = cross_val_score(eclf, X_train, y_train_cls, cv=5, scoring='accuracy', n_jobs=1)
mon_score_estime = cv_scores.mean()

print(f"📊 Score estimé sur 5 folds : {mon_score_estime:.4f}")

# --- E.2. DÉCISION ---
# Si le score est nul, on arrête tout pour ne pas perdre de temps
if mon_score_estime < 0.475: # Par exemple, ton seuil mini
    print("❌ Score trop bas. J'arrête là. Il faut changer les paramètres.")
    exit() # On coupe le script
else:
    print("✅ Score validé ! On lance l'entraînement final pour la soumission.")

# --- E.3. ENTRAÎNEMENT FINAL (Pour la soumission) ---
print("--- ÉTAPE 2 : Entraînement Final (100% des données) ---")
eclf.fit(X_train, y_train_cls)

# --- F. Prédiction et Soumission ---
print("Génération des prédictions...")
probs = eclf.predict_proba(X_test)

# Reconstruction du format de soumission (HOME_WINS, DRAW, AWAY_WINS)
# Rappel mapping LE: 0=AWAY, 1=DRAW, 2=HOME
submission = pd.DataFrame({
    'ID': ID_test,
    'HOME_WINS': probs[:, 2], # Proba classe 2
    'DRAW': probs[:, 1],      # Proba classe 1
    'AWAY_WINS': probs[:, 0]  # Proba classe 0
})

# --- GESTION DE LA SAUVEGARDE INTELLIGENTE ---

# 1. On rassemble tous les paramètres utilisés pour ne rien oublier
experiment_params = {
    "lgb": params_lgb,
    "xgb": params_xgb,
    "cat": params_cat,
    "feature_engineering": "V2 (Position Aware) - Full Features (No selection)", # Change ça à la main selon ce que tu testes
    "stacking": False,
    "voting": "Soft Voting"
}

# 2. Le score CV (Si tu l'as calculé via cross_val_score plus tôt, utilise la variable)
# Si tu ne l'as pas calculé pour gagner du temps, mets une estimation ou 0
current_cv_score = cv_scores.mean() # Remplace par la variable 'cv_scores.mean()' si dispo

# 3. Appel magique
save_experiment(
    submission_df=submission,
    cv_score=current_cv_score,
    params_dict=experiment_params,
    description="Test avec features V2 et params robustes anti-overfit"
)
# --- G. Conversion Binaire (Hard Voting) ---
print("Conversion en prédictions binaires (0/1)...")
cols_submit = ['HOME_WINS', 'DRAW', 'AWAY_WINS']
# On trouve l'index de la valeur max pour chaque ligne
max_indices = submission[cols_submit].values.argmax(axis=1)
# On crée une matrice de zéros
hard_preds = np.zeros(submission[cols_submit].shape, dtype=int)
# On met des 1 là où la proba était maximale
hard_preds[np.arange(len(submission)), max_indices] = 1
# On remplace dans le DataFrame
submission[cols_submit] = hard_preds

# --- H. Sauvegarde avec Versioning Automatique ---
# Création du dossier de soumission s'il n'existe pas
os.makedirs('submission', exist_ok=True)

# Détermination du numéro de version
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

filename = f'submission_V{version}.csv'
submission_path = os.path.join('submission', filename)

submission.to_csv(submission_path, index=False)
print(f"Fichier binaire généré avec succès : '{submission_path}'")

