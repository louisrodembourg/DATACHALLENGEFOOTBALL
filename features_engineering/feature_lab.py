import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

try:
    from football import load_data
except ImportError:
    print("⚠️ Main.py introuvable.")

warnings.filterwarnings('ignore')

# --- FONCTION QUI GÉNÈRE 10 000 COLONNES (Niveau 2) ---
def aggregate_players_advanced(player_df, prefix):
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' not in numeric_cols: numeric_cols.append('ID')
    
    # On garde MAX et STD car c'est là que se cache l'info "Star Player"
    aggs = ['mean', 'sum', 'max', 'std']
    
    cols_to_use = numeric_cols + ['POSITION']
    pivot_df = player_df[cols_to_use].groupby(['ID', 'POSITION']).agg(aggs)
    pivot_df.columns = [f'{c[0]}_{c[1]}' for c in pivot_df.columns]
    flat_df = pivot_df.unstack(level='POSITION')
    flat_df.columns = [f'{prefix}_{pos}_{col}' for col, pos in flat_df.columns]
    flat_df.reset_index(inplace=True)
    flat_df.fillna(0, inplace=True)
    return flat_df

def build_features_experimental(team_home, team_away, player_home, player_away):
    print("\n--- 🛠 CONSTRUCTION FEATURES V4 (Debug NaN) ---")
    
    # 1. Génération features
    p_home_agg = aggregate_players_advanced(player_home, 'P_HOME')
    p_away_agg = aggregate_players_advanced(player_away, 'P_AWAY')
    
    df = team_home.merge(team_away, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    
    print(f"📊 Dimensions BRUTES : {df.shape}")

    # --- 🚨 FIX CRITIQUE : ON REMPLACE LES NaN PAR 0 TOUT DE SUITE ---
    # C'est ça qui bloquait le nettoyage. Maintenant les 'trous' sont des vrais 0.
    df.fillna(0, inplace=True)
    print("   ✅ NaN remplacés par 0.")

    # 2. Ratios (Niveau 3)
    try:
        df['HOME_EFFICIENCY'] = df['TEAM_GOALS_season_sum_HOME'] / (df['TEAM_SHOTS_TOTAL_season_sum_HOME'] + 1)
        df['AWAY_EFFICIENCY'] = df['TEAM_GOALS_season_sum_AWAY'] / (df['TEAM_SHOTS_TOTAL_season_sum_AWAY'] + 1)
        df['DELTA_EFFICIENCY'] = df['HOME_EFFICIENCY'] - df['AWAY_EFFICIENCY']
    except KeyError:
        pass

    # 3. Deltas Classiques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_features = set([c.replace('_HOME', '') for c in numeric_cols if '_HOME' in c])
    for col in base_features:
        col_h, col_a = f"{col}_HOME", f"{col}_AWAY"
        if col_h in df.columns and col_a in df.columns:
            df[f'DELTA_{col}'] = df[col_h] - df[col_a]

    # =========================================================================
    # 🧹 NIVEAU 4 : LE GRAND NETTOYAGE (Maintenant ça va marcher)
    # =========================================================================
    print("👉 Démarrage du Nettoyage Agressif...")
    n_start = df.shape[1]
    
    # On ne travaille que sur les nombres pour éviter les erreurs
    # A. Virer les colonnes constantes (strictement 0 partout)
    # On utilise loc pour modifier le df en place
    df = df.loc[:, (df != 0).any(axis=0)]
    n_step1 = df.shape[1]
    print(f"   - Constantes (0 strict) supprimées : {n_start - n_step1}")

    # B. Virer les quasi-vides (>99.5% de zéros)
    # Avec fillna(0) fait au début, ça va enfin détecter les colonnes vides
    zero_fraction = (df == 0).mean()
    df = df.loc[:, zero_fraction < 0.995]
    n_step2 = df.shape[1]
    print(f"   - Quasi-vides (>99.5% 0) supprimées : {n_step1 - n_step2}")

    # C. Doublons
    df = df.loc[:, ~df.columns.duplicated()]
    
    print(f"✅ Dimensions FINALES : {df.shape}")

    # Nettoyage final ID/Text
    drop_cols = ['LEAGUE_HOME', 'TEAM_NAME_HOME', 'LEAGUE_AWAY', 'TEAM_NAME_AWAY', 'LEAGUE', 'TEAM_NAME']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    return df

def fast_evaluate():
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, _, _, _, _, _) = load_data()
    X = build_features_experimental(xt_h, xt_a, xp_h, xp_a)
    
    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y = le.fit_transform(y_classes)
    
    if 'ID' in X.columns: X = X.drop(columns=['ID'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n🚀 Test LightGBM sur {X.shape[1]} features...")
    # On booste un peu le LightGBM pour qu'il gère le volume
    clf = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.04, num_leaves=40, n_jobs=-1, verbose=-1, random_state=42)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    print(f"\n🎯 SCORE EXPERIMENTAL : {accuracy_score(y_val, clf.predict(X_val)):.5f}")

if __name__ == "__main__":
    fast_evaluate()