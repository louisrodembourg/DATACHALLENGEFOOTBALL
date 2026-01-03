import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

try:
    from football import load_data
except ImportError:
    print("⚠️ Main.py introuvable.")

warnings.filterwarnings('ignore')

# =============================================================================
# TA CONFIGURATION GAGNANTE (Features 0.499)
# =============================================================================

def aggregate_players_by_position(player_df, prefix):
    # Version SIMPLE (Mean/Sum)
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' not in numeric_cols: numeric_cols.append('ID')
    cols_to_use = numeric_cols + ['POSITION']
    pivot_df = player_df[cols_to_use].groupby(['ID', 'POSITION']).agg(['mean', 'sum'])
    pivot_df.columns = [f'{c[0]}_{c[1]}' for c in pivot_df.columns]
    flat_df = pivot_df.unstack(level='POSITION')
    flat_df.columns = [f'{prefix}_{pos}_{col}' for col, pos in flat_df.columns]
    flat_df.reset_index(inplace=True)
    flat_df.fillna(0, inplace=True)
    return flat_df

def build_features_v2(team_home, team_away, player_home, player_away):
    print("--- Construction des features (Config 0.499) ---")
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
        s_tot_h = 'TEAM_SHOTS_TOTAL_season_sum_HOME'
        s_on_a = [c for c in df.columns if 'SHOTS_ON_TARGET' in c and '_AWAY' in c and 'TEAM' in c][0]
        s_tot_a = 'TEAM_SHOTS_TOTAL_season_sum_AWAY'
        df['HOME_ACCURACY'] = df[s_on_h] / (df[s_tot_h] + 1)
        df['AWAY_ACCURACY'] = df[s_on_a] / (df[s_tot_a] + 1)
        df['HOME_GK_RESISTANCE'] = 1 - (df['TEAM_GOALS_season_sum_AWAY'] / (df[s_on_a] + 1))
        df['AWAY_GK_RESISTANCE'] = 1 - (df['TEAM_GOALS_season_sum_HOME'] / (df[s_on_h] + 1))
    except: pass

    # Smart Deltas (Physique & Créativité)
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

    # Deltas
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
# TEST DU MLP (RÉSEAU DE NEURONES)
# =============================================================================

def test_mlp():
    print("\n--- 🧠 TEST DU NEURONE (MLP) ISOLE ---")
    
    # 1. Chargement
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, _, _, _, _, _) = load_data()
    X = build_features_v2(xt_h, xt_a, xp_h, xp_a)
    
    # Target
    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y = le.fit_transform(y_classes)
    
    if 'ID' in X.columns: X = X.drop(columns=['ID'])
    
    # 2. Nettoyage Corrélation (Comme dans le main)
    # Important pour ne pas donner 5000 colonnes au MLP, il va chauffer sinon
    print("Nettoyage Corrélation (pour alléger le neurone)...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X.drop(to_drop, axis=1, inplace=True)
    print(f"Features restantes : {X.shape[1]}")

    # 3. Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Définition du Pipeline MLP
    # Scaler -> Imputer (au cas où) -> MLP
    print("\n🛠 Configuration du MLP...")
    mlp_pipe = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=0), # Sécurité NaN
        StandardScaler(),                                 # OBLIGATOIRE
        MLPClassifier(
            hidden_layer_sizes=(64, 32),  # <--- Réduit de moitié
            activation='relu',
            solver='adam',
            alpha=0.1,                    # <--- Régularisation x1000 ! (Force la généralisation)
            batch_size=32,                # <--- Batch plus petit pour apprendre plus souvent
            learning_rate='adaptive',
            learning_rate_init=0.0005,    # <--- Apprentissage plus doucement
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,     # On surveille sur un peu plus de données
            n_iter_no_change=5,           # <--- Patience réduite (on coupe vite si ça dérive)
            random_state=42,
            verbose=True
        )
    )

    print("🚀 Entraînement du neurone...")
    mlp_pipe.fit(X_train, y_train)
    
    print("\nEvaluation...")
    preds = mlp_pipe.predict(X_val)
    score = accuracy_score(y_val, preds)
    
    print(f"\n🧠 SCORE DU NEURONE SEUL : {score:.5f}")
    
    if score > 0.48:
        print("✅ VERDICT : Excellent ! Il mérite sa place dans le Stacking.")
    elif score > 0.475:
        print("⚠️ VERDICT : Moyen. Il peut aider par diversité, mais limite.")
    else:
        print("❌ VERDICT : Mauvais. Ne l'ajoute pas, il va polluer le Stacking.")

if __name__ == "__main__":
    test_mlp()