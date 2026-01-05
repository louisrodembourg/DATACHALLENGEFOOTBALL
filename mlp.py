import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

try:
    from football import load_data
except ImportError:
    print("⚠️ Main.py introuvable.")
    load_data = None

warnings.filterwarnings('ignore')

# =============================================================================
# 1. FEATURES "CHAMPIONNES" (Config 0.499)
# =============================================================================

def aggregate_players_by_position(player_df, prefix):
    # Agrégation Simple (Mean/Sum)
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
    print("--- Construction des features (Base 0.499) ---")
    
    # Agrégation
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

    # Smart Deltas (Physique & Créativité - SANS Midfield)
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

    # Nettoyage Basique
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
# 2. LE TEST DU NEURONE STABILISÉ
# =============================================================================

def test_mlp():
    print("\n--- 🧠 MLP V3 (PCA + ENTONNOIR) : STOP SI VAL > 0.499 + PREDICT X_TEST ---")

    if load_data is None:
        raise ImportError("Impossible d'importer load_data depuis football.py")

    # 1. Chargement & Features (train + test)
    (
        x_train_team_home,
        x_train_team_away,
        x_train_player_home,
        x_train_player_away,
        y_train_raw,
        _,
        x_test_team_home,
        x_test_team_away,
        x_test_player_home,
        x_test_player_away,
    ) = load_data()

    X = build_features_v2(x_train_team_home, x_train_team_away, x_train_player_home, x_train_player_away)
    X_test = build_features_v2(x_test_team_home, x_test_team_away, x_test_player_home, x_test_player_away)
    test_id = x_test_team_home['ID'] if 'ID' in x_test_team_home.columns else X_test.get('ID', None)

    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y = le.fit_transform(y_classes)

    if 'ID' in X.columns:
        X = X.drop(columns=['ID'])
    if 'ID' in X_test.columns:
        X_test = X_test.drop(columns=['ID'])

    # 2. Nettoyage Corrélation (appliqué aussi au test)
    print("Nettoyage Corrélation initial...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X.drop(columns=[c for c in to_drop if c in X.columns], inplace=True)
    X_test.drop(columns=[c for c in to_drop if c in X_test.columns], inplace=True)

    # Alignement strict des colonnes entre train et test
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    print(f"Features avant PCA : {X.shape[1]}")

    # 3. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Préprocessing (Imputer + Scaler + PCA) + MLP identique
    print("\n🛠 Configuration : PCA(150) + MLP(128,64,32) + boucle (partial_fit)...")
    preprocessor = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=0),
        StandardScaler(),
        PCA(n_components=150, random_state=42),
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.3,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1,
        warm_start=True,
        early_stopping=False,
        random_state=42,
        verbose=False,
    )

    threshold = 0.499
    max_epochs = 50
    classes_ = np.unique(y_train)
    best_score = -1.0
    best_model = None
    chosen_model = None

    print(f"🚀 Entraînement : arrêt dès que val_acc > {threshold}...")
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            mlp.partial_fit(X_train_t, y_train, classes=classes_)
        else:
            mlp.partial_fit(X_train_t, y_train)

        val_preds = mlp.predict(X_val_t)
        val_score = accuracy_score(y_val, val_preds)
        print(f"Epoch {epoch}/{max_epochs} - val_acc={val_score:.5f}")

        if val_score > best_score:
            best_score = val_score
            best_model = deepcopy(mlp)

        if val_score > threshold:
            chosen_model = deepcopy(mlp)
            print(f"✅ Seuil atteint : val_acc={val_score:.5f} (> {threshold})")
            break

    if chosen_model is None:
        chosen_model = best_model
        print(f"⚠️ Seuil non atteint. On garde le meilleur modèle (val_acc={best_score:.5f}).")

    if chosen_model is None:
        raise RuntimeError("Aucun modèle entraîné (chosen_model est None).")

    # 5. Prédiction sur X_test + export one-hot 0/1
    test_pred_int = chosen_model.predict(X_test_t)
    test_pred_label = le.inverse_transform(test_pred_int)
    out_cols = ['HOME_WINS', 'DRAW', 'AWAY_WINS']

    submission = pd.DataFrame({'ID': test_id})
    for c in out_cols:
        submission[c] = 0
    for c in out_cols:
        submission.loc[test_pred_label == c, c] = 1

    filename = 'submission_mlp_stop0499_onehot.csv'
    submission.to_csv(filename, index=False)
    print(f"✅ Soumission générée (0/1) : {filename}")

if __name__ == "__main__":
    test_mlp()