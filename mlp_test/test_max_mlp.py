import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings

try:
    from football import load_data
except ImportError:
    print("⚠️ Main.py introuvable.")

warnings.filterwarnings('ignore')

def aggregate_players_by_position(player_df, prefix):
    # 1. Sécurité Majuscules (Force tout en MAJ)
    player_df.columns = [c.upper() for c in player_df.columns]
    
    # 🚨 FIX CRITIQUE : Suppression des doublons de noms de colonnes
    # (Si on a 'position' et 'POSITION', cela garde un seul 'POSITION')
    player_df = player_df.loc[:, ~player_df.columns.duplicated()]

    # 2. Détective : Recherche de la colonne POSITION (si elle manque encore)
    if 'POSITION' not in player_df.columns:
        candidates = [c for c in player_df.columns if 'POS' in c or 'ROLE' in c]
        if candidates:
            # On renomme la première trouvée
            player_df.rename(columns={candidates[0]: 'POSITION'}, inplace=True)
        else:
            raise KeyError(f"Colonne POSITION introuvable. Colonnes : {list(player_df.columns)}")

    # 3. Agrégation standard
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' not in numeric_cols: numeric_cols.append('ID')
    
    # On s'assure de ne pas avoir de doublons dans la liste des colonnes à utiliser
    cols_to_use = list(set(numeric_cols + ['POSITION'])) 
    
    pivot_df = player_df[cols_to_use].groupby(['ID', 'POSITION']).agg(['mean', 'sum'])
    
    pivot_df.columns = [f'{c[0]}_{c[1]}' for c in pivot_df.columns]
    flat_df = pivot_df.unstack(level='POSITION')
    flat_df.columns = [f'{prefix}_{pos}_{col}' for col, pos in flat_df.columns]
    flat_df.reset_index(inplace=True)
    flat_df.fillna(0, inplace=True)
    
    return flat_df

def build_features_v2(team_home, team_away, player_home, player_away):
    print("--- Construction des features ---")
    p_home_agg = aggregate_players_by_position(player_home, 'P_HOME')
    p_away_agg = aggregate_players_by_position(player_away, 'P_AWAY')
    df = team_home.merge(team_away, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    df.fillna(0, inplace=True)

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

    try:
        c_shot_h = [c for c in df.columns if 'P_HOME' in c and 'FORWARD' in c and 'SHOTS' in c and 'TARGET' in c and 'sum' in c]
        c_save_a = [c for c in df.columns if 'P_AWAY' in c and 'GOALKEEPER' in c and 'SAVES' in c and 'sum' in c]
        if c_shot_h and c_save_a: df['DUEL_ATT_H_GK_A'] = df[c_shot_h[0]] - df[c_save_a[0]]
        c_shot_a = [c for c in df.columns if 'P_AWAY' in c and 'FORWARD' in c and 'SHOTS' in c and 'TARGET' in c and 'sum' in c]
        c_save_h = [c for c in df.columns if 'P_HOME' in c and 'GOALKEEPER' in c and 'SAVES' in c and 'sum' in c]
        if c_shot_a and c_save_h: df['DUEL_ATT_A_GK_H'] = df[c_shot_a[0]] - df[c_save_h[0]]

        col_duel = 'PLAYER_DUELS_WON'
        cols_duels_h = [c for c in df.columns if 'P_HOME' in c and col_duel in c and 'sum' in c]
        if cols_duels_h: 
            cols_duels_a = [c for c in df.columns if 'P_AWAY' in c and col_duel in c and 'sum' in c]
            df['PHYSICAL_DOMINANCE'] = df[cols_duels_h].sum(axis=1) - df[cols_duels_a].sum(axis=1)

        col_key = 'PLAYER_KEY_PASSES'
        cols_key_h = [c for c in df.columns if 'P_HOME' in c and col_key in c and 'sum' in c]
        if cols_key_h:
            cols_key_a = [c for c in df.columns if 'P_AWAY' in c and col_key in c and 'sum' in c]
            df['CREATIVITY_DIFF'] = df[cols_key_h].sum(axis=1) - df[cols_key_a].sum(axis=1)
    except: pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_features = set([c.replace('_HOME', '') for c in numeric_cols if '_HOME' in c])
    for col in base_features:
        col_h, col_a = f"{col}_HOME", f"{col}_AWAY"
        if col_h in df.columns and col_a in df.columns: df[f'DELTA_{col}'] = df[col_h] - df[col_a]

    df = df.loc[:, (df != 0).any(axis=0)]
    zeros = (df == 0).mean()
    df = df.loc[:, zeros < 0.995]
    df = df.loc[:, ~df.columns.duplicated()]
    cols_to_drop = df.select_dtypes(include=['object']).columns
    if len(cols_to_drop) > 0: df.drop(columns=cols_to_drop, inplace=True)
    drop_cols = ['LEAGUE_HOME', 'TEAM_NAME_HOME', 'LEAGUE_AWAY', 'TEAM_NAME_AWAY', 'LEAGUE', 'TEAM_NAME']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    return df

def run_pure_mlp():
    print("🚀 DÉMARRAGE DU RUN 'PURE MLP SNIPER' (SMART LOADER) 🚀")
    
    # 1. CHARGEMENT INTELLIGENT (Le cœur du fix)
    raw_data = load_data()
    print(f"📦 Données reçues : {len(raw_data)} éléments.")

    # On prépare les boîtes
    train_team = []   # ~12 303
    test_team = []    # ~25 368
    train_player = [] # ~270 000
    test_player = []  # ~500 000
    y_target = None

    # On trie les valises selon leur taille !
    for item in raw_data:
        # On ignore les listes d'IDs seules, on veut les DataFrames
        if not hasattr(item, 'shape'): continue
        
        rows = item.shape[0]
        
        if rows < 15000: # C'est le TRAIN TEAM (~12k)
            # On vérifie si c'est la Target (qui contient 'HOME_WINS')
            if hasattr(item, 'columns') and 'HOME_WINS' in item.columns:
                y_target = item
            elif len(item.shape) == 2: # C'est un DataFrame d'équipe
                train_team.append(item)
                
        elif 20000 <= rows < 50000: # C'est le TEST TEAM (~25k)
            if len(item.shape) == 2:
                test_team.append(item)
                
        elif 200000 <= rows < 400000: # C'est TRAIN PLAYER (~270k)
             train_player.append(item)
             
        elif rows >= 400000: # C'est TEST PLAYER (~500k)
             test_player.append(item)

    print(f"📊 Détection : {len(train_team)} TrainTeams, {len(test_team)} TestTeams.")

    # Assignation (On suppose que le premier est Home, le second Away)
    xt_h, xt_a = train_team[0], train_team[1]
    xp_h, xp_a = train_player[0], train_player[1]
    xt_test_h, xt_test_a = test_team[0], test_team[1]
    xp_test_h, xp_test_a = test_player[0], test_player[1]
    y_train_raw = y_target
    
    # On récupère les IDs propres depuis le DataFrame Test identifié
    if 'ID' in xt_test_h.columns:
        test_id = xt_test_h['ID'].values
    else:
        test_id = xt_test_h.index.values # Au cas où
    
    print("✅ Données remises dans le bon ordre !")

    # --- FIX MAJUSCULES & CLEAN ---
    def clean_cols(df):
        df.columns = [c.upper() for c in df.columns]
        return df.loc[:, ~df.columns.duplicated()]

    xt_h, xt_a = clean_cols(xt_h), clean_cols(xt_a)
    xp_h, xp_a = clean_cols(xp_h), clean_cols(xp_a)
    xt_test_h, xt_test_a = clean_cols(xt_test_h), clean_cols(xt_test_a)
    xp_test_h, xp_test_a = clean_cols(xp_test_h), clean_cols(xp_test_a)

    # --- INJECTION FORCÉE ID (Maintenant ça va marcher car les tailles matchent) ---
    print("💉 Injection des IDs pour garantir le merge...")
    
    # On force les IDs du Test à être identiques
    xt_test_h['ID'] = test_id
    xt_test_a['ID'] = test_id
    
    # Typage int
    xt_test_h['ID'] = xt_test_h['ID'].astype(int)
    xt_test_a['ID'] = xt_test_a['ID'].astype(int)
    
    # Pour le Train, on sécurise aussi
    if 'ID' not in xt_h.columns: xt_h['ID'] = xt_h.index
    if 'ID' not in xt_a.columns: xt_a['ID'] = xt_h.index
    xt_h['ID'] = xt_h['ID'].astype(int)
    xt_a['ID'] = xt_a['ID'].astype(int)

    # 2. FEATURES
    print("Construction X_train...")
    X_train = build_features_v2(xt_h, xt_a, xp_h, xp_a)
    print(f"   -> Dim Train : {X_train.shape}")
    
    print("Construction X_test...")
    X_test = build_features_v2(xt_test_h, xt_test_a, xp_test_h, xp_test_a)
    print(f"   -> Dim Test : {X_test.shape}")
    
    if X_test.shape[0] == 0:
        raise ValueError("ERREUR : X_test vide après injection ID ! Problème structurel.")

    # 3. TARGET
    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y_train = le.fit_transform(y_classes)
    
    # 4. NETTOYAGE CORRÉLATION
    print("Nettoyage Corrélation...")
    if 'ID' in X_train.columns: X_train.drop(columns=['ID'], inplace=True)
    
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    X_train.drop(columns=to_drop, inplace=True)
    
    # 5. ALIGNEMENT
    print("Alignement Final...")
    final_features = X_train.columns.tolist()
    X_test = X_test.reindex(columns=final_features, fill_value=0)
    print(f"✅ Prêt pour le tir : Train={X_train.shape}, Test={X_test.shape}")

    # 6. MODÈLE SNIPER
    print("🧠 Entraînement Neurone (Sniper)...")
    model = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=0),
        StandardScaler(),
        PCA(n_components=150, random_state=42),
        MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), # 3 couches pour digérer l'info
            activation='relu',
            solver='adam',
            alpha=0.05,                   # Régularisation moyenne
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=4,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=42,
            verbose=True
        )
    )
    
    model.fit(X_train, y_train)
    
    # 7. SOUMISSION
    print("🔮 Génération soumission...")
    probs = model.predict_proba(X_test)
    
    submission = pd.DataFrame(probs, columns=['AWAY_WINS', 'DRAW', 'HOME_WINS'])
    submission['ID'] = test_id
    submission = submission[['ID', 'HOME_WINS', 'DRAW', 'AWAY_WINS']]
    
    filename = "submission_mlp_sniper.csv"
    submission.to_csv(filename, index=False)
    print(f"✅ SUCCÈS ! Fichier '{filename}' généré !")

def convert_submission_to_one_hot(filename):
    print(f"🔄 Conversion {filename} en format 0/1 (Hard Voting)...")
    try:
        df = pd.read_csv(filename)
        cols = ['HOME_WINS', 'DRAW', 'AWAY_WINS']
        
        # On s'assure que les colonnes existent
        if not all(col in df.columns for col in cols):
            print(f"⚠️ Colonnes manquantes dans {filename}. Colonnes trouvées : {df.columns}")
            return

        # On trouve la colonne avec la proba maximale pour chaque ligne
        max_col = df[cols].idxmax(axis=1)
        
        # On remplace par 1 si c'est le max, 0 sinon
        for col in cols:
            df[col] = (max_col == col).astype(int)
            
        df.to_csv(filename, index=False)
        print(f"✅ Fichier converti en 0/1 : {filename}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la conversion : {e}")

if __name__ == "__main__":
    #run_pure_mlp()
    convert_submission_to_one_hot("submission_mlp_sniper.csv")
