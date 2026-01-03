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

# =============================================================================
# 🎛️ TABLE DE MIXAGE V3 (TARGETED FEATURES)
# =============================================================================

# 1. DEEP QUALITY (Validé au tour précédent)
USE_DEEP_QUALITY = True

# 2. CROSS DELTAS "INTELLIGENTS" (C'est ça qu'on teste maintenant)
# Utilise Accurate Passes, Duels Won, Key Passes
USE_SMART_DELTAS = True

# 3. NETTOYAGE
USE_BASIC_CLEANING = True
USE_CORR_CLEANING  = True # Laisse à False pour l'instant pour aller vite

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def aggregate_players_simple(player_df, prefix):
    # Agrégation MEAN + SUM (La gagnante)
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

# =============================================================================
# PIPELINE DE CONSTRUCTION
# =============================================================================

def build_features_experimental(team_home, team_away, player_home, player_away):
    print("\n--- 🧪 LABO V3 : FEATURES CIBLÉES ---")
    
    # 1. Agrégation
    p_home_agg = aggregate_players_simple(player_home, 'P_HOME')
    p_away_agg = aggregate_players_simple(player_away, 'P_AWAY')

    # Merge
    df = team_home.merge(team_away, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    df.fillna(0, inplace=True)
    
    # -------------------------------------------------------------------------
    # 🎯 NIVEAU 3 : DEEP QUALITY (Validé)
    # -------------------------------------------------------------------------
    if USE_DEEP_QUALITY:
        try:
            # Efficacité (Buts / Tirs)
            df['HOME_EFFICIENCY'] = df['TEAM_GOALS_season_sum_HOME'] / (df['TEAM_SHOTS_TOTAL_season_sum_HOME'] + 1)
            df['AWAY_EFFICIENCY'] = df['TEAM_GOALS_season_sum_AWAY'] / (df['TEAM_SHOTS_TOTAL_season_sum_AWAY'] + 1)
            
            # Précision (Cadrés / Total) - Noms supposés d'après tes logs précédents
            s_on_h = 'TEAM_SHOTS_ON_TARGET_season_sum_HOME'
            s_tot_h = 'TEAM_SHOTS_TOTAL_season_sum_HOME'
            if s_on_h in df.columns:
                df['HOME_ACCURACY'] = df[s_on_h] / (df[s_tot_h] + 1)
                df['AWAY_ACCURACY'] = df['TEAM_SHOTS_ON_TARGET_season_sum_AWAY'] / (df['TEAM_SHOTS_TOTAL_season_sum_AWAY'] + 1)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # ⚔️ NIVEAU 1 v2 : SMART DELTAS (Avec tes vrais noms de colonnes)
    # -------------------------------------------------------------------------
    if USE_SMART_DELTAS:
        print("👉 Ajout Smart Deltas (Passes, Duels, Créativité)...")
        
        # A. DOMINATION DU MILIEU (Midfield Control)
        # On compare les passes RÉUSSIES (Accurate) des milieux
        # Nom reconstruit : P_HOME + MIDFIELDER + PLAYER_ACCURATE_PASSES_season_sum + sum
        # Attention : aggregate a ajouté '_mean' et '_sum' à la fin de tes noms de base
        
        col_name_pass = 'PLAYER_ACCURATE_PASSES_season_sum_sum' # On prend la somme des sommes
        
        h_mid_pass = f'P_HOME_MIDFIELDER_{col_name_pass}'
        a_mid_pass = f'P_AWAY_MIDFIELDER_{col_name_pass}'
        
        if h_mid_pass in df.columns:
            df['MIDFIELD_DOMINATION'] = df[h_mid_pass] - df[a_mid_pass]
            print("   ✅ Midfield Domination (Accurate Passes) ajouté.")
        else:
            print(f"   ⚠️ Midfield Pass introuvable (Cherché: {h_mid_pass})")

        # B. IMPACT PHYSIQUE (Duels & Interceptions)
        # On compare la somme totale des duels gagnés par l'équipe
        # Pour simplifier, on somme toutes les positions (G+D+M+F) ou on prend juste les totaux si dispo
        # Ici on fait : Somme Duels Gagnés Home - Somme Duels Gagnés Away (Toutes positions)
        
        col_name_duel = 'PLAYER_DUELS_WON_season_sum_sum'
        
        # On recrée une somme globale manuellement car c'est éparpillé par position
        # Astuce : On prend toutes les colonnes qui finissent par ce nom
        cols_duels_h = [c for c in df.columns if 'P_HOME' in c and col_name_duel in c]
        cols_duels_a = [c for c in df.columns if 'P_AWAY' in c and col_name_duel in c]
        
        if cols_duels_h:
            df['HOME_TOTAL_DUELS_WON'] = df[cols_duels_h].sum(axis=1)
            df['AWAY_TOTAL_DUELS_WON'] = df[cols_duels_a].sum(axis=1)
            df['PHYSICAL_DOMINANCE'] = df['HOME_TOTAL_DUELS_WON'] - df['AWAY_TOTAL_DUELS_WON']
            print("   ✅ Physical Dominance (Duels Won) ajouté.")

        # C. CRÉATIVITÉ (Key Passes)
        # Delta des passes clés
        col_name_key = 'PLAYER_KEY_PASSES_season_sum_sum'
        cols_key_h = [c for c in df.columns if 'P_HOME' in c and col_name_key in c]
        cols_key_a = [c for c in df.columns if 'P_AWAY' in c and col_name_key in c]
        
        if cols_key_h:
            df['CREATIVITY_DIFF'] = df[cols_key_h].sum(axis=1) - df[cols_key_a].sum(axis=1)
            print("   ✅ Creativity Diff (Key Passes) ajouté.")

    # Deltas Classiques (Base)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_features = set([c.replace('_HOME', '') for c in numeric_cols if '_HOME' in c])
    for col in base_features:
        col_h, col_a = f"{col}_HOME", f"{col}_AWAY"
        if col_h in df.columns and col_a in df.columns:
            df[f'DELTA_{col}'] = df[col_h] - df[col_a]

    # -------------------------------------------------------------------------
    # 🧹 NETTOYAGE
    # -------------------------------------------------------------------------
    if USE_BASIC_CLEANING:
        print("👉 Nettoyage...")
        df = df.loc[:, (df != 0).any(axis=0)]
        zeros = (df == 0).mean()
        df = df.loc[:, zeros < 0.995]
        df = df.loc[:, ~df.columns.duplicated()]
    
    if USE_CORR_CLEANING:
        # Copie le bloc du message précédent si tu veux le réactiver
        pass 
    
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
    
    print(f"🚀 Test LightGBM sur {X.shape[1]} features...")
    clf = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.04, num_leaves=31, n_jobs=-1, verbose=-1, random_state=42)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    print(f"\n🎯 SCORE : {accuracy_score(y_val, clf.predict(X_val)):.5f}")

if __name__ == "__main__":
    fast_evaluate()