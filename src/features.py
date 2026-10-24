import pandas as pd
import numpy as np

def aggregate_players_by_position(player_df, prefix):
    """
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
    Combines data, calculates ratios, smart deltas, and performs cleaning.
    """
    print("--- Feature Construction ---")
    
    # Fusions
    p_home_agg = aggregate_players_by_position(player_home, 'P_HOME')
    p_away_agg = aggregate_players_by_position(player_away, 'P_AWAY')
    # Merge team stats with aggregated player stats
    df = team_home.merge(team_away, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    df.fillna(0, inplace=True)

    # Ratios and Efficiencies
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

    # -------------------------------------------------------------------------
    # 🆕 NEW: MOMENTUM FEATURES (Short Term vs Long Term)
    # -------------------------------------------------------------------------
    # Compare "5_last_match_average" vs "season_average" to detect form
    # If 5_last > season => Positive Momentum
    
    momentum_metrics = [
        'TEAM_GOALS', 'TEAM_GAME_WON', 'TEAM_SHOTS_ON_TARGET', 
        'TEAM_DANGEROUS_ATTACKS', 'TEAM_SUCCESSFUL_PASSES_PERCENTAGE',
        'TEAM_CORNERS', 'TEAM_POSSESSION' # Some names might differ, check carefully
    ]
    
    # Auto-detect existing metrics that have both suffixes
    suffix_short = '5_last_match_average'
    suffix_long = 'season_average'

    print("--- Calculating Momentum (Short vs Long Term) ---")
    
    for col in team_home.columns:
        # Check if this column is a "season_average" metric
        if col.endswith(f"_{suffix_long}"):
            metric_base = col.replace(f"_{suffix_long}", "")
            
            # Construct names for current dataframe (merged)
            col_long_home = f"{metric_base}_{suffix_long}_HOME"
            col_short_home = f"{metric_base}_{suffix_short}_HOME"
            
            col_long_away = f"{metric_base}_{suffix_long}_AWAY"
            col_short_away = f"{metric_base}_{suffix_short}_AWAY"

            # Check if short term version exists in df
            if col_short_home in df.columns:
                # HOME Momentum
                df[f'MOMENTUM_{metric_base}_HOME'] = df[col_short_home] - df[col_long_home]
                # Scaled Momentum (relative to season average)
                # df[f'MOMENTUM_PCT_{metric_base}_HOME'] = _safe_div(df[col_short_home] - df[col_long_home], df[col_long_home])

            if col_short_away in df.columns:
                # AWAY Momentum
                df[f'MOMENTUM_{metric_base}_AWAY'] = df[col_short_away] - df[col_long_away]


    # Smart Deltas (Physics & Creativity)
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

    # Classic Deltas (Home - Away)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_features = sorted(list(set([c.replace('_HOME', '') for c in numeric_cols if '_HOME' in c])))
    for col in base_features:
        col_h, col_a = f"{col}_HOME", f"{col}_AWAY"
        if col_h in df.columns and col_a in df.columns: df[f'DELTA_{col}'] = df[col_h] - df[col_a]

    # Targeted Removal
    deltas_to_drop = [
        'DELTA_TEAM_GAME_WON_season_sum',
        'DELTA_TEAM_GAME_LOST_season_sum',
    ]
    df.drop(columns=[c for c in deltas_to_drop if c in df.columns], inplace=True)

    # Basic Cleaning
    df = df.loc[:, (df != 0).any(axis=0)] # Remove columns with all 0s
    zeros = (df == 0).mean()
    df = df.loc[:, zeros < 0.995] # Remove quasi-empty columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    cols_to_drop = df.select_dtypes(include=['object']).columns
    if len(cols_to_drop) > 0: df.drop(columns=cols_to_drop, inplace=True)

    drop_cols = ['LEAGUE_HOME', 'TEAM_NAME_HOME', 'LEAGUE_AWAY', 'TEAM_NAME_AWAY', 'LEAGUE', 'TEAM_NAME']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    return df
