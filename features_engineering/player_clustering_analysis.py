
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data import load_data

def analyze_player_clusters():
    print("--- Loading Data ---")
    (xt_h, xt_a, xp_h, xp_a, _, _, _, _, _, _) = load_data()
    
    # Combine Home and Away players to have more data for clustering
    # Only picking numeric stats
    # We drop ID and POSITION for the clustering itself, but keep them for interpretation
    
    print("--- Preparing Player Data ---")
    
    # Common columns (intersection of home and away if differs, usually identical)
    cols_h = set(xp_h.columns)
    cols_a = set(xp_a.columns)
    common_cols = list(cols_h.intersection(cols_a))
    
    # Stack everything
    players = pd.concat([xp_h[common_cols], xp_a[common_cols]], axis=0, ignore_index=True)
    
    print(f"Total Players Rows: {players.shape[0]}")
    
    # Filter numeric columns for clustering
    numeric_cols = players.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' in numeric_cols: numeric_cols.remove('ID')
    
    # Handle NaNs
    players_numeric = players[numeric_cols].fillna(0)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(players_numeric)
    
    # Clustering K-Means
    # Let's try K=5 (Goalkeeper, Defender, Midfielder, Attacker, + ?) to K=8
    k = 6
    print(f"--- Running K-Means (K={k}) ---")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    players['CLUSTER'] = clusters
    
    # Analysis of Clusters
    print("\n--- Cluster Interpretations ---")
    
    # Identify what distinguishes each cluster
    # We compare the mean of each cluster to the global mean
    global_mean = players_numeric.mean()
    cluster_means = players.groupby('CLUSTER')[numeric_cols].mean()
    
    # Standard deviation to see relevance? 
    # Or just Z-Score of the center relative to global distribution
    
    # Display Top Features for each Cluster
    for i in range(k):
        print(f"\nExample Features for Cluster {i}:")
        # Sort features by how much they deviate from the global mean (normalized difference)
        # diff = (cluster_mean - global_mean) / global_std
        mean_diff = (cluster_means.loc[i] - global_mean) / players_numeric.std()
        
        # Get top positive and negative distinguishing features
        top_features = mean_diff.abs().sort_values(ascending=False).head(10)
        
        # Check Position composition if available
        if 'POSITION' in players.columns:
            pos_counts = players[players['CLUSTER'] == i]['POSITION'].value_counts(normalize=True)
            top_pos = pos_counts.head(3).to_dict()
            print(f"  Dominant Positions: {top_pos}")
            
        print(f"  Distinguishing Stats (Z-Score):")
        for feature in top_features.index:
            score = mean_diff[feature]
            val = cluster_means.loc[i, feature]
            print(f"    - {feature}: {score:.2f} (Avg Val: {val:.2f})")

if __name__ == "__main__":
    analyze_player_clusters()
