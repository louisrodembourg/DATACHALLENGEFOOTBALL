import pandas as pd
import os

def load_data(base_path='data/'):
    """
    Loads all necessary datasets (Train/Test Teams, Train/Test Players, Targets).
    Assumes `base_path` is relative to the project root.
    """
    if not os.path.exists(base_path):
        # Fallback if running from a subdirectory without proper context (though not recommended)
        if os.path.exists(f'../{base_path}'):
            base_path = f'../{base_path}'
            
    print(f"--- Loading Data from {base_path} ---")
    
    try:
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
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Ensure you are running the script from the project root or checks your 'data/' folder structure.")
        raise
