import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sys

# Gestion des imports (Load Data)
try:
    from football import load_data
except ImportError:
    try:
        from football import load_data
    except ImportError:
        print("❌ Impossible de trouver load_data.")
        sys.exit()

# =============================================================================
# 1. PRÉPARATION DES DONNÉES (Feature Engineering)
# =============================================================================
def aggregate_players_by_position(player_df, prefix):
    # Sécurité doublons colonnes
    player_df = player_df.loc[:, ~player_df.columns.duplicated()]
    
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

def build_features(xt_h, xt_a, xp_h, xp_a):
    print("   -> Construction des features...")
    p_home_agg = aggregate_players_by_position(xp_h, 'P_HOME')
    p_away_agg = aggregate_players_by_position(xp_a, 'P_AWAY')
    
    df = xt_h.merge(xt_a, on='ID', suffixes=('_HOME', '_AWAY'))
    df = df.merge(p_home_agg, on='ID', how='left')
    df = df.merge(p_away_agg, on='ID', how='left')
    
    # Remplissage basique
    df.fillna(0, inplace=True)
    
    # Nettoyage colonnes inutiles
    drop_cols = ['LEAGUE_HOME', 'TEAM_NAME_HOME', 'LEAGUE_AWAY', 'TEAM_NAME_AWAY', 'LEAGUE', 'TEAM_NAME']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    # Suppression colonnes object restantes
    cols_to_drop = df.select_dtypes(include=['object']).columns
    if len(cols_to_drop) > 0: df.drop(columns=cols_to_drop, inplace=True)
    
    return df

# =============================================================================
# 2. ARCHITECTURE DU MODÈLE (OPTIMISÉE V2 - FUNNEL)
# =============================================================================
class SoccerMLP(nn.Module):
    def __init__(self, input_size, num_classes=3):
        super(SoccerMLP, self).__init__()
        
        # ÉTAGE 1 : Compression massive (Input -> 512)
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512), # Normalise pour stabiliser l'apprentissage
            nn.ReLU(),
            nn.Dropout(0.4)      # Oublie 40% des infos pour éviter le par cœur
        )
        
        # ÉTAGE 2 : Affinage (512 -> 256)
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # ÉTAGE 3 : Précision (256 -> 64)
        self.layer3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # SORTIE (3 Classes)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

# =============================================================================
# 3. LE RUN PYTORCH
# =============================================================================
def run_pytorch_training():
    print("🚀 DÉMARRAGE DU RUN PYTORCH (OPTIMISÉ + HARD PREDS) 🚀")
    
    # --- A. CHARGEMENT & TRI DES DONNÉES ---
    raw_data = load_data()
    train_team, test_team, train_player, test_player = [], [], [], []
    y_target = None

    # Smart Sorting
    for item in raw_data:
        if not hasattr(item, 'shape'): continue
        rows = item.shape[0]
        if rows < 15000:
            if hasattr(item, 'columns') and 'HOME_WINS' in item.columns: y_target = item
            elif len(item.shape) == 2: train_team.append(item)
        elif 20000 <= rows < 50000: test_team.append(item)
        elif 200000 <= rows < 400000: train_player.append(item)
        elif rows >= 400000: test_player.append(item)

    xt_h, xt_a = train_team[0], train_team[1]
    xp_h, xp_a = train_player[0], train_player[1]
    xt_test_h, xt_test_a = test_team[0], test_team[1]
    xp_test_h, xp_test_a = test_player[0], test_player[1]
    
    # Récupération IDs Test
    if 'ID' in xt_test_h.columns: test_ids = xt_test_h['ID'].values
    else: test_ids = xt_test_h.index.values

    # Nettoyage Majuscules
    def clean_df(df):
        df.columns = [c.upper() for c in df.columns]
        return df.loc[:, ~df.columns.duplicated()]
    
    xt_h, xt_a = clean_df(xt_h), clean_df(xt_a)
    xp_h, xp_a = clean_df(xp_h), clean_df(xp_a)
    xt_test_h, xt_test_a = clean_df(xt_test_h), clean_df(xt_test_a)
    xp_test_h, xp_test_a = clean_df(xp_test_h), clean_df(xp_test_a)

    # --- B. CONSTRUCTION FEATURES ---
    print("Construction X_train...")
    X_train_df = build_features(xt_h, xt_a, xp_h, xp_a)
    if 'ID' in X_train_df.columns: X_train_df.drop(columns=['ID'], inplace=True)

    print("Construction X_test...")
    X_test_df = build_features(xt_test_h, xt_test_a, xp_test_h, xp_test_a)
    
    # Alignement Colonnes (Le Test doit copier le Train)
    final_cols = X_train_df.columns.tolist()
    X_test_df = X_test_df.reindex(columns=final_cols, fill_value=0)
    
    # --- C. PRÉPARATION TENSORS ---
    # 1. Target (Y)
    y_classes = y_target[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_classes)
    
    # 2. Scaling (Important !)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df) 

    # 3. Conversion Torch
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # 4. Split Train/Val
    X_tr, X_val, y_tr, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # 5. DataLoaders
    batch_size = 64
    train_data = TensorDataset(X_tr, y_tr)
    val_data = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # --- D. ENTRAÎNEMENT AVEC SCHEDULER & CHECKPOINT ---
    input_size = X_train_df.shape[1]
    model = SoccerMLP(input_size, num_classes=3)
    
    criterion = nn.CrossEntropyLoss()
    # AdamW + Weight Decay : Pour éviter l'overfitting
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # FIX: Suppression de 'verbose=True' pour compatibilité PyTorch récent
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    num_epochs = 30
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\n🧠 Démarrage de l'entraînement sur {input_size} features...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        acc = 100 * correct / total
        
        # Mise à jour vitesse d'apprentissage
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%")
        
        # SAUVEGARDE DU MEILLEUR
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"   💾 RECORD ! (Val Loss: {best_val_loss:.4f})")
            
    print("\n🏁 Entraînement terminé.")

    # --- E. PRÉDICTION "HARD" (0/1) ---
    print("♻️ Chargement de la meilleure version du modèle...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("🔮 Prédiction FERME (0 ou 1) sur le Test Set...")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        
        # Calcul des probas (juste pour trouver le max)
        probs = torch.softmax(outputs, dim=1)
        
        # On trouve l'index du gagnant
        _, predicted_indices = torch.max(probs, 1)
        
        # On crée une matrice de Zéros
        hard_preds = np.zeros(probs.shape, dtype=int)
        
        # On met des '1' là où le modèle a prédit le gagnant
        hard_preds[np.arange(len(probs)), predicted_indices.cpu().numpy()] = 1
        
        print(f"   -> Exemple de sortie : {hard_preds[0]}")

    # --- F. SOUMISSION ---
    print(f"Ordre des classes détecté : {le.classes_}")
    
    # On utilise hard_preds !
    submission = pd.DataFrame(hard_preds, columns=le.classes_)
    submission['ID'] = test_ids
    
    # Réorganisation
    submission = submission[['ID', 'HOME_WINS', 'DRAW', 'AWAY_WINS']]
    
    filename = "submission_pytorch_mlp_HARD.csv"
    submission.to_csv(filename, index=False)
    print(f"✅ Fichier '{filename}' généré (Format 0/1) !")

if __name__ == "__main__":
    run_pytorch_training()