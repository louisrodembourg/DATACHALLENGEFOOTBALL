import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import os

# Proper path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from src.data import load_data
    from src.features import build_features
except ImportError:
    print("❌ Impossible d'importer load_data/build_features depuis src.")
    sys.exit()

# =============================================================================
# 2. ARCHITECTURE TRANSFORMER (Le cœur du papier)
# =============================================================================

class FeatureTokenizer(nn.Module):
    """Projette les features (2600) en une séquence de tokens."""
    def __init__(self, input_dim, num_tokens, token_dim):
        super(FeatureTokenizer, self).__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        
        # Cette couche apprend à regrouper tes 2600 features en 'num_tokens' concepts
        self.projection = nn.Linear(input_dim, num_tokens * token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim)) # Token spécial "Classification"

    def forward(self, x):
        batch_size = x.size(0)
        # Projection : (Batch, Input) -> (Batch, Num_Tokens * Token_Dim)
        x = self.projection(x)
        # Reshape : (Batch, Num_Tokens, Token_Dim)
        x = x.view(batch_size, self.num_tokens, self.token_dim)
        return x

class SoccerTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=3, num_tokens=64, token_dim=32, num_heads=4, num_layers=2, dropout=0.3):
        super(SoccerTransformer, self).__init__()
        
        # 1. Tokenizer (Transforme la ligne Excel en séquence compréhensible par le Transformer)
        self.tokenizer = FeatureTokenizer(input_dim, num_tokens, token_dim)
        
        # 2. Positional Encoding (Optionnel sur tabulaire, mais aide parfois)
        # Ici on laisse le Transformer apprendre les relations sans position fixe
        
        # 3. Transformer Encoder (Self-Attention Mechanism)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, 
            nhead=num_heads, 
            dim_feedforward=token_dim * 4, 
            dropout=dropout,
            batch_first=True, # Important: (Batch, Seq, Features)
            norm_first=True   # Pre-LayerNorm (plus stable)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Head de Classification
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_tokens * token_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (Batch, 2600)
        tokens = self.tokenizer(x)          # -> (Batch, 64, 32)
        
        # Le Transformer applique l'Attention: "Quel token doit regarder quel autre token ?"
        attended_tokens = self.transformer_encoder(tokens) # -> (Batch, 64, 32)
        
        # Classification
        logits = self.output_head(attended_tokens)
        return logits

# =============================================================================
# 3. LE RUN
# =============================================================================
def run_transformer_training():
    print("🚀 DÉMARRAGE DU RUN TRANSFORMER (Self-Attention) 🚀")
    
    # --- CHARGEMENT ---
    raw_data = load_data()
    train_team, test_team, train_player, test_player = [], [], [], []
    y_target = None
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
    
    if 'ID' in xt_test_h.columns: test_ids = xt_test_h['ID'].values
    else: test_ids = xt_test_h.index.values

    def clean_df(df):
        df.columns = [c.upper() for c in df.columns]
        return df.loc[:, ~df.columns.duplicated()]
    
    xt_h, xt_a, xp_h, xp_a = clean_df(xt_h), clean_df(xt_a), clean_df(xp_h), clean_df(xp_a)
    xt_test_h, xt_test_a, xp_test_h, xp_test_a = clean_df(xt_test_h), clean_df(xt_test_a), clean_df(xp_test_h), clean_df(xp_test_a)

    print("Construction X_train...")
    X_train_df = build_features(xt_h, xt_a, xp_h, xp_a)
    if 'ID' in X_train_df.columns: X_train_df.drop(columns=['ID'], inplace=True)
    print("Construction X_test...")
    X_test_df = build_features(xt_test_h, xt_test_a, xp_test_h, xp_test_a)
    final_cols = X_train_df.columns.tolist()
    X_test_df = X_test_df.reindex(columns=final_cols, fill_value=0)
    
    # Target & Scaling
    y_classes = y_target[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_classes)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df) 

    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # --- CONFIG TRANSFORMER ---
    input_dim = X_train_df.shape[1]
    # Architecture inspirée du papier : Self-Attention
    model = SoccerTransformer(
        input_dim=input_dim, 
        num_classes=3, 
        num_tokens=16,   # On divise les 2600 features en 32 concepts
        token_dim=32,    # Chaque concept a 64 dimensions
        num_heads=2,     # 4 têtes d'attention (pour voir différents aspects)
        num_layers=1,    # Profondeur
        dropout=0.5     # Régularisation
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-2) # LR plus faible pour Transformer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    num_epochs = 30
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\n🧠 Entraînement Transformer (Dim: {input_dim} -> 32 tokens de dim 64)...")
    
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
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"   💾 RECORD ! (Val Loss: {best_val_loss:.4f})")

    # --- PRÉDICTION HARD (0/1) ---
    print("\n♻️ Chargement du meilleur Transformer...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted_indices = torch.max(probs, 1)
        
        hard_preds = np.zeros(probs.shape, dtype=int)
        hard_preds[np.arange(len(probs)), predicted_indices.cpu().numpy()] = 1
        
        print(f"   -> Exemple sortie : {hard_preds[0]}")

    print(f"Ordre des classes : {le.classes_}")
    submission = pd.DataFrame(hard_preds, columns=le.classes_)
    submission['ID'] = test_ids
    submission = submission[['ID', 'HOME_WINS', 'DRAW', 'AWAY_WINS']]
    
    filename = "submission_transformer_hard.csv"
    submission.to_csv(filename, index=False)
    print(f"✅ Fichier '{filename}' généré !")

if __name__ == "__main__":
    run_transformer_training()