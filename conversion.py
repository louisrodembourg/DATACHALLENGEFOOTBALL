import pandas as pd
import numpy as np

# 1. Charge ton fichier de probabilités existant
df = pd.read_csv('submission_qrt_optimized.csv')

# 2. Identifie la colonne gagnante pour chaque ligne
cols = ['AWAY_WINS', 'DRAW', 'HOME_WINS'] # Ordre alphabétique souvent par défaut, vérifie tes colonnes
# Note: Dans ton fichier généré, l'ordre était HOME, DRAW, AWAY. Utilisons celui-là.
cols_submit = ['HOME_WINS', 'DRAW', 'AWAY_WINS']

# On trouve l'index de la valeur max (0, 1 ou 2)
# argmax sur l'axe 1 (les colonnes)
max_indices = df[cols_submit].values.argmax(axis=1)

# 3. Crée une matrice de zéros et mets des 1 au bon endroit
hard_preds = np.zeros(df[cols_submit].shape, dtype=int)
hard_preds[np.arange(len(df)), max_indices] = 1

# 4. Remplace dans le DataFrame
df[cols_submit] = hard_preds

# 5. Sauvegarde le nouveau fichier "Strict"
import os
os.makedirs('submission', exist_ok=True)
submission_path = os.path.join('submission', 'submission_qrt_binaireV2.csv')

df.to_csv(submission_path, index=False)
print(f"Fichier '{submission_path}' généré avec succès.")