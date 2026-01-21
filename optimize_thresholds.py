"""
Script d'optimisation des seuils de décision (Threshold Optimization).
Basé sur le pipeline de 'football_clean.py'.

Objectif : Trouver les coefficients [w_away, w_draw, w_home] qui maximisent l'accuracy
en ajustant les probabilités brutes sorties du modèle.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import os
import re
import glob
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from typing import cast

# Import des fonctions depuis football_clean
# Assurez-vous que football_clean.py est dans le même dossier
try:
    from football_clean import load_data, build_features, save_experiment
except ImportError:
    try:
        from football import load_data, build_features, save_experiment
    except ImportError:
        raise ImportError("Impossible d'importer load_data/build_features depuis football_clean.py")

# =============================================================================
# 1. PRÉPARATION DES DONNÉES (Copie conforme du pipeline clean)
# =============================================================================
def get_processed_data():
    print("--- Chargement et Préparation des données ---")
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, y_supp, 
     xtest_h, xtest_a, xpt_h, xpt_a) = load_data()

    X_train = build_features(xt_h, xt_a, xp_h, xp_a)
    X_test = build_features(xtest_h, xtest_a, xpt_h, xpt_a)

    # 1. Corrélation
    print("Suppression corrélations (>0.95)...")
    features_for_corr = X_train.drop(columns=['ID'], errors='ignore').select_dtypes(include=[np.number])
    corr_matrix = features_for_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_train.drop(columns=to_drop, errors='ignore', inplace=True)
    X_test.drop(columns=to_drop, errors='ignore', inplace=True)

    # 2. Alignement
    print("Alignement Train/Test...")
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)
    
    # ID management
    if 'ID' in X_test.columns: ID_test = X_test['ID']
    else: ID_test = X_test.index
    
    X_train.drop(columns=['ID'], errors='ignore', inplace=True)
    X_test.drop(columns=['ID'], errors='ignore', inplace=True)

    # 3. Targets
    y_classes = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1)
    le = LabelEncoder()
    y_train_cls = le.fit_transform(y_classes)
    y_train_reg = y_supp['GOAL_DIFF_HOME_AWAY']

    # 4. Top-K Selection (Important pour matcher la performance)
    TOPK_FEATURES = 800
    print(f"Sélection Top-{TOPK_FEATURES} features...")
    selector_model = lgb.LGBMClassifier(n_estimators=500, random_state=42, n_jobs=-1, verbose=-1)
    selector_model.fit(X_train, y_train_cls)
    booster = selector_model.booster_
    importances = booster.feature_importance(importance_type='gain')
    feature_names = booster.feature_name()
    imp_df = pd.DataFrame({'feature': feature_names, 'importance_gain': importances})
    imp_df.sort_values('importance_gain', ascending=False, inplace=True)
    keep = imp_df['feature'].head(min(TOPK_FEATURES, imp_df.shape[0])).tolist()
    X_train = X_train[keep].copy()
    X_test = X_test[keep].copy()

    # 5. Aux Feature (Goal Diff)
    print("Ajout Feature Goal Diff (Stacking)...")
    regressor = CatBoostRegressor(iterations=300, depth=6, verbose=0, random_state=42)
    X_train['PRED_GOAL_DIFF'] = cross_val_predict(cast(BaseEstimator, regressor), X_train, y_train_reg, cv=5, n_jobs=-1)
    regressor.fit(X_train.drop(columns=['PRED_GOAL_DIFF']), y_train_reg)
    X_test['PRED_GOAL_DIFF'] = regressor.predict(X_test)
    
    return X_train, y_train_cls, X_test, ID_test, le

# =============================================================================
# 2. OPTIMISATION DES SEUILS
# =============================================================================

def objective_thresholds(trial, probas, y_true):
    # On cherche 3 multiplicateurs [w0, w1, w2] autour de 1.0
    # w_draw (indice 1) est souvent le plus critique à booster
    w_away = trial.suggest_float('w_away', 0.8, 1.3)
    w_draw = trial.suggest_float('w_draw', 0.9, 1.6) # On permet d'augmenter plus le nul
    w_home = trial.suggest_float('w_home', 0.8, 1.3)
    
    weights = np.array([w_away, w_draw, w_home])
    
    # Application des poids : P_weighted = P_raw * W
    weighted_probas = probas * weights
    
    # Argmax sur les probas pondérées
    preds = weighted_probas.argmax(axis=1)
    
    return accuracy_score(y_true, preds)

if __name__ == "__main__":
    X_train, y_train, X_test, ID_test, le_encoder = get_processed_data()
    
    print("\n--- Entraînement Modèle pour Probabilités (LGBM) ---")
    # Utilisation des hyperparamètres optimisés de football_clean.py
    params_lgb = {
        'n_estimators': 1500, # Un peu boosté pour la précision
        'learning_rate': 0.007, 
        'num_leaves': 96, 
        'colsample_bytree': 0.73, 
        'subsample': 0.95, 
        'random_state': 42,
        'max_depth': 18, 
        'n_jobs': -1, 
        'verbose': -1,
         # Ajout weight balance si besoin, mais on le fait post-hoc
    }
    
    model = lgb.LGBMClassifier(**params_lgb)
    
    # 1. Générer les probabilités Out-Of-Fold (OOF) sur le Train
    # C'est CRITIQUE : on ne peut pas optimiser les seuils sur des probas "train" (trop confiantes)
    # Il faut des probas qui ressemblent à celles du test.
    print("Génération des probabilités OOF (Cross-Validation 5 folds)...")
    oof_probas = cross_val_predict(
        model, 
        X_train, 
        y_train, 
        cv=5, 
        method='predict_proba', 
        n_jobs=-1
    )
    
    base_acc = accuracy_score(y_train, oof_probas.argmax(axis=1))
    print(f"Accuracy de base (Argmax standard) : {base_acc:.5f}")

    # 2. Optimiser les seuils avec Optuna
    print("\n--- Optimisation des Seuils (Optuna) ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_thresholds(trial, oof_probas, y_train), n_trials=100)
    
    best_weights = [
        study.best_params['w_away'], 
        study.best_params['w_draw'], 
        study.best_params['w_home']
    ]
    best_acc = study.best_value
    
    print(f"\n✅ Terminé !")
    print(f"Accuracy optimisée : {best_acc:.5f} (Gain: +{best_acc - base_acc:.5f})")
    print(f"Meilleurs Poids [Away, Draw, Home] : {best_weights}")

    # 3. Prédire sur le Test et appliquer les poids
    print("\n--- Génération Soumission Finale ---")
    model.fit(X_train, y_train)
    test_probas = model.predict_proba(X_test)
    
    # Application des poids optimaux
    weighted_test_probas = test_probas * np.array(best_weights)
    final_preds_indices = weighted_test_probas.argmax(axis=1)
    
    # Création CSV
    # Reconstitution One-Hot manuelle basée sur les indices prédits
    submission = pd.DataFrame({'ID': ID_test})
    
    # Initialiser à 0
    submission['HOME_WINS'] = 0
    submission['DRAW'] = 0
    submission['AWAY_WINS'] = 0
    
    # Encoder mapping: le.classes_ est généralement ['AWAY_WINS', 'DRAW', 'HOME_WINS']
    # Vérifions l'ordre de l'encoder
    class_names = le_encoder.classes_ 
    print(f"Ordre des classes encodeur : {class_names}")
    
    # Remplir les 1 selon la prédiction
    # Attention: y_train_cls a été fait avec fit_transform sur ['AWAY_WINS', 'DRAW', 'HOME_WINS']
    # Donc 0=Away, 1=Draw, 2=Home (ordre alphabétique standard)
    
    # On assigne 1 à la colonne correspondante
    # Si pred=0 (Away), on met 1 dans AWAY_WINS
    for idx, cls_name in enumerate(class_names):
        submission.loc[final_preds_indices == idx, cls_name] = 1
        
    description = f"LGBM Optimized Thresholds - Acc {best_acc:.4f} - Weights {np.round(best_weights, 2)}"
    
    save_experiment(best_acc, study.best_params, description, submission)
