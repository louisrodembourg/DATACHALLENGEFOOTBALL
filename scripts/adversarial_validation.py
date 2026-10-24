"""
Adversarial validation: train and test come from different distributions (AUC ~0.98), which explains
why CV gains do not transfer online. This script:

1. Scores how "test-like" each train match is (OOF classifier train-vs-test).
2. Builds a validation set from the most test-like train matches (a proxy of the online score).
3. Compares feature sets on that proxy: all features, team-only, and all minus the most shifted features.

Usage:
    python scripts/adversarial_validation.py
    python scripts/adversarial_validation.py --submit team_only --top 250   # fit on full train + submission
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
CACHE_DIR = os.path.join(project_root, 'cache')
sys.path.append(project_root)

from src.utils import save_experiment

warnings.filterwarnings('ignore')

PARAMS_CAT = {
    'iterations': 800, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 10.0,
    'border_count': 101, 'subsample': 0.8, 'random_strength': 5.0, 'rsm': 0.6,
    'verbose': 0, 'random_seed': 42, 'bootstrap_type': 'Bernoulli', 'allow_writing_files': False
}
VAL_SHARE = 0.25  # share of the most test-like train matches used as validation


def adversarial_scores(X_train, X_test):
    """OOF probability that each train row belongs to the test set."""
    path = os.path.join(CACHE_DIR, 'adv_scores.npy')
    if os.path.exists(path):
        return np.load(path)
    X = np.vstack([X_train, X_test])
    z = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3,
                             verbose=-1, n_jobs=-1)
    p = cross_val_predict(clf, X, z, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    print(f"Adversarial AUC (train vs test): {roc_auc_score(z, p):.4f}")
    np.save(path, p[:len(X_train)])
    return p[:len(X_train)]


def evaluate(X_tr, y_tr, X_val, y_val, gd_tr, n_top):
    """CatBoost on the top-n LGBM features + leak-free PRED_GOAL_DIFF, scored on the proxy validation."""
    reg = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_seed=42,
                            allow_writing_files=False)
    f_tr = cross_val_predict(reg, X_tr, gd_tr, cv=5, n_jobs=-1)
    f_val = reg.fit(X_tr, gd_tr).predict(X_val)
    ranker = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, importance_type='gain', random_state=42,
                                verbose=-1, n_jobs=-1).fit(X_tr, y_tr)
    top = np.argsort(ranker.feature_importances_)[::-1][:n_top]
    A_tr = np.column_stack([X_tr[:, top], f_tr])
    A_val = np.column_stack([X_val[:, top], f_val])
    p = CatBoostClassifier(**PARAMS_CAT).fit(A_tr, y_tr).predict_proba(A_val)
    return accuracy_score(y_val, p.argmax(1)), log_loss(y_val, p)


def submit(Xtr, y, gd, Xte, ID_test, n_top, seeds, name, proxy_score):
    """Full-train fit: OOF PRED_GOAL_DIFF for train, full-fit one for test, top-n features, seed-averaged CatBoost."""
    reg = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_seed=42,
                            allow_writing_files=False)
    f_tr = cross_val_predict(reg, Xtr, gd, cv=5, n_jobs=-1)
    f_te = reg.fit(Xtr, gd).predict(Xte)
    ranker = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, importance_type='gain', random_state=42,
                                verbose=-1, n_jobs=-1).fit(Xtr, y)
    top = np.argsort(ranker.feature_importances_)[::-1][:n_top]
    A_tr, A_te = np.column_stack([Xtr[:, top], f_tr]), np.column_stack([Xte[:, top], f_te])
    probs = np.mean([CatBoostClassifier(**{**PARAMS_CAT, 'random_seed': s}).fit(A_tr, y).predict_proba(A_te)
                     for s in seeds], axis=0)
    np.save(os.path.join(CACHE_DIR, f'test_adv_{name}_top{n_top}.npy'), probs)
    preds = probs.argmax(1)
    submission = pd.DataFrame({'ID': ID_test})
    submission['HOME_WINS'] = (preds == 2).astype(int)
    submission['DRAW'] = (preds == 1).astype(int)
    submission['AWAY_WINS'] = (preds == 0).astype(int)
    save_experiment(proxy_score, {"feature_set": name, "top": n_top, "seeds": seeds},
                    f"CatBoost {name} top{n_top} (score = adversarial proxy validation)", submission)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial validation")
    parser.add_argument("--submit", default=None, help="Feature set to fit on full train and submit")
    parser.add_argument("--top", type=int, default=250)
    parser.add_argument("--proxy-score", type=float, default=float('nan'), help="Proxy accuracy (for logging)")
    args = parser.parse_args()

    cache = pd.read_pickle(os.path.join(CACHE_DIR, 'features.pkl'))
    Xtr_df = cache['X_train'].drop(columns=['PRED_GOAL_DIFF'])
    Xte_df = cache['X_test'].drop(columns=['PRED_GOAL_DIFF'])
    y = np.asarray(cache['y'])
    gd = pd.read_csv(os.path.join(project_root, 'data/benchmark_and_extras/Y_train_supp.csv'))['GOAL_DIFF_HOME_AWAY'].values
    cols = Xtr_df.columns

    adv = adversarial_scores(Xtr_df.values, Xte_df.values)
    val = adv >= np.quantile(adv, 1 - VAL_SHARE)
    print(f"Proxy validation: {val.sum()} most test-like train matches (mean p(test)={adv[val].mean():.3f} "
          f"vs {adv[~val].mean():.3f} for the others)")

    # Per-feature shift: KS statistic train vs test
    ks = pd.Series({c: ks_2samp(Xtr_df[c].values, Xte_df[c].values).statistic for c in cols})
    player = [c for c in cols if 'P_HOME' in c or 'P_AWAY' in c or 'PLAYER' in c]
    team = [c for c in cols if c not in player]
    print(f"KS shift: median={ks.median():.3f}, >0.10: {(ks > 0.10).sum()} cols, >0.05: {(ks > 0.05).sum()} cols")
    print("Most shifted:", ks.sort_values(ascending=False).head(6).round(3).to_dict())

    feature_sets = {
        'all': list(cols),
        'team_only': team,
        'ks<0.10': list(ks[ks < 0.10].index),
        'ks<0.05': list(ks[ks < 0.05].index),
    }
    if args.submit:
        cs = feature_sets[args.submit]
        submit(Xtr_df[cs].values, y, gd, Xte_df[cs].values, cache['ID_test'].values, args.top, [42, 7, 123],
               args.submit, args.proxy_score)
        sys.exit(0)

    print("\n=== Proxy validation (test-like matches) ===")
    for name, cs in feature_sets.items():
        X = Xtr_df[cs].values
        for n_top in (250, 1350):
            n = min(n_top, len(cs))
            acc, ll = evaluate(X[~val], y[~val], X[val], y[val], gd[~val], n)
            print(f"{name:10s} ({len(cs):4d} cols) top{n:<5d} acc={acc:.4f}  logloss={ll:.4f}", flush=True)
