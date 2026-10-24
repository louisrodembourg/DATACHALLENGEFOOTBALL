"""
Ensemble search over a library of fast, diverse models (+ cached TabICL probabilities).

1. Build: for each candidate model, out-of-fold probabilities (same 5 folds as run_pipeline.py)
   and test probabilities (fit on full train). Saved in cache/lib/, skipped if already there.
2. Select: greedy forward selection with replacement (Caruana et al., 2004) minimising log-loss,
   evaluated with NESTED CV (selection on 4 folds, scoring on the 5th) for an honest estimate.

Usage:
    python scripts/ensemble_search.py               # build + select
    python scripts/ensemble_search.py --submit      # also write the submission of the selected ensemble
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, log_loss

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.utils import save_experiment

warnings.filterwarnings('ignore')

CACHE_DIR = os.path.join(project_root, 'cache')
LIB_DIR = os.path.join(CACHE_DIR, 'lib')

# Same parameters as run_pipeline.py
PARAMS_CAT = {
    'iterations': 800, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 10.0,
    'border_count': 101, 'subsample': 0.8, 'random_strength': 5.0, 'rsm': 0.6,
    'verbose': 0, 'bootstrap_type': 'Bernoulli', 'allow_writing_files': False
}
PARAMS_LGB = {
    'n_estimators': 300, 'learning_rate': 0.015, 'num_leaves': 31, 'subsample': 0.8, 'subsample_freq': 1,
    'colsample_bytree': 0.6, 'max_depth': 6, 'min_child_samples': 100, 'reg_alpha': 10.0, 'reg_lambda': 20.0,
    'verbose': -1, 'n_jobs': -1
}
PARAMS_XGB = {
    'n_estimators': 300, 'learning_rate': 0.02, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.6,
    'eval_metric': 'mlogloss', 'tree_method': 'hist', 'reg_alpha': 5.0, 'reg_lambda': 10.0,
    'gamma': 1.0, 'min_child_weight': 5, 'n_jobs': -1
}


class GoalDiffOrdinal:
    """Ordinal view of the problem: regress the goal difference, then map it to H/D/A probabilities
    with a multinomial logistic regression fitted on inner out-of-fold predictions."""

    def __init__(self, goal_diff, seed=42):
        self.goal_diff = goal_diff
        self.seed = seed

    def fit(self, X, y, idx):
        self.reg = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, loss_function='RMSE',
                                     verbose=0, random_seed=self.seed, allow_writing_files=False)
        gd = self.goal_diff[idx]
        inner = cross_val_predict(self.reg, X, gd, cv=3)
        self.reg.fit(X, gd)
        self.link = LogisticRegression(C=1.0).fit(inner.reshape(-1, 1), y)
        return self

    def predict_proba(self, X):
        return self.link.predict_proba(self.reg.predict(X).reshape(-1, 1))


# name -> (number of top features, factory)
CANDIDATES = {
    'cat250_s42': (250, lambda: CatBoostClassifier(**PARAMS_CAT, random_seed=42)),
    'cat250_s7': (250, lambda: CatBoostClassifier(**PARAMS_CAT, random_seed=7)),
    'cat250_s123': (250, lambda: CatBoostClassifier(**PARAMS_CAT, random_seed=123)),
    'cat100': (100, lambda: CatBoostClassifier(**PARAMS_CAT, random_seed=42)),
    'cat600': (600, lambda: CatBoostClassifier(**PARAMS_CAT, random_seed=42)),
    'cat1000': (1000, lambda: CatBoostClassifier(**PARAMS_CAT, random_seed=42)),
    'cat1350': (1350, lambda: CatBoostClassifier(**PARAMS_CAT, random_seed=42)),
    'cat600_s7': (600, lambda: CatBoostClassifier(**PARAMS_CAT, random_seed=7)),
    'xgb600': (600, lambda: xgb.XGBClassifier(**PARAMS_XGB, random_state=42)),
    'lgb1350': (1350, lambda: lgb.LGBMClassifier(**PARAMS_LGB, random_state=42)),
    'lgb250': (250, lambda: lgb.LGBMClassifier(**PARAMS_LGB, random_state=42)),
    'lgb600': (600, lambda: lgb.LGBMClassifier(**PARAMS_LGB, random_state=42)),
    'xgb250': (250, lambda: xgb.XGBClassifier(**PARAMS_XGB, random_state=42)),
    'logreg100': (100, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))),
    'logreg250': (250, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=2000))),
    'gd_ordinal': (250, 'ordinal'),
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Ensemble search")
    parser.add_argument("--submit", action="store_true", help="Write the submission of the selected ensemble")
    parser.add_argument("--leakfree", action="store_true",
                        help="Recompute PRED_GOAL_DIFF inside each fold (honest CV); TabICL excluded (its OOF is leaky)")
    parser.add_argument("--rounds", type=int, default=25, help="Greedy selection rounds")
    return parser.parse_args()


def rank_features(X, y):
    """Feature ranking by LGBM gain (same ranker as the SelectFromModel of run_pipeline.py)."""
    ranker = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, importance_type='gain',
                                random_state=42, verbose=-1, n_jobs=-1).fit(X, y)
    return np.argsort(ranker.feature_importances_)[::-1]


def fit_predict(name, X_tr, y_tr, X_te, tr_idx, goal_diff):
    k, factory = CANDIDATES[name]
    if factory == 'ordinal':
        model = GoalDiffOrdinal(goal_diff).fit(X_tr[:, :k], y_tr, tr_idx)
    else:
        model = factory().fit(X_tr[:, :k], y_tr)
    return model.predict_proba(X_te[:, :k])


def fold_goal_diff(X, goal_diff, tr, te, gd_col, k):
    """Leak-free PRED_GOAL_DIFF for one CV fold, built like the test-time feature:
    out-of-fold predictions inside the fold's train rows, and a regressor fitted on those rows for the val rows.
    (The cached feature was OOF over the whole train set, so train rows had seen the val rows' goal diffs.)"""
    path = os.path.join(CACHE_DIR, f'gd_fold{k}.npz')
    if os.path.exists(path):
        d = np.load(path)
        return d['tr'], d['te']
    other = np.delete(np.arange(X.shape[1]), gd_col)
    reg = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, loss_function='RMSE', verbose=0,
                            random_seed=42, allow_writing_files=False)
    gd_tr = cross_val_predict(reg, X[tr][:, other], goal_diff[tr], cv=5, n_jobs=-1)
    gd_te = reg.fit(X[tr][:, other], goal_diff[tr]).predict(X[te][:, other])
    np.savez(path, tr=gd_tr, te=gd_te)
    return gd_tr, gd_te


def build_library(X, y, X_test, goal_diff, folds, gd_col=None):
    os.makedirs(LIB_DIR, exist_ok=True)
    todo = [n for n in CANDIDATES if not os.path.exists(os.path.join(LIB_DIR, f'test_{n}.npy'))]
    if not todo:
        return
    print(f"--- Building {len(todo)} candidates: {todo} ---")

    # Rankings computed once per fold (and once on full train), reused by every candidate
    oof = {n: np.zeros((len(y), 3)) for n in todo}
    for k, (tr, te) in enumerate(folds):
        X_tr, X_te = X[tr], X[te]
        if gd_col is not None:
            t0 = time.time()
            gd_tr, gd_te = fold_goal_diff(X, goal_diff, tr, te, gd_col, k)
            X_tr[:, gd_col], X_te[:, gd_col] = gd_tr, gd_te
            print(f"  fold {k + 1} leak-free PRED_GOAL_DIFF ({time.time() - t0:.0f}s)", flush=True)
        order = rank_features(X_tr, y[tr])
        X_tr, X_te = X_tr[:, order], X_te[:, order]
        for n in todo:
            t0 = time.time()
            oof[n][te] = fit_predict(n, X_tr, y[tr], X_te, tr, goal_diff)
            print(f"  fold {k + 1} {n:12s} acc={accuracy_score(y[te], oof[n][te].argmax(1)):.4f} ({time.time() - t0:.0f}s)",
                  flush=True)

    order = rank_features(X, y)
    X_full, X_test_o = X[:, order], X_test[:, order]
    for n in todo:
        test = fit_predict(n, X_full, y, X_test_o, np.arange(len(y)), goal_diff)
        np.save(os.path.join(LIB_DIR, f'oof_{n}.npy'), oof[n])
        np.save(os.path.join(LIB_DIR, f'test_{n}.npy'), test)


def load_library(with_tabicl=True):
    lib = {}
    for f in sorted(os.listdir(LIB_DIR)):
        if f.startswith('oof_'):
            n = f[4:-4]
            lib[n] = (np.load(os.path.join(LIB_DIR, f)), np.load(os.path.join(LIB_DIR, f'test_{n}.npy')))
    # TabICL probabilities from test_tabular_fm.py (8h on CPU: never recomputed here)
    tabicl_oof = os.path.join(CACHE_DIR, 'oof_tabicl_n12303_f5.npy')
    if with_tabicl and os.path.exists(tabicl_oof):
        lib['tabicl'] = (np.load(tabicl_oof), np.load(os.path.join(CACHE_DIR, 'test_tabicl.npy')))
    return lib


def greedy_select(oofs, y, idx, rounds):
    """Caruana forward selection with replacement, minimising log-loss on rows `idx`. Returns weights."""
    names = list(oofs)
    counts = dict.fromkeys(names, 0)
    blend = np.zeros((len(idx), 3))
    for r in range(rounds):
        best = min(names, key=lambda n: log_loss(y[idx], (blend * r + oofs[n][idx]) / (r + 1), labels=[0, 1, 2]))
        blend = (blend * r + oofs[best][idx]) / (r + 1)
        counts[best] += 1
    return {n: c / rounds for n, c in counts.items() if c}


def blend_of(weights, probs):
    return sum(w * probs[n] for n, w in weights.items())


if __name__ == "__main__":
    args = _parse_args()
    cache = pd.read_pickle(os.path.join(CACHE_DIR, 'features.pkl'))
    X = cache['X_train'].values
    y = np.asarray(cache['y'])
    X_test = cache['X_test'].values
    goal_diff = pd.read_csv(os.path.join(project_root, 'data/benchmark_and_extras/Y_train_supp.csv'))['GOAL_DIFF_HOME_AWAY'].values
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))

    gd_col = None
    if args.leakfree:
        LIB_DIR = os.path.join(CACHE_DIR, 'lib_leakfree')
        gd_col = list(cache['X_train'].columns).index('PRED_GOAL_DIFF')
    build_library(X, y, X_test, goal_diff, folds, gd_col)
    lib = load_library(with_tabicl=not args.leakfree)
    oofs = {n: v[0] for n, v in lib.items()}

    print("\n=== Candidates (OOF) ===")
    fold_acc = lambda p: np.array([accuracy_score(y[te], p[te].argmax(1)) for _, te in folds])
    for n, p in sorted(oofs.items(), key=lambda kv: log_loss(y, kv[1])):
        print(f"{n:12s} acc={accuracy_score(y, p.argmax(1)):.4f}  logloss={log_loss(y, p):.4f}")

    # Nested CV: select on 4 folds, score on the held-out one
    print("\n=== Nested CV of the greedy ensemble ===")
    base = fold_acc(oofs['cat250_s42'])
    nested = []
    for k, (_, te) in enumerate(folds):
        sel_idx = np.concatenate([folds[j][1] for j in range(5) if j != k])
        w = greedy_select(oofs, y, sel_idx, args.rounds)
        nested.append(accuracy_score(y[te], blend_of(w, oofs)[te].argmax(1)))
        print(f"  fold {k + 1}: ens={nested[-1]:.4f}  cat={base[k]:.4f}  weights={ {n: round(v, 2) for n, v in w.items()} }")
    nested = np.array(nested)
    print(f"Nested ensemble acc = {nested.mean():.4f} ± {nested.std():.4f}  vs cat250 {base.mean():.4f} "
          f"(gain {nested.mean() - base.mean():+.4f}, wins {(nested > base).sum()}/5)")

    weights = greedy_select(oofs, y, np.arange(len(y)), args.rounds)
    print(f"Final weights (all folds): { {n: round(v, 2) for n, v in weights.items()} }")

    if args.submit:
        test = blend_of(weights, {n: v[1] for n, v in lib.items()})
        preds = test.argmax(1)
        submission = pd.DataFrame({'ID': cache['ID_test'].values})
        submission['HOME_WINS'] = (preds == 2).astype(int)
        submission['DRAW'] = (preds == 1).astype(int)
        submission['AWAY_WINS'] = (preds == 0).astype(int)
        save_experiment(float(nested.mean()), {"weights": weights, "nested_folds": nested.tolist()},
                        f"Greedy ensemble{' leak-free' if args.leakfree else ''} (nested CV score)", submission)
