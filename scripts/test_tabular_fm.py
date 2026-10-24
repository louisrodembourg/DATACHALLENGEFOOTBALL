"""
Benchmark of Tabular Foundation Models (TabICL, optional TabPFN) vs the CatBoost baseline.

Data preparation is identical to `run_pipeline.py` (features, correlation removal,
auxiliary goal-diff feature, in-fold LGBM selection of 250 features) so CV scores are comparable.

Usage:
    python scripts/test_tabular_fm.py --n-rows 2000 --folds 1        # smoke test
    python scripts/test_tabular_fm.py                                 # full 5-fold CV
    python scripts/test_tabular_fm.py --submit --weight 0.5           # full fit + submission
"""

import os
import sys
import time
import argparse
import resource
import warnings

import numpy as np
import pandas as pd
# torch MUST be imported before lightgbm/catboost: importing it after their OpenMP pool
# has started deadlocks on macOS (two libomp copies)
import torch  # noqa: F401
import psutil
from tabicl import TabICLClassifier
from tabicl._model.inference import InferenceManager
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, log_loss

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data import load_data
from src.features import build_features
from src.utils import save_experiment

warnings.filterwarnings('ignore')

# On MPS (unified memory) TabICL estimates free GPU memory as `recommended_max - allocated`,
# ignoring RAM used by other processes -> oversized batches, swap thrashing or OOM.
# Cap the estimate by the RAM the OS really has available.
_orig_gpu_mem = InferenceManager.get_available_gpu_memory


def _gpu_mem_capped_by_ram(self):
    free_mb = _orig_gpu_mem(self)
    if torch.device(self.exe_device).type == 'mps':
        free_mb = min(free_mb, psutil.virtual_memory().available / (1024 * 1024))
    return free_mb


InferenceManager.get_available_gpu_memory = _gpu_mem_capped_by_ram

CACHE_DIR = os.path.join(project_root, 'cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'features.pkl')

# Same parameters as run_pipeline.py
PARAMS_CAT = {
    'iterations': 800, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 10.0,
    'border_count': 101, 'subsample': 0.8, 'random_strength': 5.0, 'rsm': 0.6,
    'verbose': 0, 'random_seed': 42, 'bootstrap_type': 'Bernoulli',
    'allow_writing_files': False
}
N_SELECTED = 250
PREDICT_CHUNK = 2500


def _parse_args():
    parser = argparse.ArgumentParser(description="Tabular Foundation Models benchmark")
    parser.add_argument("--models", default="cat,tabicl", help="Comma list among: cat, tabicl, tabpfn")
    parser.add_argument("--n-rows", type=int, default=None, help="Stratified subsample (smoke test)")
    parser.add_argument("--folds", type=int, default=5, help="Number of the 5 folds to actually run")
    # CPU inference hangs after LightGBM/CatBoost on macOS (OpenMP clash) and is 4x slower: use the Apple GPU
    parser.add_argument("--device", default="mps", help="mps or cpu (TabICL / TabPFN)")
    parser.add_argument("--tabicl-estimators", type=int, default=8)
    parser.add_argument("--fm-features", type=int, default=100, help="Top-k selected features given to TabICL/TabPFN")
    parser.add_argument("--mem-safety", type=float, default=0.5, help="TabICL auto-batching safety factor (0-1)")
    parser.add_argument("--submit", action="store_true", help="Full fit + test prediction")
    parser.add_argument("--weight", type=float, default=0.5, help="Weight of the FM in the blend (--submit)")
    parser.add_argument("--cv-score", type=float, default=float('nan'), help="OOF CV of the submitted blend (--submit, for logging)")
    return parser.parse_args()


def _peak_ram_gb():
    # macOS returns bytes, Linux kilobytes
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == 'darwin' else rss / 1e6


# =============================================================================
# 1. Feature cache (same steps as run_pipeline.py 5.A -> 5.C)
# =============================================================================

def build_cache():
    print("--- Building feature cache (one-time) ---")
    (xt_h, xt_a, xp_h, xp_a, y_train_raw, y_supp,
     xtest_h, xtest_a, xpt_h, xpt_a) = load_data()

    X_train = build_features(xt_h, xt_a, xp_h, xp_a)
    del xt_h, xt_a, xp_h, xp_a
    X_test = build_features(xtest_h, xtest_a, xpt_h, xpt_a)
    del xtest_h, xtest_a, xpt_h, xpt_a

    # Correlation Removal
    features_for_corr = X_train.drop(columns=['ID'], errors='ignore').select_dtypes(include=[np.number])
    corr_matrix = features_for_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(f"📉 Removing {len(to_drop)} correlated columns (>0.95)...")
    X_train.drop(columns=to_drop, errors='ignore', inplace=True)
    X_test.drop(columns=to_drop, errors='ignore', inplace=True)

    X_train, X_test = X_train.align(X_test, join='inner', axis=1)
    ID_test = X_test['ID'] if 'ID' in X_test.columns else pd.Series(X_test.index)
    X_train_final = X_train.drop(columns=['ID'], errors='ignore')
    X_test_final = X_test.drop(columns=['ID'], errors='ignore')

    # 0: AWAY_WINS, 1: DRAW, 2: HOME_WINS
    y_target = y_train_raw[['AWAY_WINS', 'DRAW', 'HOME_WINS']].idxmax(axis=1).replace({
        'AWAY_WINS': 0, 'DRAW': 1, 'HOME_WINS': 2
    }).astype(int)

    # Auxiliary Feature (Goal Diff)
    print("Training Auxiliary Model (Goal Diff)...")
    y_goal_diff = y_supp['GOAL_DIFF_HOME_AWAY']
    cat_reg = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, loss_function='RMSE', verbose=0, random_seed=42)
    goal_diff_preds_train = cross_val_predict(cat_reg, X_train_final, y_goal_diff, cv=5, n_jobs=-1)
    cat_reg.fit(X_train_final, y_goal_diff)
    X_train_final['PRED_GOAL_DIFF'] = goal_diff_preds_train
    X_test_final['PRED_GOAL_DIFF'] = cat_reg.predict(X_test_final)

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {'X_train': X_train_final.astype(np.float32), 'y': y_target,
             'X_test': X_test_final.astype(np.float32), 'ID_test': ID_test}
    pd.to_pickle(cache, CACHE_FILE)
    print(f"✅ Cache saved: {CACHE_FILE}  Train={X_train_final.shape} Test={X_test_final.shape}")
    return cache


def load_cache():
    if os.path.exists(CACHE_FILE):
        print(f"--- Loading feature cache {CACHE_FILE} ---")
        return pd.read_pickle(CACHE_FILE)
    return build_cache()


# =============================================================================
# 2. Models
# =============================================================================

def make_selector():
    return SelectFromModel(
        estimator=lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, importance_type='gain',
                                     random_state=42, verbose=-1, n_jobs=-1),
        max_features=N_SELECTED,
        threshold=-np.inf
    )


def make_model(name, args):
    if name == 'cat':
        return CatBoostClassifier(**PARAMS_CAT)
    if name == 'tabicl':
        return TabICLClassifier(n_estimators=args.tabicl_estimators, device=args.device,
                                batch_size=1, random_state=42,  # batch_size=1 avoids MPS out-of-memory
                                # On MPS, TabICL's free-memory estimate ignores other apps (unified memory)
                                # and its OOM retry does not catch MPS errors: shrink its auto batch sizes
                                inference_config={k: {'safety_factor': args.mem_safety}
                                                  for k in ('COL_CONFIG', 'ROW_CONFIG', 'ICL_CONFIG')})
    if name == 'tabpfn':
        os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', '1')
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device=args.device, ignore_pretraining_limits=True, random_state=42)
    raise ValueError(f"Unknown model: {name}")


def fit_predict(names, X_tr, y_tr, X_te, args, timings):
    selector = make_selector()
    X_tr_sel = selector.fit_transform(X_tr, y_tr)
    X_te_sel = selector.transform(X_te)
    # Foundation models: keep only the top-k of the same LGBM ranking (memory grows with rows x columns)
    importances = selector.estimator_.feature_importances_[selector.get_support()]
    top_fm = np.sort(np.argsort(importances)[::-1][:args.fm_features])
    probs = {}
    for name in names:
        t0 = time.time()
        model = make_model(name, args)
        cols = slice(None) if name == 'cat' else top_fm
        model.fit(X_tr_sel[:, cols], y_tr)
        X_pred = X_te_sel[:, cols]
        if name == 'cat':
            probs[name] = model.predict_proba(X_pred)
        else:
            # Predict in chunks of a CV-fold size: the full test set (25k rows) at once does not fit in RAM
            chunks = [model.predict_proba(X_pred[i:i + PREDICT_CHUNK]) for i in range(0, len(X_pred), PREDICT_CHUNK)]
            probs[name] = np.vstack(chunks)
        timings.setdefault(name, []).append(time.time() - t0)
        print(f"    {name:7s} {timings[name][-1]:7.1f}s  peak RAM {_peak_ram_gb():.1f} GB", flush=True)
    return probs


# =============================================================================
# 3. Main
# =============================================================================

if __name__ == "__main__":
    args = _parse_args()
    names = [m.strip() for m in args.models.split(',') if m.strip()]
    cache = load_cache()
    X, y = cache['X_train'], cache['y']

    if args.n_rows:
        X, _, y, _ = train_test_split(X, y, train_size=args.n_rows, stratify=y, random_state=42)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True).values
    print(f"Data: {X.shape}, models: {names}, device: {args.device}")

    timings = {}

    if args.submit:
        fm = [n for n in names if n != 'cat']
        assert 'cat' in names and len(fm) == 1, "--submit expects --models cat,<fm>"
        probs = fit_predict(names, X, y, cache['X_test'], args, timings)
        for n in names:
            np.save(os.path.join(CACHE_DIR, f"test_{n}.npy"), probs[n])
        blend = args.weight * probs[fm[0]] + (1 - args.weight) * probs['cat']
        preds = np.argmax(blend, axis=1)
        submission = pd.DataFrame({'ID': cache['ID_test'].values})
        submission['HOME_WINS'] = (preds == 2).astype(int)
        submission['DRAW'] = (preds == 1).astype(int)
        submission['AWAY_WINS'] = (preds == 0).astype(int)
        save_experiment(args.cv_score, {"models": names, "weight": args.weight},
                        f"Blend {fm[0]} w={args.weight} + cat", submission)
        sys.exit(0)

    # --- Cross-Validation (same folds as run_pipeline.py) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = {n: np.full((len(y), 3), np.nan) for n in names}
    done = np.zeros(len(y), dtype=bool)

    for k, (tr, te) in enumerate(cv.split(X, y)):
        if k >= args.folds:
            break
        print(f"--- Fold {k + 1}/{args.folds} (train={len(tr)}, val={len(te)}) ---", flush=True)
        probs = fit_predict(names, X.iloc[tr], y[tr], X.iloc[te], args, timings)
        for n in names:
            oof[n][te] = probs[n]
        done[te] = True
        for n in names:
            print(f"    {n:7s} fold acc = {accuracy_score(y[te], probs[n].argmax(1)):.4f}")

    # Save OOF probabilities (blends can be recomputed without retraining)
    tag = f"n{len(y)}_f{args.folds}"
    for n in names:
        np.save(os.path.join(CACHE_DIR, f"oof_{n}_{tag}.npy"), oof[n])
    np.save(os.path.join(CACHE_DIR, f"oof_y_{tag}.npy"), np.where(done, y, -1))

    # --- Report ---
    yd = y[done]
    print("\n=== Results (OOF) ===")
    results = {}
    for n in names:
        p = oof[n][done]
        results[n] = accuracy_score(yd, p.argmax(1))
        print(f"{n:22s} acc={results[n]:.4f}  logloss={log_loss(yd, p, labels=[0, 1, 2]):.4f}  "
              f"time/fold={np.mean(timings[n]):.0f}s")
    if 'cat' in names:
        for fm in [n for n in names if n != 'cat']:
            for w in (0.3, 0.5, 0.7):
                p = w * oof[fm][done] + (1 - w) * oof['cat'][done]
                key = f"blend {fm} w={w}"
                results[key] = accuracy_score(yd, p.argmax(1))
                print(f"{key:22s} acc={results[key]:.4f}  logloss={log_loss(yd, p, labels=[0, 1, 2]):.4f}")
    print(f"Peak RAM: {_peak_ram_gb():.1f} GB")

    if not args.n_rows and args.folds == 5:
        best = max(results, key=results.get)
        save_experiment(results[best], {"models": names, "results": results, "best": best},
                        f"Tabular FM benchmark (best: {best})", None)
