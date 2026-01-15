"""Quick A/B test for TEAM delta features (1N2 target).

Two modes:
- Simple holdout (legacy): baseline vs engineered, one split.
- N-fold validation: baseline vs engineered + aggregated importances to spot the most useful deltas.

Run examples:
    python delta_tree_eval.py
    python delta_tree_eval.py --n-splits 5 --n-repeats 2 --top-k 30 --importance gain
    python delta_tree_eval.py --mode cv --n-splits 5 --ablation dropone --ablation-k 15
    python delta_tree_eval.py --source football --mode cv --n-splits 5 --top-k 40 --importance gain
    python delta_tree_eval.py --source football --mode cv --n-splits 5 --n-repeats 2 --ablation topk --topk-scope all --ablation-k 800
    python delta_tree_eval.py --source football --mode cv --n-splits 5 --n-repeats 2 --ablation topk --topk-scope all --topk-grid 100,200,400,800,1200
"""

from __future__ import annotations

import warnings
import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from football import load_data, build_features_v2

warnings.filterwarnings("ignore")


EPS = 1e-6


TEAM_METRICS = [
    "TEAM_ATTACKS",
    "TEAM_BALL_POSSESSION",
    "TEAM_BALL_SAFE",
    "TEAM_CORNERS",
    "TEAM_DANGEROUS_ATTACKS",
    "TEAM_FOULS",
    "TEAM_GAME_DRAW",
    "TEAM_GAME_LOST",
    "TEAM_GAME_WON",
    "TEAM_GOALS",
    "TEAM_INJURIES",
    "TEAM_OFFSIDES",
    "TEAM_PASSES",
    "TEAM_PENALTIES",
    "TEAM_REDCARDS",
    "TEAM_SAVES",
    "TEAM_SHOTS_INSIDEBOX",
    "TEAM_SHOTS_OFF_TARGET",
    "TEAM_SHOTS_ON_TARGET",
    "TEAM_SHOTS_OUTSIDEBOX",
    "TEAM_SHOTS_TOTAL",
    "TEAM_SUBSTITUTIONS",
    "TEAM_SUCCESSFUL_PASSES",
    "TEAM_SUCCESSFUL_PASSES_PERCENTAGE",
    "TEAM_YELLOWCARDS",
]


@dataclass(frozen=True)
class ModelResult:
    name: str
    n_features: int
    acc: float


@dataclass(frozen=True)
class CVResult:
    name: str
    n_features: int
    acc_mean: float
    acc_std: float


@dataclass(frozen=True)
class AblationRow:
    feature: str
    acc_mean: float
    acc_std: float
    delta_vs_full: float


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b + EPS)


def _col(base: str, side: str) -> str:
    return f"{base}_{side}"


def build_team_base(team_home: pd.DataFrame, team_away: pd.DataFrame) -> pd.DataFrame:
    df = team_home.merge(team_away, on="ID", suffixes=("_HOME", "_AWAY"))
    df = df.copy()

    # Drop obvious text columns if present
    drop_cols = [
        "LEAGUE_HOME",
        "TEAM_NAME_HOME",
        "LEAGUE_AWAY",
        "TEAM_NAME_AWAY",
        "LEAGUE",
        "TEAM_NAME",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Keep numeric + ID
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "ID" in df.columns and "ID" not in numeric_cols:
        numeric_cols = ["ID"] + numeric_cols
    df = df[numeric_cols]
    df.fillna(0, inplace=True)
    return df


def build_football_features(
    team_home: pd.DataFrame,
    team_away: pd.DataFrame,
    player_home: pd.DataFrame,
    player_away: pd.DataFrame,
    *,
    drop_deep_quality: bool,
) -> pd.DataFrame:
    df = build_features_v2(team_home, team_away, player_home, player_away)

    if drop_deep_quality:
        # Features créées par le bloc "2. DEEP QUALITY" dans football.py
        cols = [
            'HOME_EFFICIENCY',
            'AWAY_EFFICIENCY',
            'HOME_ACCURACY',
            'AWAY_ACCURACY',
            'HOME_GK_RESISTANCE',
            'AWAY_GK_RESISTANCE',
        ]
        df.drop(columns=[c for c in cols if c in df.columns], inplace=True)

    df = df.select_dtypes(include=[np.number]).copy()
    df.fillna(0, inplace=True)
    return df


def _scopes_for_metric(df_cols: Iterable[str], metric: str) -> list[str]:
    # metric is like TEAM_SHOTS_TOTAL
    # columns are like TEAM_SHOTS_TOTAL_season_sum_HOME
    scopes = set()
    for c in df_cols:
        if not c.endswith("_HOME"):
            continue
        if not c.startswith(metric + "_"):
            continue
        base = c[: -len("_HOME")]
        scopes.add(base[len(metric) + 1 :])  # text after 'METRIC_'
    return sorted(scopes)


def add_engineered_team_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add deltas/ratios and a few interaction-style features.

    df is the merged team dataframe with suffixes _HOME/_AWAY.
    """

    out = df.copy()

    # ------------------------------------------------------------------
    # 1) Classic deltas for the requested TEAM_* metrics only
    # ------------------------------------------------------------------
    for metric in TEAM_METRICS:
        scopes = _scopes_for_metric(out.columns, metric)
        for scope in scopes:
            base = f"{metric}_{scope}"
            h = _col(base, "HOME")
            a = _col(base, "AWAY")
            if h in out.columns and a in out.columns:
                out[f"DELTA_{base}"] = out[h] - out[a]
                out[f"RATIO_{base}"] = _safe_div(out[h], out[a])

    # ------------------------------------------------------------------
    # 2) Possession share (possession is % average/std only)
    # ------------------------------------------------------------------
    for scope in _scopes_for_metric(out.columns, "TEAM_BALL_POSSESSION"):
        base = f"TEAM_BALL_POSSESSION_{scope}"
        h = _col(base, "HOME")
        a = _col(base, "AWAY")
        if h in out.columns and a in out.columns:
            out[f"HOME_POSSESSION_SHARE_{scope}"] = _safe_div(out[h], out[h] + out[a])
            out[f"DELTA_POSSESSION_{scope}"] = out[h] - out[a]

    # ------------------------------------------------------------------
    # 3) Efficiency / quality features (by scope)
    # ------------------------------------------------------------------
    scopes = set(_scopes_for_metric(out.columns, "TEAM_SHOTS_TOTAL"))
    scopes |= set(_scopes_for_metric(out.columns, "TEAM_GOALS"))
    scopes |= set(_scopes_for_metric(out.columns, "TEAM_SHOTS_ON_TARGET"))
    scopes |= set(_scopes_for_metric(out.columns, "TEAM_SHOTS_INSIDEBOX"))
    scopes |= set(_scopes_for_metric(out.columns, "TEAM_ATTACKS"))
    scopes |= set(_scopes_for_metric(out.columns, "TEAM_DANGEROUS_ATTACKS"))
    scopes |= set(_scopes_for_metric(out.columns, "TEAM_PASSES"))
    scopes |= set(_scopes_for_metric(out.columns, "TEAM_SUCCESSFUL_PASSES"))
    scopes = sorted(scopes)

    for scope in scopes:
        shots = f"TEAM_SHOTS_TOTAL_{scope}"
        sot = f"TEAM_SHOTS_ON_TARGET_{scope}"
        inside = f"TEAM_SHOTS_INSIDEBOX_{scope}"
        goals = f"TEAM_GOALS_{scope}"
        attacks = f"TEAM_ATTACKS_{scope}"
        dang = f"TEAM_DANGEROUS_ATTACKS_{scope}"
        passes = f"TEAM_PASSES_{scope}"
        succ = f"TEAM_SUCCESSFUL_PASSES_{scope}"
        saves = f"TEAM_SAVES_{scope}"

        h_shots, a_shots = _col(shots, "HOME"), _col(shots, "AWAY")
        h_sot, a_sot = _col(sot, "HOME"), _col(sot, "AWAY")
        h_inside, a_inside = _col(inside, "HOME"), _col(inside, "AWAY")
        h_goals, a_goals = _col(goals, "HOME"), _col(goals, "AWAY")
        h_att, a_att = _col(attacks, "HOME"), _col(attacks, "AWAY")
        h_dang, a_dang = _col(dang, "HOME"), _col(dang, "AWAY")
        h_pass, a_pass = _col(passes, "HOME"), _col(passes, "AWAY")
        h_succ, a_succ = _col(succ, "HOME"), _col(succ, "AWAY")
        h_saves, a_saves = _col(saves, "HOME"), _col(saves, "AWAY")

        # Shot accuracy: SOT / total
        if h_sot in out.columns and h_shots in out.columns:
            out[f"HOME_SHOT_ACCURACY_{scope}"] = _safe_div(out[h_sot], out[h_shots])
        if a_sot in out.columns and a_shots in out.columns:
            out[f"AWAY_SHOT_ACCURACY_{scope}"] = _safe_div(out[a_sot], out[a_shots])
        if f"HOME_SHOT_ACCURACY_{scope}" in out.columns and f"AWAY_SHOT_ACCURACY_{scope}" in out.columns:
            out[f"DELTA_SHOT_ACCURACY_{scope}"] = out[f"HOME_SHOT_ACCURACY_{scope}"] - out[
                f"AWAY_SHOT_ACCURACY_{scope}"
            ]

        # Conversion: goals / total
        if h_goals in out.columns and h_shots in out.columns:
            out[f"HOME_CONVERSION_{scope}"] = _safe_div(out[h_goals], out[h_shots])
        if a_goals in out.columns and a_shots in out.columns:
            out[f"AWAY_CONVERSION_{scope}"] = _safe_div(out[a_goals], out[a_shots])
        if f"HOME_CONVERSION_{scope}" in out.columns and f"AWAY_CONVERSION_{scope}" in out.columns:
            out[f"DELTA_CONVERSION_{scope}"] = out[f"HOME_CONVERSION_{scope}"] - out[f"AWAY_CONVERSION_{scope}"]

        # Inside box share
        if h_inside in out.columns and h_shots in out.columns:
            out[f"HOME_INSIDEBOX_SHARE_{scope}"] = _safe_div(out[h_inside], out[h_shots])
        if a_inside in out.columns and a_shots in out.columns:
            out[f"AWAY_INSIDEBOX_SHARE_{scope}"] = _safe_div(out[a_inside], out[a_shots])
        if f"HOME_INSIDEBOX_SHARE_{scope}" in out.columns and f"AWAY_INSIDEBOX_SHARE_{scope}" in out.columns:
            out[f"DELTA_INSIDEBOX_SHARE_{scope}"] = out[f"HOME_INSIDEBOX_SHARE_{scope}"] - out[
                f"AWAY_INSIDEBOX_SHARE_{scope}"
            ]

        # Danger ratio: dangerous attacks / attacks
        if h_dang in out.columns and h_att in out.columns:
            out[f"HOME_DANGER_RATIO_{scope}"] = _safe_div(out[h_dang], out[h_att])
        if a_dang in out.columns and a_att in out.columns:
            out[f"AWAY_DANGER_RATIO_{scope}"] = _safe_div(out[a_dang], out[a_att])
        if f"HOME_DANGER_RATIO_{scope}" in out.columns and f"AWAY_DANGER_RATIO_{scope}" in out.columns:
            out[f"DELTA_DANGER_RATIO_{scope}"] = out[f"HOME_DANGER_RATIO_{scope}"] - out[f"AWAY_DANGER_RATIO_{scope}"]

        # Pass completion: successful passes / passes
        if h_succ in out.columns and h_pass in out.columns:
            out[f"HOME_PASS_COMPLETION_{scope}"] = _safe_div(out[h_succ], out[h_pass])
        if a_succ in out.columns and a_pass in out.columns:
            out[f"AWAY_PASS_COMPLETION_{scope}"] = _safe_div(out[a_succ], out[a_pass])
        if f"HOME_PASS_COMPLETION_{scope}" in out.columns and f"AWAY_PASS_COMPLETION_{scope}" in out.columns:
            out[f"DELTA_PASS_COMPLETION_{scope}"] = out[f"HOME_PASS_COMPLETION_{scope}"] - out[
                f"AWAY_PASS_COMPLETION_{scope}"
            ]

        # Cross Attack vs Defense proxies
        # - Home attack vs away GK
        if h_sot in out.columns and a_saves in out.columns:
            out[f"CROSS_HOME_SOT_minus_AWAY_SAVES_{scope}"] = out[h_sot] - out[a_saves]
        # - Away attack vs home GK
        if a_sot in out.columns and h_saves in out.columns:
            out[f"CROSS_AWAY_SOT_minus_HOME_SAVES_{scope}"] = out[a_sot] - out[h_saves]

        # Saves per shots-on-target faced (approx)
        if h_saves in out.columns and a_sot in out.columns:
            out[f"HOME_SAVES_PER_SOT_FACED_{scope}"] = _safe_div(out[h_saves], out[a_sot])
        if a_saves in out.columns and h_sot in out.columns:
            out[f"AWAY_SAVES_PER_SOT_FACED_{scope}"] = _safe_div(out[a_saves], out[h_sot])
        if f"HOME_SAVES_PER_SOT_FACED_{scope}" in out.columns and f"AWAY_SAVES_PER_SOT_FACED_{scope}" in out.columns:
            out[f"DELTA_SAVES_PER_SOT_FACED_{scope}"] = out[f"HOME_SAVES_PER_SOT_FACED_{scope}"] - out[
                f"AWAY_SAVES_PER_SOT_FACED_{scope}"
            ]

    # Clean: keep numeric only
    out = out.select_dtypes(include=[np.number]).copy()
    out.fillna(0, inplace=True)

    # Drop quasi-empty features (very sparse) to keep the benchmark stable
    out = out.loc[:, (out != 0).any(axis=0)]
    zero_frac = (out == 0).mean()
    out = out.loc[:, zero_frac < 0.999]

    return out


def make_target_1n2(y_train_raw: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    # Expected: columns ['AWAY_WINS','DRAW','HOME_WINS'] are 0/1
    y_classes = y_train_raw[["AWAY_WINS", "DRAW", "HOME_WINS"]].idxmax(axis=1)
    le = LabelEncoder()
    y = le.fit_transform(y_classes)
    return y, le


def fit_eval_lightgbm(X: pd.DataFrame, y: np.ndarray, *, name: str) -> ModelResult:
    X_ = X.copy()
    if "ID" in X_.columns:
        X_ = X_.drop(columns=["ID"])

    X_train, X_val, y_train, y_val = train_test_split(
        X_,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    pred = clf.predict(X_val)
    acc = float(accuracy_score(y_val, pred))
    return ModelResult(name=name, n_features=int(X_.shape[1]), acc=acc)


def cross_validate_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    name: str,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    importance_type: str,
) -> tuple[CVResult, pd.DataFrame]:
    """Stratified K-fold CV with aggregated feature importances.

    Returns:
    - CVResult (mean/std accuracy)
    - DataFrame with mean feature importance across folds (all folds from all repeats)
    """

    X_ = X.copy()
    if "ID" in X_.columns:
        X_ = X_.drop(columns=["ID"])

    X_.fillna(0, inplace=True)
    feature_names = list(X_.columns)

    accs: list[float] = []
    importance_sum = np.zeros(len(feature_names), dtype=np.float64)
    n_models = 0

    if importance_type not in {"gain", "split"}:
        raise ValueError("importance_type must be 'gain' or 'split'")

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + rep)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_, y), start=1):
            X_train, X_val = X_.iloc[train_idx], X_.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            clf = lgb.LGBMClassifier(
                n_estimators=4000,
                learning_rate=0.02,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=random_state + rep,
                verbose=-1,
            )

            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(150, verbose=False)],
            )

            pred = clf.predict(X_val)
            accs.append(float(accuracy_score(y_val, pred)))

            booster = clf.booster_
            imp = booster.feature_importance(importance_type=importance_type)
            if len(imp) == len(feature_names):
                importance_sum += imp
                n_models += 1
            else:
                # Should not happen, but keep robust.
                n_models += 1

            if (rep == 0 and fold_idx == 1) or (fold_idx == n_splits and rep == n_repeats - 1):
                # Very light progress signal
                pass

    acc_mean = float(np.mean(accs))
    acc_std = float(np.std(accs))
    cv = CVResult(name=name, n_features=int(X_.shape[1]), acc_mean=acc_mean, acc_std=acc_std)

    imp_mean = (importance_sum / max(1, n_models)).astype(np.float64)
    imp_df = pd.DataFrame({"feature": feature_names, f"importance_{importance_type}": imp_mean})
    imp_df.sort_values(by=f"importance_{importance_type}", ascending=False, inplace=True)
    imp_df.reset_index(drop=True, inplace=True)
    return cv, imp_df


def _print_top_deltas(imp_df: pd.DataFrame, *, top_k: int, importance_type: str) -> None:
    col = f"importance_{importance_type}"
    is_delta = imp_df["feature"].str.startswith(("DELTA_", "CROSS_", "RATIO_"))
    top = imp_df[is_delta].head(top_k)
    if top.empty:
        print("\nAucun feature DELTA_/CROSS_/RATIO_ trouvé dans les importances.")
        return

    print(f"\nTop {min(top_k, len(top))} deltas/ratios (importance='{importance_type}') :")
    for i, row in enumerate(top.itertuples(index=False), start=1):
        feat = getattr(row, "feature")
        val = getattr(row, col)
        print(f"{i:>2}. {feat}: {val:.4f}")


def _get_top_delta_features(imp_df: pd.DataFrame, *, top_k: int) -> list[str]:
    is_delta = imp_df["feature"].str.startswith(("DELTA_", "CROSS_", "RATIO_"))
    feats = imp_df.loc[is_delta, "feature"].head(top_k).tolist()
    # Keep unique order
    seen: set[str] = set()
    out: list[str] = []
    for f in feats:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def _get_top_all_features(imp_df: pd.DataFrame, *, top_k: int) -> list[str]:
    feats = imp_df["feature"].tolist()
    feats = [f for f in feats if f != "ID"]
    feats = feats[:top_k]
    # Keep unique order
    seen: set[str] = set()
    out: list[str] = []
    for f in feats:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def _select_base_plus_features(X_eng: pd.DataFrame, *, extra_features: list[str]) -> pd.DataFrame:
    """Return a view of X_eng keeping base (non-engineered) + selected engineered features.

    Base is defined as all non DELTA_/RATIO_/CROSS_ numeric columns in X_eng.
    """
    cols = list(X_eng.columns)
    base_cols = [
        c
        for c in cols
        if c == "ID" or not (c.startswith("DELTA_") or c.startswith("RATIO_") or c.startswith("CROSS_"))
    ]
    keep = base_cols + [c for c in extra_features if c in X_eng.columns]
    # Deduplicate while preserving order
    seen: set[str] = set()
    keep2: list[str] = []
    for c in keep:
        if c not in seen:
            keep2.append(c)
            seen.add(c)
    out = X_eng[keep2].copy()
    out.fillna(0, inplace=True)
    return out


def _select_only_features(X: pd.DataFrame, *, keep_features: list[str]) -> pd.DataFrame:
    cols = [c for c in keep_features if c in X.columns and c != "ID"]
    # Deduplicate while preserving order
    seen: set[str] = set()
    cols2: list[str] = []
    for c in cols:
        if c not in seen:
            cols2.append(c)
            seen.add(c)
    out = X[cols2].copy()
    out.fillna(0, inplace=True)
    return out


def _parse_int_list(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    # unique, preserve order
    seen: set[int] = set()
    out2: list[int] = []
    for v in out:
        if v not in seen:
            out2.append(v)
            seen.add(v)
    return out2


def run_ablation_dropone(
    *,
    X_eng: pd.DataFrame,
    y: np.ndarray,
    top_feats: list[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
    importance_type: str,
) -> None:
    """Ablation study: compute CV score with baseline+top_feats, then remove each feature."""

    if not top_feats:
        print("\nAblation: aucun delta sélectionné (top_feats vide).")
        return

    X_top = _select_base_plus_features(X_eng, extra_features=top_feats)
    cv_full, _ = cross_validate_lightgbm(
        X_top,
        y,
        name=f"top{len(top_feats)}",
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
        importance_type=importance_type,
    )
    print(
        f"\n--- Ablation drop-one sur top{len(top_feats)} deltas ---\n"
        f"top{len(top_feats)}: acc={cv_full.acc_mean:.5f} ± {cv_full.acc_std:.5f} | n_features={cv_full.n_features}"
    )

    rows: list[AblationRow] = []
    for feat in top_feats:
        reduced = [f for f in top_feats if f != feat]
        X_red = _select_base_plus_features(X_eng, extra_features=reduced)
        cv_red, _ = cross_validate_lightgbm(
            X_red,
            y,
            name=f"minus_{feat}",
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=seed,
            importance_type=importance_type,
        )
        rows.append(
            AblationRow(
                feature=feat,
                acc_mean=cv_red.acc_mean,
                acc_std=cv_red.acc_std,
                delta_vs_full=cv_red.acc_mean - cv_full.acc_mean,
            )
        )

    # Sort by biggest drop when removed (most negative delta)
    rows_sorted = sorted(rows, key=lambda r: r.delta_vs_full)
    print("\nImpact du retrait (acc_sans_feature - acc_full). Plus c'est négatif, plus la feature aide :")
    for i, r in enumerate(rows_sorted[: min(30, len(rows_sorted))], start=1):
        print(f"{i:>2}. {r.feature}: Δ={r.delta_vs_full:+.5f} | acc={r.acc_mean:.5f} ± {r.acc_std:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark deltas TEAM_* for 1N2")
    parser.add_argument("--n-splits", type=int, default=5, help="Nombre de folds CV (StratifiedKFold)")
    parser.add_argument("--n-repeats", type=int, default=1, help="Nombre de répétitions de la CV")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    parser.add_argument("--top-k", type=int, default=25, help="Top-K features delta/ratio à afficher")
    parser.add_argument(
        "--importance",
        type=str,
        default="gain",
        choices=["gain", "split"],
        help="Type d'importance LightGBM à agréger",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cv",
        choices=["cv", "holdout"],
        help="'cv' (N validation) ou 'holdout' (split unique)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="team",
        choices=["team", "football"],
        help="'team' = TEAM stats seulement; 'football' = features avec deltas déjà construits dans football.py",
    )
    parser.add_argument(
        "--football-drop-deep-quality",
        action="store_true",
        help="En mode --source football, supprime les features du bloc 'DEEP QUALITY' (test A/B).",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "topk", "dropone"],
        help=(
            "Ablation study en CV: 'topk' (évalue un sous-ensemble topK) ou 'dropone' (retire 1 par 1). "
            "Par défaut topK = deltas/ratios; configurable via --topk-scope."
        ),
    )
    parser.add_argument(
        "--ablation-k",
        type=int,
        default=15,
        help="Nombre de deltas/ratios à utiliser pour l'ablation (top-K par importance)",
    )
    parser.add_argument(
        "--topk-scope",
        type=str,
        default="delta",
        choices=["delta", "all"],
        help="Scope du topK: 'delta' = DELTA/RATIO/CROSS uniquement, 'all' = toutes les features (topK-only).",
    )
    parser.add_argument(
        "--topk-grid",
        type=str,
        default="",
        help="Optionnel: liste de K (ex '100,200,400,800') pour tester plusieurs tailles topK en une commande.",
    )
    args = parser.parse_args()

    (team_h, team_a, p_h, p_a, y_train_raw, _y_supp, *_rest) = load_data()
    y, _le = make_target_1n2(y_train_raw)

    if args.source == "football":
        X_fb = build_football_features(team_h, team_a, p_h, p_a, drop_deep_quality=bool(args.football_drop_deep_quality))

        if args.mode == "holdout":
            print("\n--- football.py features (holdout) ---")
            res_fb = fit_eval_lightgbm(X_fb, y, name="football_features")
            print(f"football: acc={res_fb.acc:.5f} | n_features={res_fb.n_features}")
            return

        print(f"\n--- CV: football.py features (n_splits={args.n_splits}, n_repeats={args.n_repeats}) ---")
        cv_fb, imp_fb = cross_validate_lightgbm(
            X_fb,
            y,
            name="football_features",
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.seed,
            importance_type=args.importance,
        )
        print(f"football: acc={cv_fb.acc_mean:.5f} ± {cv_fb.acc_std:.5f} | n_features={cv_fb.n_features}")

        _print_top_deltas(imp_fb, top_k=args.top_k, importance_type=args.importance)

        if args.ablation != "none":
            grid = _parse_int_list(args.topk_grid)
            if grid:
                print("\n--- TopK grid (scope='{}') ---".format(args.topk_scope))
                best_k = None
                best_acc = -1.0
                for k0 in grid:
                    k = int(max(1, k0))
                    if args.topk_scope == "all":
                        feats = _get_top_all_features(imp_fb, top_k=k)
                        X_top = _select_only_features(X_fb, keep_features=feats)
                        name = f"football_top{k}_only"
                    else:
                        feats = _get_top_delta_features(imp_fb, top_k=k)
                        X_top = _select_base_plus_features(X_fb, extra_features=feats)
                        name = f"football_base+top{k}_delta"

                    cv_top, _ = cross_validate_lightgbm(
                        X_top,
                        y,
                        name=name,
                        n_splits=args.n_splits,
                        n_repeats=args.n_repeats,
                        random_state=args.seed,
                        importance_type=args.importance,
                    )
                    print(f"{name}: acc={cv_top.acc_mean:.5f} ± {cv_top.acc_std:.5f} | n_features={cv_top.n_features}")
                    if cv_top.acc_mean > best_acc:
                        best_acc = cv_top.acc_mean
                        best_k = k

                if best_k is not None:
                    print(f"Best K={best_k} with acc={best_acc:.5f} (vs full={cv_fb.acc_mean:.5f})")
                return

            k = int(max(1, args.ablation_k))
            if args.topk_scope == "all":
                top_feats = _get_top_all_features(imp_fb, top_k=k)
            else:
                top_feats = _get_top_delta_features(imp_fb, top_k=k)

            if args.ablation == "topk":
                if args.topk_scope == "all":
                    X_top = _select_only_features(X_fb, keep_features=top_feats)
                    label = f"football_top{k}_only"
                else:
                    X_top = _select_base_plus_features(X_fb, extra_features=top_feats)
                    label = f"football_base+top{k}_delta"

                cv_top, _ = cross_validate_lightgbm(
                    X_top,
                    y,
                    name=label,
                    n_splits=args.n_splits,
                    n_repeats=args.n_repeats,
                    random_state=args.seed,
                    importance_type=args.importance,
                )
                print(
                    f"\n--- Ablation topK (scope='{args.topk_scope}') ---\n"
                    f"{label}: acc={cv_top.acc_mean:.5f} ± {cv_top.acc_std:.5f} | n_features={cv_top.n_features}"
                )
                print("Δacc vs full football set = {:.5f}".format(cv_top.acc_mean - cv_fb.acc_mean))
            elif args.ablation == "dropone":
                if args.topk_scope != "delta":
                    print("\nAblation drop-one est seulement supportée pour topk-scope='delta'.")
                else:
                    run_ablation_dropone(
                        X_eng=X_fb,
                        y=y,
                        top_feats=top_feats,
                        n_splits=args.n_splits,
                        n_repeats=args.n_repeats,
                        seed=args.seed,
                        importance_type=args.importance,
                    )
        return

    # Default source: team-only dataset + engineered deltas/ratios
    X_base = build_team_base(team_h, team_a)

    if args.mode == "holdout":
        print("\n--- Team-only baseline (holdout) ---")
        res_base = fit_eval_lightgbm(X_base, y, name="baseline")
        print(f"baseline: acc={res_base.acc:.5f} | n_features={res_base.n_features}")

        print("\n--- Team-only + engineered deltas/ratios (holdout) ---")
        X_eng = add_engineered_team_deltas(X_base)
        res_eng = fit_eval_lightgbm(X_eng, y, name="engineered")
        print(f"engineered: acc={res_eng.acc:.5f} | n_features={res_eng.n_features}")
        print("\nΔacc (engineered - baseline) = {:.5f}".format(res_eng.acc - res_base.acc))
        return

    print(f"\n--- CV: baseline (n_splits={args.n_splits}, n_repeats={args.n_repeats}) ---")
    cv_base, _imp_base = cross_validate_lightgbm(
        X_base,
        y,
        name="baseline",
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
        importance_type=args.importance,
    )
    print(f"baseline: acc={cv_base.acc_mean:.5f} ± {cv_base.acc_std:.5f} | n_features={cv_base.n_features}")

    print(f"\n--- CV: engineered (n_splits={args.n_splits}, n_repeats={args.n_repeats}) ---")
    X_eng = add_engineered_team_deltas(X_base)
    cv_eng, imp_eng = cross_validate_lightgbm(
        X_eng,
        y,
        name="engineered",
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
        importance_type=args.importance,
    )
    print(f"engineered: acc={cv_eng.acc_mean:.5f} ± {cv_eng.acc_std:.5f} | n_features={cv_eng.n_features}")

    delta = cv_eng.acc_mean - cv_base.acc_mean
    print("\nΔacc moyen (engineered - baseline) = {:.5f}".format(delta))

    _print_top_deltas(imp_eng, top_k=args.top_k, importance_type=args.importance)

    if args.ablation != "none":
        grid = _parse_int_list(args.topk_grid)
        if grid:
            print("\n--- TopK grid (scope='{}') ---".format(args.topk_scope))
            best_k = None
            best_acc = -1.0
            for k0 in grid:
                k = int(max(1, k0))
                if args.topk_scope == "all":
                    feats = _get_top_all_features(imp_eng, top_k=k)
                    X_top = _select_only_features(X_eng, keep_features=feats)
                    name = f"engineered_top{k}_only"
                else:
                    feats = _get_top_delta_features(imp_eng, top_k=k)
                    X_top = _select_base_plus_features(X_eng, extra_features=feats)
                    name = f"baseline+top{k}_delta"

                cv_top, _ = cross_validate_lightgbm(
                    X_top,
                    y,
                    name=name,
                    n_splits=args.n_splits,
                    n_repeats=args.n_repeats,
                    random_state=args.seed,
                    importance_type=args.importance,
                )
                print(f"{name}: acc={cv_top.acc_mean:.5f} ± {cv_top.acc_std:.5f} | n_features={cv_top.n_features}")
                if cv_top.acc_mean > best_acc:
                    best_acc = cv_top.acc_mean
                    best_k = k
            if best_k is not None:
                print(f"Best K={best_k} with acc={best_acc:.5f} (vs full={cv_eng.acc_mean:.5f})")
            return

        k = int(max(1, args.ablation_k))
        if args.topk_scope == "all":
            top_feats = _get_top_all_features(imp_eng, top_k=k)
        else:
            top_feats = _get_top_delta_features(imp_eng, top_k=k)
        if args.ablation == "topk":
            if args.topk_scope == "all":
                X_top = _select_only_features(X_eng, keep_features=top_feats)
                label = f"engineered_top{k}_only"
            else:
                X_top = _select_base_plus_features(X_eng, extra_features=top_feats)
                label = f"baseline+top{k}_delta"
            cv_top, _ = cross_validate_lightgbm(
                X_top,
                y,
                name=label,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                random_state=args.seed,
                importance_type=args.importance,
            )
            print(
                f"\n--- Ablation topK (scope='{args.topk_scope}') ---\n"
                f"{label}: acc={cv_top.acc_mean:.5f} ± {cv_top.acc_std:.5f} | n_features={cv_top.n_features}"
            )
            print("Δacc vs baseline = {:.5f}".format(cv_top.acc_mean - cv_base.acc_mean))
            print("Δacc vs full engineered = {:.5f}".format(cv_top.acc_mean - cv_eng.acc_mean))
        elif args.ablation == "dropone":
            if args.topk_scope != "delta":
                print("\nAblation drop-one est seulement supportée pour topk-scope='delta'.")
            else:
                run_ablation_dropone(
                    X_eng=X_eng,
                    y=y,
                    top_feats=top_feats,
                    n_splits=args.n_splits,
                    n_repeats=args.n_repeats,
                    seed=args.seed,
                    importance_type=args.importance,
                )


if __name__ == "__main__":
    main()
