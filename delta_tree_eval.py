"""Quick A/B test for TEAM delta features (1N2 target).

- Baseline: raw team_home + team_away features
- Delta model: baseline + engineered deltas/ratios/efficiency features

Run:
    python delta_tree_eval.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from football import load_data

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


def main() -> None:
    (team_h, team_a, _ph, _pa, y_train_raw, _y_supp, *_rest) = load_data()

    X_base = build_team_base(team_h, team_a)
    y, _le = make_target_1n2(y_train_raw)

    print("\n--- Team-only baseline ---")
    res_base = fit_eval_lightgbm(X_base, y, name="baseline")
    print(f"baseline: acc={res_base.acc:.5f} | n_features={res_base.n_features}")

    print("\n--- Team-only + engineered deltas/ratios ---")
    X_eng = add_engineered_team_deltas(X_base)
    res_eng = fit_eval_lightgbm(X_eng, y, name="engineered")
    print(f"engineered: acc={res_eng.acc:.5f} | n_features={res_eng.n_features}")

    print("\nΔacc (engineered - baseline) = {:.5f}".format(res_eng.acc - res_base.acc))


if __name__ == "__main__":
    main()
