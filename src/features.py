from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "province",
    "year",
    "batch",
    "school",
    "major",
    "candidate_rank",
    "major_min_rank",
    "major_avg_rank",
    "plan_count",
    "plan_yoy_change",
    "heat_index",
    "admit",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready features. Input should be already province-consistent."""
    validate_columns(df)
    out = df.copy()

    out = out.sort_values(["province", "batch", "school", "major", "year"])
    grp = ["province", "batch", "school", "major"]

    out["major_min_rank_t1"] = out.groupby(grp)["major_min_rank"].shift(1)
    out["major_min_rank_t2"] = out.groupby(grp)["major_min_rank"].shift(2)
    out["major_min_rank_t3"] = out.groupby(grp)["major_min_rank"].shift(3)

    out["major_avg_rank_t1"] = out.groupby(grp)["major_avg_rank"].shift(1)
    out["major_avg_rank_t2"] = out.groupby(grp)["major_avg_rank"].shift(2)
    out["major_avg_rank_t3"] = out.groupby(grp)["major_avg_rank"].shift(3)

    out["plan_count_t1"] = out.groupby(grp)["plan_count"].shift(1)
    out["plan_count_t2"] = out.groupby(grp)["plan_count"].shift(2)
    out["plan_count_t3"] = out.groupby(grp)["plan_count"].shift(3)

    out["heat_index_t1"] = out.groupby(grp)["heat_index"].shift(1)

    out["rank_gap_vs_last_min"] = out["candidate_rank"] - out["major_min_rank_t1"]
    out["rank_gap_vs_avg"] = out["candidate_rank"] - out["major_avg_rank_t1"]

    min_trend = (out["major_min_rank_t1"] + out["major_min_rank_t2"] + out["major_min_rank_t3"]) / 3.0
    out["rank_gap_vs_trend"] = out["candidate_rank"] - min_trend

    out["plan_3y_mean"] = (out["plan_count_t1"] + out["plan_count_t2"] + out["plan_count_t3"]) / 3.0
    out["plan_3y_std"] = out[["plan_count_t1", "plan_count_t2", "plan_count_t3"]].std(axis=1)
    out["plan_3y_volatility"] = out["plan_3y_std"] / out["plan_3y_mean"].replace(0, pd.NA)

    out["heat_yoy_change"] = (out["heat_index"] - out["heat_index_t1"]) / out["heat_index_t1"].replace(0, pd.NA)

    # fill numeric NA with group medians then global medians
    num_cols = out.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        out[col] = out[col].fillna(out.groupby(["province", "batch"])[col].transform("median"))
        out[col] = out[col].fillna(out[col].median())

    return out


def feature_columns() -> list[str]:
    return [
        "candidate_rank",
        "major_min_rank_t1",
        "major_min_rank_t2",
        "major_min_rank_t3",
        "major_avg_rank_t1",
        "major_avg_rank_t2",
        "major_avg_rank_t3",
        "rank_gap_vs_last_min",
        "rank_gap_vs_avg",
        "rank_gap_vs_trend",
        "plan_count",
        "plan_yoy_change",
        "plan_3y_volatility",
        "heat_index",
        "heat_yoy_change",
        "year",
    ]
