from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import build_features, feature_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--province", required=True)
    parser.add_argument("--outdir", default="artifacts/logit")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["province"] == args.province].copy()
    if df.empty:
        raise ValueError(f"no data for province={args.province}")

    df = build_features(df)
    df = df.sort_values("year")
    holdout_year = df["year"].max()

    train = df[df["year"] < holdout_year]
    test = df[df["year"] == holdout_year]

    num_cols = feature_columns()
    cat_cols = ["batch", "school", "major"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    model.fit(train[num_cols + cat_cols], train["admit"])
    pred = model.predict_proba(test[num_cols + cat_cols])[:, 1]

    metrics = {
        "province": args.province,
        "holdout_year": int(holdout_year),
        "auc": float(roc_auc_score(test["admit"], pred)) if test["admit"].nunique() > 1 else None,
        "brier": float(brier_score_loss(test["admit"], pred)),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outdir / f"logit_{args.province}.joblib")
    (outdir / f"metrics_{args.province}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
