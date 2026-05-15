from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from features import build_features, feature_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--province", required=True)
    parser.add_argument("--outdir", default="artifacts/lgbm")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["province"] == args.province].copy()
    if df.empty:
        raise ValueError(f"no data for province={args.province}")

    df = build_features(df)
    df = pd.get_dummies(df, columns=["batch", "school", "major"], dummy_na=True)
    df = df.sort_values("year")
    holdout_year = df["year"].max()

    cols = [c for c in df.columns if c in feature_columns() or c.startswith(("batch_", "school_", "major_"))]

    train = df[df["year"] < holdout_year]
    test = df[df["year"] == holdout_year]

    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300)
    model.fit(train[cols], train["admit"])
    pred = model.predict_proba(test[cols])[:, 1]

    metrics = {
        "province": args.province,
        "holdout_year": int(holdout_year),
        "auc": float(roc_auc_score(test["admit"], pred)) if test["admit"].nunique() > 1 else None,
        "brier": float(brier_score_loss(test["admit"], pred)),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": cols}, outdir / f"lgbm_like_{args.province}.joblib")
    (outdir / f"metrics_{args.province}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
