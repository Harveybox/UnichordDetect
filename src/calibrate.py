from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="csv with columns: y_true,p_raw")
    parser.add_argument("--out", default="artifacts/calibrator.joblib")
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.pred)
    y = df["y_true"].to_numpy()
    p = df["p_raw"].to_numpy()

    cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    cal.fit(p, y)
    p_cal = cal.predict(p)

    ece = float(np.mean(np.abs(p_cal - y)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, args.out)
    print(json.dumps({"ece_like": ece}, ensure_ascii=False))


if __name__ == "__main__":
    main()
