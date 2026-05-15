from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def calibration_table(y_true: np.ndarray, p_pred: np.ndarray, bins: int = 10) -> pd.DataFrame:
    b = pd.qcut(p_pred, q=bins, duplicates="drop")
    df = pd.DataFrame({"y": y_true, "p": p_pred, "bin": b})
    g = df.groupby("bin", observed=False)
    return g.agg(pred_mean=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size")).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="csv with columns: province,y_true,p_pred")
    parser.add_argument("--out", default="reports/province_backtest_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.pred)
    rows = []
    for prov, sub in df.groupby("province"):
        brier = brier_score_loss(sub["y_true"], sub["p_pred"])
        cal = calibration_table(sub["y_true"].to_numpy(), sub["p_pred"].to_numpy(), bins=10)
        mae_cal = float((cal["pred_mean"] - cal["obs_rate"]).abs().mean())
        rows.append({"province": prov, "brier": brier, "calibration_mae": mae_cal, "n": len(sub)})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(json.dumps({"rows": len(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
