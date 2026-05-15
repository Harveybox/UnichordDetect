from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def top_features_from_coefficients(feature_values: dict[str, float], coefficients: dict[str, float], k: int = 5):
    contr = []
    for f, v in feature_values.items():
        c = coefficients.get(f, 0.0)
        contr.append((f, c * v))
    contr.sort(key=lambda x: abs(x[1]), reverse=True)
    return contr[:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="json object")
    parser.add_argument("--coefs", required=True, help="json object")
    args = parser.parse_args()

    fv = json.loads(args.features)
    cf = json.loads(args.coefs)

    top = top_features_from_coefficients(fv, cf)
    print(json.dumps({"top_features": top}, ensure_ascii=False))


if __name__ == "__main__":
    main()
