from __future__ import annotations

import numpy as np


def simulate_probability(
    base_rank_gap: float,
    plan_yoy_change: float,
    heat_yoy_change: float,
    n_samples: int = 3000,
    rank_std: float = 200.0,
    plan_std: float = 0.08,
    heat_std: float = 0.1,
) -> dict:
    """Simple MC simulation for uncertainty interval around admission probability."""
    rg = np.random.normal(base_rank_gap, rank_std, n_samples)
    py = np.random.normal(plan_yoy_change, plan_std, n_samples)
    hy = np.random.normal(heat_yoy_change, heat_std, n_samples)

    z = -0.0025 * rg + 1.2 * py - 0.9 * hy
    p = 1 / (1 + np.exp(-z))

    return {
        "mean": float(np.mean(p)),
        "p5": float(np.quantile(p, 0.05)),
        "p95": float(np.quantile(p, 0.95)),
    }
