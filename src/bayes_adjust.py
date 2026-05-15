from __future__ import annotations

import numpy as np


def bayes_adjust_probability(p_raw: np.ndarray, n: np.ndarray, alpha0: float, beta0: float) -> np.ndarray:
    """Posterior mean for Beta-Binomial style shrinkage over model probability."""
    p_raw = np.asarray(p_raw, dtype=float)
    n = np.asarray(n, dtype=float)
    return (alpha0 + n * p_raw) / (alpha0 + beta0 + n)
