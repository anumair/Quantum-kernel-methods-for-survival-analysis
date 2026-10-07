"""Synthetic survival data with a known risk function: the R7 "dial".

    linear       eta = w . x                               (what linear Cox / linear SVM can fit)
    interaction  eta = x1*x2 + x3*x4                       (pairwise interactions; RBF should fit it)
    periodic     eta = sin(2 x1) cos(2 x2) + sin(x3 + x4)  (oscillating; suits cos/sin-based quantum kernels)

Times follow a Cox model with exponential baseline, T = -log(U) / (lambda0 * exp(eta)),
with independent uniform censoring tuned to about 40% censored.
"""

import numpy as np
import pandas as pd

from qksm.data import SurvivalData

RISKS = ("linear", "interaction", "periodic")
N_FEATURES = 8


def _eta(kind, X):
    x = X.T
    if kind == "linear":
        w = np.array([1.0, -0.8, 0.6, 0.5, -0.4, 0.3, 0.0, 0.0])
        eta = X @ w
    elif kind == "interaction":
        eta = x[0] * x[1] + x[2] * x[3]
    elif kind == "periodic":
        eta = np.sin(2 * x[0]) * np.cos(2 * x[1]) + np.sin(x[2] + x[3])
    else:
        raise ValueError(f"unknown risk {kind!r}")
    return 1.5 * (eta - eta.mean()) / eta.std()   # same signal strength for every kind


def make(kind, n=500, censor_frac=0.4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(n, N_FEATURES))
    eta = _eta(kind, X)
    T = -np.log(rng.uniform(size=n)) / (0.1 * np.exp(eta))
    # choose the uniform censoring range that gives ~censor_frac censored
    U = rng.uniform(size=n)
    lo, hi = 1e-3, 1e4
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        frac = np.mean(U * mid < T)          # censored fraction; falls as mid grows
        lo, hi = (mid, hi) if frac > censor_frac else (lo, mid)
    C = U * np.sqrt(lo * hi)
    time, event = np.minimum(T, C), T <= C
    cols = [f"x{i + 1}" for i in range(N_FEATURES)]
    return SurvivalData(f"synth-{kind}", pd.DataFrame(X, columns=cols), time, event)
