"""Kernels as plug-in parts.

Every kernel follows the same interface so the survival models never know
which kernel they get:

    spec.grid                      -> list of hyperparameter values to tune over
    spec.matrices(X_tr, X_te, p)   -> (K_tr, K_te) with shapes (n_tr, n_tr), (n_te, n_tr)

All fitted state (scalers, ranges, bandwidth heuristics) comes from X_tr only.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sksurv.kernels import ClinicalKernelTransform

from qksm.data import Standardizer

# Same grid size for every tunable kernel (fair tuning budget, see docs/PLAN.md).
GRID_SIZE = 7


class LinearKernel:
    """k(x, z) = x·z on standardized features."""

    name = "linear"

    def __init__(self, ordinal):
        self.ordinal = ordinal
        self.grid = [None]

    def matrices(self, X_tr, X_te, param=None):
        prep = Standardizer(self.ordinal).fit(X_tr)
        A, B = prep.transform(X_tr), prep.transform(X_te)
        return A @ A.T, B @ A.T


class RBFKernel:
    """k(x, z) = exp(-gamma ||x - z||²) on standardized features.

    gamma = multiplier × (1 / median squared distance), the median heuristic,
    with multipliers 2^-3 ... 2^3 (GRID_SIZE values).
    """

    name = "rbf"

    def __init__(self, ordinal):
        self.ordinal = ordinal
        self.grid = list(2.0 ** np.linspace(-3, 3, GRID_SIZE))

    def matrices(self, X_tr, X_te, multiplier):
        prep = Standardizer(self.ordinal).fit(X_tr)
        A, B = prep.transform(X_tr), prep.transform(X_te)
        gamma = multiplier / np.median(pdist(A, "sqeuclidean"))
        return np.exp(-gamma * cdist(A, A, "sqeuclidean")), np.exp(-gamma * cdist(B, A, "sqeuclidean"))


class ClinicalKernel:
    """Clinical kernel of Daemen & De Moor (paper Eq. 12-13), via scikit-survival.

    Continuous/ordinal: (range - |x - z|) / range, with range from the training data.
    Nominal: 1 if equal else 0. Averaged over features.
    """

    name = "clinical"

    def __init__(self, ordinal):
        self.ordinal = ordinal
        self.grid = [None]

    def matrices(self, X_tr, X_te, param=None):
        t = ClinicalKernelTransform(ordinal_categories=self.ordinal or None).fit(X_tr)
        return t.transform(X_tr), t.transform(X_te)


CLASSICAL = {"linear": LinearKernel, "rbf": RBFKernel, "clinical": ClinicalKernel}
