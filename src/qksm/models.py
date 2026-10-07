"""Survival models that take a precomputed kernel matrix."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sksurv.svm import FastSurvivalSVM

# The three forms of Van Belle et al. (2011), via scikit-survival's rank_ratio.
FORMS = {"ranking": 1.0, "hybrid": 0.5, "regression": 0.0}

# 7 values 2^-14 ... 2^4, shared by every kernel. A scan on VLC/GBSG2 put the
# CV optimum anywhere from 2^-12 (linear) to 2^4 (clinical regression).
ALPHA_GRID = list(2.0 ** np.arange(-14, 5, 3))


def empirical_kernel_map(K_tr, tol=1e-10):
    """Projection P such that Phi = K @ P satisfies Phi_tr @ Phi_tr.T == K_tr.

    With K_tr = V diag(lam) V^T, Phi_tr = V sqrt(lam) and P = V / sqrt(lam).
    Eigenvalues below tol * max are dropped (numerical zero / tiny negatives).
    """
    lam, V = np.linalg.eigh(K_tr)
    keep = lam > tol * lam.max()
    return V[:, keep] / np.sqrt(lam[keep])


class KernelSurvivalSVM:
    """Survival SVM on a precomputed kernel, always returning a *risk score*.

    Trained as scikit-survival's linear FastSurvivalSVM on the empirical kernel
    map Phi (Phi Phi^T = K). By the representer theorem this gives the same
    solution as the kernel SVM (verified in tests/test_models.py), but is
    20-60x faster than FastKernelSurvivalSVM(kernel="precomputed").

    rank_ratio = 1 -> ranking objective; predictions are already risk scores.
    rank_ratio < 1 -> regression on log(time); predictions are times, so we negate
                      them (higher = riskier) and fit an intercept (paper Eq. 14's b).
    """

    def __init__(self, alpha, rank_ratio, max_iter=200, random_state=0):
        self.rank_ratio = rank_ratio
        self.model = FastSurvivalSVM(
            alpha=alpha,
            rank_ratio=rank_ratio,
            fit_intercept=rank_ratio < 1.0,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, K_tr, y):
        self.P_ = empirical_kernel_map(K_tr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(K_tr @ self.P_, y)
        return self

    def risk(self, K_te):
        pred = self.model.predict(K_te @ self.P_)
        return pred if self.rank_ratio == 1.0 else -pred
