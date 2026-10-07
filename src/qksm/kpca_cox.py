"""KPCA-Cox: the Kernel Cox version of the QKSM (Objective 1).

Kernel PCA on the training kernel -> keep the top m components -> ridge-penalized
Cox model on them. It approximates Kernel Cox (Li & Luan 2003): a Cox model whose
log-risk lives in the kernel's feature space, restricted to its m leading directions.

Same outer splits and inner 5-fold CV as the SVM; the grid is kernel param x m,
with m taking 7 values like the SVM's alpha, so the tuning budget is the same.
"""

import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis

from qksm.evaluate import cindex, hazard_ratio, logrank_chi2

M_GRID = [2, 3, 5, 8, 12, 20, 30]
RIDGE = 0.1


def kpca_project(K_tr, K_te, m):
    """Centered kernel PCA: training scores and test projections on the top m components."""
    n = len(K_tr)
    one = np.ones((n, n)) / n
    Kc = K_tr - one @ K_tr - K_tr @ one + one @ K_tr @ one
    one_te = np.ones((len(K_te), n)) / n
    Kc_te = K_te - one_te @ K_tr - K_te @ one + one_te @ K_tr @ one
    lam, V = np.linalg.eigh(Kc)
    lam, V = lam[::-1], V[:, ::-1]
    m = int(min(m, (lam > 1e-10 * lam[0]).sum()))
    lam, V = lam[:m], V[:, :m]
    return V * np.sqrt(lam), Kc_te @ V / np.sqrt(lam)


class KPCACox:
    def __init__(self, m):
        self.m = m

    def fit(self, K_tr, y):
        self.K_tr_ = K_tr
        Z, _ = kpca_project(K_tr, K_tr[:1], self.m)
        self.mu_, self.sd_ = Z.mean(0), Z.std(0) + 1e-12
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cox_ = CoxPHSurvivalAnalysis(alpha=RIDGE).fit((Z - self.mu_) / self.sd_, y)
        return self

    def risk(self, K_te):
        _, Z_te = kpca_project(self.K_tr_, K_te, self.m)
        return self.cox_.predict((Z_te - self.mu_) / self.sd_)


def tune_and_test(data, kernel, tr, te, n_folds=5, seed=0):
    X_tr, X_te = data.X.iloc[tr], data.X.iloc[te]
    t, e, y = data.time[tr], data.event[tr], data.y[tr]
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(tr, e))
    best = (-np.inf, None, None, None)
    for param in kernel.grid:
        K_tr, K_te = kernel.matrices(X_tr, X_te, param)
        for m in M_GRID:
            s = np.mean([cindex(t[b], e[b], KPCACox(m).fit(K_tr[np.ix_(a, a)], y[a]).risk(K_tr[np.ix_(b, a)]))
                         for a, b in folds])
            if s > best[0]:
                best = (s, param, m, (K_tr, K_te))
    cv, param, m, (K_tr, K_te) = best
    risk = KPCACox(m).fit(K_tr, y).risk(K_te)
    tt, ee = data.time[te], data.event[te]
    return {"cindex": cindex(tt, ee, risk), "logrank": logrank_chi2(tt, ee, risk),
            "hazard_ratio": hazard_ratio(tt, ee, risk), "cv_cindex": cv, "param": param, "alpha": m}
