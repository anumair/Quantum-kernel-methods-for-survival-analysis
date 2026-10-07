"""Non-kernel baselines (Objective 3): Cox-LASSO and Random Survival Forest.

Both use the same outer splits and the same inner 5-fold CV on C-index as the
kernel models, and the same standardized features as the linear/RBF kernels.
"""

import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis

from qksm.data import Standardizer
from qksm.evaluate import cindex, hazard_ratio, logrank_chi2


def _cox_lasso(alpha):
    return CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[alpha], fit_baseline_model=False, max_iter=100000)


def _rsf(min_leaf):
    return RandomSurvivalForest(n_estimators=300, min_samples_leaf=min_leaf, max_features="sqrt",
                                n_jobs=1, random_state=0)


BASELINES = {
    # name -> (model factory, grid); 7 values each, same budget as the kernels' alpha grid
    "cox-lasso": (_cox_lasso, list(np.geomspace(1e-3, 0.5, 7))),
    "rsf": (_rsf, [3, 5, 10, 15, 20, 30, 50]),
}


def tune_and_test(data, name, tr, te, n_folds=5, seed=0):
    factory, grid = BASELINES[name]
    prep = Standardizer(data.ordinal).fit(data.X.iloc[tr])
    A, B = prep.transform(data.X.iloc[tr]), prep.transform(data.X.iloc[te])
    t, e, y = data.time[tr], data.event[tr], data.y[tr]
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(A, e))

    def score(p):
        s = []
        for a, b in folds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = factory(p).fit(A[a], y[a])
            s.append(cindex(t[b], e[b], m.predict(A[b])))
        return np.mean(s)

    scores = [score(p) for p in grid]
    best = grid[int(np.argmax(scores))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        risk = factory(best).fit(A, y).predict(B)
    tt, ee = data.time[te], data.event[te]
    return {"cindex": cindex(tt, ee, risk), "logrank": logrank_chi2(tt, ee, risk),
            "hazard_ratio": hazard_ratio(tt, ee, risk), "cv_cindex": max(scores), "param": best}
