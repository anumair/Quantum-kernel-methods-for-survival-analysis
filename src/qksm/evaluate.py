"""Evaluation protocol of Van Belle et al. (2011), Section 4.

- Repeated random splits: 2/3 train, 1/3 test (stratified on the event indicator).
- Hyperparameters tuned by 5-fold CV on the training part only, scored by C-index.
- Test metrics: Harrell's C-index, logrank chi² (median split), hazard ratio.
"""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sksurv.metrics import concordance_index_censored

from qksm.models import ALPHA_GRID, KernelSurvivalSVM

# ---------------------------------------------------------------- metrics


def cindex(time, event, risk):
    """Harrell's C-index: fraction of comparable pairs ordered correctly by risk."""
    return concordance_index_censored(event.astype(bool), time, risk)[0]


def logrank_chi2(time, event, risk):
    """Logrank chi² between patients below vs above the median risk (paper §4.2)."""
    high = risk > np.median(risk)
    if high.all() or not high.any():
        return np.nan
    res = logrank_test(time[high], time[~high], event[high], event[~high])
    return res.test_statistic


def hazard_ratio(time, event, risk):
    """Hazard ratio of a univariate Cox model on the risk rescaled to [0, 1] (paper §4.2)."""
    span = risk.max() - risk.min()
    if span == 0:
        return np.nan
    df = pd.DataFrame({"r": (risk - risk.min()) / span, "T": time, "E": event.astype(int)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter().fit(df, "T", "E")
    return float(np.exp(cph.params_["r"]))


# ---------------------------------------------------------------- protocol


def outer_splits(data, n_splits=20, seed=0):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1 / 3, random_state=seed)
    return list(sss.split(data.X, data.event))


def tune_and_test(data, kernel, rank_ratio, tr, te, n_folds=5, seed=0):
    """Tune (kernel param, alpha) by inner CV on `tr`, refit on `tr`, score on `te`.

    The kernel matrix is computed once on the outer training part; inner folds
    reuse slices of it. Kernel fitting (scaling, ranges, bandwidth) therefore sees
    the whole training part but never the test part.
    """
    X_tr, X_te = data.X.iloc[tr], data.X.iloc[te]
    t_tr, e_tr = data.time[tr], data.event[tr]
    y_tr = data.y[tr]
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(tr, e_tr))

    best = (-np.inf, None, None, None)
    for param in kernel.grid:
        K_tr, K_te = kernel.matrices(X_tr, X_te, param)
        for alpha in ALPHA_GRID:
            scores = []
            for a, b in folds:
                m = KernelSurvivalSVM(alpha, rank_ratio).fit(K_tr[np.ix_(a, a)], y_tr[a])
                scores.append(cindex(t_tr[b], e_tr[b], m.risk(K_tr[np.ix_(b, a)])))
            if np.mean(scores) > best[0]:
                best = (np.mean(scores), param, alpha, (K_tr, K_te))

    cv_score, param, alpha, (K_tr, K_te) = best
    risk = KernelSurvivalSVM(alpha, rank_ratio).fit(K_tr, y_tr).risk(K_te)
    t_te, e_te = data.time[te], data.event[te]
    return {
        "cindex": cindex(t_te, e_te, risk),
        "logrank": logrank_chi2(t_te, e_te, risk),
        "hazard_ratio": hazard_ratio(t_te, e_te, risk),
        "cv_cindex": cv_score,
        "param": param,
        "alpha": alpha,
    }


def run(data, kernels, forms, n_splits=20, seed=0, n_jobs=-1, verbose=0):
    """All (kernel, form) combinations over the same outer splits. Returns a long DataFrame."""
    splits = outer_splits(data, n_splits, seed)
    jobs = [
        (k, form, rr, i, tr, te)
        for k in kernels
        for form, rr in forms.items()
        for i, (tr, te) in enumerate(splits)
    ]
    # One BLAS thread per worker: otherwise 10 workers x 10 threads thrash the CPU.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(tune_and_test)(data, k, rr, tr, te, seed=seed + i) for k, form, rr, i, tr, te in jobs
        )
    rows = [
        {"dataset": data.name, "kernel": k.name, "form": form, "split": i, **res}
        for (k, form, rr, i, tr, te), res in zip(jobs, results)
    ]
    return pd.DataFrame(rows)


def summarize(df, metric="cindex"):
    """Median ± IQR per (dataset, kernel, form), the paper's reporting style."""
    g = df.groupby(["dataset", "kernel", "form"])[metric]
    out = pd.DataFrame({
        "median": g.median(),
        "iqr": g.quantile(0.75) - g.quantile(0.25),
        "n": g.size(),
    })
    out["text"] = out.apply(lambda r: f"{r['median']:.3f} ± {r['iqr']:.3f}", axis=1)
    return out


def paired_wilcoxon(df, a, b, metric="cindex"):
    """Paired Wilcoxon signed-rank test between two models scored on the same splits.

    `a` and `b` are dicts selecting rows, e.g. {"kernel": "rbf", "form": "hybrid"}.
    Splits share training data, so p-values are indicative only (see docs/PLAN.md).
    """
    def pick(sel):
        m = np.ones(len(df), bool)
        for k, v in sel.items():
            m &= df[k].to_numpy() == v
        return df[m].sort_values("split")[metric].to_numpy()

    x, y = pick(a), pick(b)
    return wilcoxon(x, y).pvalue
