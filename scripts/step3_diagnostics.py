"""Step 3 / 5b: pre-training diagnostics for every kernel configuration.

For each dataset, each kernel x hyperparameter (c or bandwidth, NOT tuned here), and
each of the first 5 outer splits, record on the training part:
  - health check (R1): off-diagonal mean/std, effective rank
  - survival KTA (Objective 2), its event-only cross-check, permutation p-value (split 0)
  - alignment with the linear and RBF kernels (R4), geometric difference vs them (R3)
and after training (hybrid Survival SVM, alpha tuned by inner CV with the kernel fixed):
  - test C-index, and train C-index for the overfitting gap (R2)

    uv run python scripts/step3_diagnostics.py [--splits 5]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from qksm import data as D
from qksm import diagnostics as G
from qksm import evaluate as E
from qksm import kernels as Kn
from qksm.models import ALPHA_GRID, KernelSurvivalSVM
from qksm.quantum import CIRCUITS

RESULTS = Path(__file__).resolve().parents[1] / "results"
RANK_RATIO = 0.5  # hybrid form, as in the paper's Model 2


def configs(ordinal):
    out = [(Kn.LinearKernel(ordinal), None), (Kn.ClinicalKernel(ordinal), None)]
    rbf = Kn.RBFKernel(ordinal)
    out += [(rbf, p) for p in rbf.grid]
    for circ in CIRCUITS:
        k = Kn.QuantumKernel(ordinal, circ)
        out += [(k, c) for c in k.grid]
    pk = Kn.ProjectedQuantumKernel(ordinal, "Q-2L")
    out += [(pk, c) for c in pk.grid]
    return out


def fit_alpha(K_tr, y, t, e, seed):
    """Tune alpha only (kernel fixed) by inner 5-fold CV on C-index."""
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(K_tr, e))
    best, best_a = -np.inf, None
    for a in ALPHA_GRID:
        s = np.mean([E.cindex(t[v], e[v], KernelSurvivalSVM(a, RANK_RATIO).fit(K_tr[np.ix_(f, f)], y[f])
                             .risk(K_tr[np.ix_(v, f)])) for f, v in folds])
        if s > best:
            best, best_a = s, a
    return best_a


def one(data, kernel, param, split, tr, te, refs, n_perm):
    X_tr, X_te = data.X.iloc[tr], data.X.iloc[te]
    t, e, y = data.time[tr], data.event[tr], data.y[tr]
    K_tr, K_te = kernel.matrices(X_tr, X_te, param)

    row = {"dataset": data.name, "kernel": kernel.name, "param": param, "split": split, **G.health(K_tr)}
    row["kta"] = G.survival_kta(K_tr, t, e)
    row["kta_event_only"] = G.kta_event_only(K_tr, t, e)
    if n_perm:
        _, row["kta_perm_p"], row["kta_null_mean"], row["kta_null_std"] = G.kta_permutation_test(K_tr, t, e, n_perm)
    for ref_name, K_ref in refs.items():
        row[f"align_{ref_name}"] = G.kernel_alignment(K_tr, K_ref)
        row[f"g_vs_{ref_name}"] = G.geometric_difference(K_ref, K_tr)

    alpha = fit_alpha(K_tr, y, t, e, seed=split)
    m = KernelSurvivalSVM(alpha, RANK_RATIO).fit(K_tr, y)
    row["alpha"] = alpha
    row["test_cindex"] = E.cindex(data.time[te], data.event[te], m.risk(K_te))
    row["train_cindex"] = E.cindex(t, e, m.risk(K_tr))
    return row


def run_dataset(name, n_splits, n_perm):
    data = D.load(name)
    splits = E.outer_splits(data, n_splits)
    jobs = []
    for s, (tr, te) in enumerate(splits):
        X_tr, X_te = data.X.iloc[tr], data.X.iloc[te]
        refs = {"linear": Kn.LinearKernel(data.ordinal).matrices(X_tr, X_te)[0],
                "rbf": Kn.RBFKernel(data.ordinal).matrices(X_tr, X_te, 1.0)[0]}
        for k, p in configs(data.ordinal):
            jobs.append((data, k, p, s, tr, te, refs, n_perm if s == 0 else 0))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=-1, verbose=5)(delayed(one)(*j) for j in jobs)
    return pd.DataFrame(rows)


def main(n_splits, n_perm):
    df = pd.concat([run_dataset(n, n_splits, n_perm) for n in ["vlc", "gbsg2"]], ignore_index=True)
    df["gap"] = df["train_cindex"] - df["test_cindex"]
    df.to_csv(RESULTS / "step3_diagnostics.csv", index=False)

    # Does pre-training KTA predict post-training test C-index? (mean over splits per config)
    cfg = df.groupby(["dataset", "kernel", "param"], dropna=False).mean(numeric_only=True).reset_index()
    cfg.to_csv(RESULTS / "step3_configs.csv", index=False)
    lines = []
    for name, g in cfg.groupby("dataset"):
        for col in ["kta", "kta_event_only"]:
            rho, p = spearmanr(g[col], g["test_cindex"])
            lines.append(f"{name}: Spearman({col}, test C-index) = {rho:.3f} (p = {p:.3g}, {len(g)} configs)")
    print("\n".join(lines))
    with open(RESULTS / "step3_kta_vs_cindex.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--perm", type=int, default=200)
    a = p.parse_args()
    main(a.splits, a.perm)
