"""Remaining approaches and evidence:

  a7    A7 hybrid kernel w*K_Q + (1-w)*K_linear (hybrid SVM), c chosen by KTA
  kpca  KPCA-Cox (Kernel Cox version of the QKSM), classical vs quantum kernels
  lc    R6 learning curves: test C-index vs training-set fraction

    uv run python scripts/step6_extra.py a7 kpca lc
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from qksm import data as D
from qksm import evaluate as E
from qksm import kernels as Kn
from qksm import kpca_cox as KC
from qksm import synthetic as S

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(name):
    return S.make(name.removeprefix("synth-")) if name.startswith("synth-") else D.load(name)


def parallel(fn, jobs):
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=-1, verbose=2)(delayed(fn)(*j) for j in jobs)


def run_a7(n_splits):
    frames = []
    for name in ["vlc", "gbsg2", "synth-interaction", "synth-periodic"]:
        data = load(name)
        ks = [Kn.HybridKernel(data, "Q-2L"), Kn.HybridKernel(data, "Q-ZZ")]
        frames.append(E.run(data, ks, {"hybrid": 0.5}, n_splits=n_splits))
        print(name, "a7 done", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "step6_a7_splits.csv", index=False)
    return df


def run_kpca(n_splits):
    rows = []
    for name in ["vlc", "gbsg2", "synth-interaction"]:
        data = load(name)
        o = data.ordinal
        ks = [Kn.LinearKernel(o), Kn.RBFKernel(o), Kn.ClinicalKernel(o),
              Kn.QuantumKernel(o, "Q-1L"), Kn.QuantumKernel(o, "Q-2L"), Kn.QuantumKernel(o, "Q-ZZ")]
        splits = E.outer_splits(data, n_splits)
        jobs = [(data, k, tr, te, i) for k in ks for i, (tr, te) in enumerate(splits)]
        res = parallel(lambda d, k, tr, te, i: KC.tune_and_test(d, k, tr, te, seed=i), jobs)
        rows += [{"dataset": name, "kernel": k.name, "form": "kpca-cox", "split": i, **r}
                 for (d, k, tr, te, i), r in zip(jobs, res)]
        print(name, "kpca done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "step6_kpca_splits.csv", index=False)
    return df


def run_lc(n_splits):
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for name in ["gbsg2", "synth-interaction"]:
        data = load(name)
        o = data.ordinal
        ks = [Kn.LinearKernel(o), Kn.RBFKernel(o), Kn.QuantumKernel(o, "Q-1L"),
              Kn.QuantumKernel(o, "Q-2L"), Kn.QuantumKernel(o, "Q-ZZ")]
        jobs = []
        for i, (tr, te) in enumerate(E.outer_splits(data, n_splits)):
            perm = np.random.default_rng(i).permutation(tr)
            for f in fracs:
                sub = np.sort(perm[: int(round(f * len(tr)))])
                jobs += [(data, k, sub, te, i, f) for k in ks]
        res = parallel(lambda d, k, sub, te, i, f: E.tune_and_test(d, k, 0.5, sub, te, seed=i), jobs)
        rows += [{"dataset": name, "kernel": k.name, "frac": f, "n_train": len(sub), "split": i, **r}
                 for (d, k, sub, te, i, f), r in zip(jobs, res)]
        print(name, "lc done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "step6_learning_curves.csv", index=False)
    return df


def main(parts, n_splits):
    out = []
    if "a7" in parts:
        df = run_a7(n_splits)
        t = df.groupby(["dataset", "kernel"]).agg(cindex=("cindex", "median"), w_median=("param", "median"),
                                                   w_mean=("param", "mean"))
        out.append("## A7 hybrid kernel (median C-index, chosen w)\n\n" + t.round(3).to_markdown())
    if "kpca" in parts:
        df = run_kpca(n_splits)
        t = df.groupby(["dataset", "kernel"]).cindex.median().unstack("dataset")
        out.append("## KPCA-Cox (median C-index)\n\n" + t.round(3).to_markdown())
    if "lc" in parts:
        df = run_lc(max(10, n_splits // 2))
        t = df.groupby(["dataset", "kernel", "frac"]).cindex.median().unstack("frac")
        out.append("## R6 learning curves (median C-index by training fraction)\n\n" + t.round(3).to_markdown())
    text = "\n\n".join(out)
    (RESULTS / f"step6_{'_'.join(parts)}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("parts", nargs="+", choices=["a7", "kpca", "lc"])
    p.add_argument("--splits", type=int, default=20)
    a = p.parse_args()
    t0 = time.time()
    main(a.parts, a.splits)
    print(f"total {time.time() - t0:.0f}s")
