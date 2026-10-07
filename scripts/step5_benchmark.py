"""Step 5: benchmark (Objective 3) and the synthetic dial (R7).

- Baselines Cox-LASSO and Random Survival Forest.
- On synthetic sets: all kernels (classical + quantum A1-A4b, A6), hybrid form
  (the paper's Model 2), same splits and inner CV as Steps 1 and 4.
- --wrap adds approach A5b (quantum kernels with c in [1, 8], angles wrap around).

Rows for the (dataset, kernel) pairs being run replace earlier rows in
results/step5_splits.csv; everything else is kept.

    uv run python scripts/step5_benchmark.py                                  # everything
    uv run python scripts/step5_benchmark.py --datasets synth-periodic --wrap
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed, parallel_config

from qksm import baselines as B
from qksm import data as D
from qksm import evaluate as E
from qksm import kernels as Kn
from qksm import synthetic as S
from qksm.quantum import CIRCUITS

RESULTS = Path(__file__).resolve().parents[1] / "results"
ALL = ["vlc", "gbsg2"] + [f"synth-{k}" for k in S.RISKS]
WRAP_CIRCUITS = ["Q-1L", "Q-2L", "Q-ZZ"]


def load(name):
    return S.make(name.removeprefix("synth-")) if name.startswith("synth-") else D.load(name)


def kernels_for(name, ordinal, wrap):
    ks = []
    if name.startswith("synth-"):
        ks += ([cls(ordinal) for cls in Kn.CLASSICAL.values()]
               + [Kn.QuantumKernel(ordinal, c) for c in CIRCUITS]
               + [Kn.ProjectedQuantumKernel(ordinal, "Q-2L")])
    if wrap:
        ks += [Kn.QuantumKernel(ordinal, c, wrap=True) for c in WRAP_CIRCUITS]
        if name.startswith("synth-"):
            ks.append(Kn.RBFKernel(ordinal, narrow=True))   # fairness check for A5b
    return ks


def run_baselines(data, n_splits):
    splits = E.outer_splits(data, n_splits)
    jobs = [(name, i, tr, te) for name in ("cox-lasso", "rsf") for i, (tr, te) in enumerate(splits)]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=-1)(delayed(B.tune_and_test)(data, n, tr, te, seed=i) for n, i, tr, te in jobs)
    return pd.DataFrame([{"dataset": data.name, "kernel": n, "form": "baseline", "split": i, **r}
                         for (n, i, tr, te), r in zip(jobs, res)])


def main(datasets, n_splits, wrap, baselines, only=None):
    frames = []
    for name in datasets:
        data = load(name)
        t0 = time.time()
        if baselines:
            frames.append(run_baselines(data, n_splits))
        ks = kernels_for(name, data.ordinal, wrap)
        if only:
            ks = [k for k in ks if k.name in only]
        if ks:
            frames.append(E.run(data, ks, {"hybrid": 0.5}, n_splits=n_splits))
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)

    new = pd.concat(frames, ignore_index=True)
    out = RESULTS / "step5_splits.csv"
    if out.exists():
        old = pd.read_csv(out)
        key = set(zip(new.dataset, new.kernel))
        old = old[[(d, k) not in key for d, k in zip(old.dataset, old.kernel)]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out, index=False)

    table = E.summarize(new)["median"].unstack(["dataset"]).round(3)
    table = table[[c for c in ALL if c in table.columns]]
    table.to_csv(RESULTS / "step5_summary.csv")
    with open(RESULTS / "step5_summary.md", "w") as f:
        f.write(f"Median test C-index over {n_splits} splits (kernels: hybrid form; baselines: Cox-LASSO, RSF).\n\n")
        f.write(table.to_markdown())
    print(table.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=ALL)
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--wrap", action="store_true", help="add A5b wrap-around quantum kernels")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--only", nargs="*", help="run only these kernels, e.g. --only rbf-narrow")
    a = p.parse_args()
    main(a.datasets, a.splits, a.wrap, not a.no_baselines, a.only)
