"""Step 5: benchmark (Objective 3) and the synthetic dial (R7).

- Baselines Cox-LASSO and Random Survival Forest on VLC, GBSG2 and the 3 synthetic sets.
- All kernels (classical + quantum A1-A4, A6) on the 3 synthetic sets, hybrid form
  (the paper's Model 2), same splits and inner CV as Steps 1 and 4.

    uv run python scripts/step5_benchmark.py [--splits 20]
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


def all_kernels(ordinal):
    return ([cls(ordinal) for cls in Kn.CLASSICAL.values()]
            + [Kn.QuantumKernel(ordinal, c) for c in CIRCUITS]
            + [Kn.ProjectedQuantumKernel(ordinal, "Q-2L")])


def run_baselines(data, n_splits):
    splits = E.outer_splits(data, n_splits)
    jobs = [(name, i, tr, te) for name in B.BASELINES for i, (tr, te) in enumerate(splits)]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=-1)(delayed(B.tune_and_test)(data, n, tr, te, seed=i) for n, i, tr, te in jobs)
    return pd.DataFrame([{"dataset": data.name, "kernel": n, "form": "baseline", "split": i, **r}
                         for (n, i, tr, te), r in zip(jobs, res)])


def main(n_splits):
    frames = []
    synth = [S.make(k) for k in S.RISKS]
    for data in [D.load("vlc"), D.load("gbsg2")] + synth:
        t0 = time.time()
        frames.append(run_baselines(data, n_splits))
        print(f"{data.name}: baselines done in {time.time() - t0:.0f}s", flush=True)
    for data in synth:
        t0 = time.time()
        frames.append(E.run(data, all_kernels(data.ordinal), {"hybrid": 0.5}, n_splits=n_splits))
        print(f"{data.name}: kernels done in {time.time() - t0:.0f}s", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "step5_splits.csv", index=False)
    table = E.summarize(df)["median"].unstack(["dataset"]).round(3)
    table.to_csv(RESULTS / "step5_summary.csv")
    with open(RESULTS / "step5_summary.md", "w") as f:
        f.write(f"Median test C-index over {n_splits} splits (kernels: hybrid form).\n\n")
        f.write(table.to_markdown())
    print(table.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=int, default=20)
    main(p.parse_args().splits)
