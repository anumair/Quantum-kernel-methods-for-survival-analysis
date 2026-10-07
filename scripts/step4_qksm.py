"""Step 4: Quantum Kernel Survival Model on VLC and GBSG2 (approaches A1-A4, A4b, A6).

Same protocol as Step 1 (20 splits, inner 5-fold CV), each circuit its own row,
tuning only c (7 values) and alpha (7 values). Writes results/step4_*.csv and a
comparison table next to the Step 1 classical results.

    uv run python scripts/step4_qksm.py [--splits 20]
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from qksm import data as D
from qksm import evaluate as E
from qksm import kernels as Kn
from qksm.models import FORMS
from qksm.quantum import CIRCUITS

RESULTS = Path(__file__).resolve().parents[1] / "results"


def quantum_kernels(ordinal):
    return [Kn.QuantumKernel(ordinal, c) for c in CIRCUITS] + [Kn.ProjectedQuantumKernel(ordinal, "Q-2L")]


def main(n_splits, only):
    frames = []
    for name in ["vlc", "gbsg2"]:
        data = D.load(name)
        kernels = quantum_kernels(data.ordinal)
        if only:
            kernels = [k for k in kernels if k.name in only]
        t0 = time.time()
        frames.append(E.run(data, kernels, FORMS, n_splits=n_splits, verbose=5))
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    q = pd.concat(frames, ignore_index=True)
    out = RESULTS / "step4_splits.csv"
    if only and out.exists():   # add/replace just these kernels' rows
        old = pd.read_csv(out)
        q = pd.concat([old[~old.kernel.isin(only)], q], ignore_index=True)
    q.to_csv(out, index=False)

    both = pd.concat([pd.read_csv(RESULTS / "step1_splits.csv"), q], ignore_index=True)
    table = E.summarize(both)["median"].unstack("form").round(3)
    table = table[["ranking", "hybrid", "regression", "cox"]]
    table.to_csv(RESULTS / "step4_comparison.csv")
    with open(RESULTS / "step4_comparison.md", "w") as f:
        f.write("Median test C-index over 20 splits (classical rows from Step 1).\n\n")
        f.write(table.to_markdown())
    print(table.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--only", nargs="*", help="run only these kernels, e.g. --only Q-Z")
    a = p.parse_args()
    main(a.splits, a.only)
