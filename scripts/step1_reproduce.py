"""Step 1: reproduce Van Belle et al. (2011) classically on VLC and GBSG2.

Survival SVM in three forms (ranking / hybrid / regression) x kernels
(linear / RBF / clinical), 20 repeated 2/3-1/3 splits, inner 5-fold CV on C-index.
Also fits a linear Cox model as the paper's `phlinear` reference.

    uv run python scripts/step1_reproduce.py [--splits 20]
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sksurv.linear_model import CoxPHSurvivalAnalysis

from qksm import data as D
from qksm import evaluate as E
from qksm import kernels as Kn
from qksm.models import FORMS

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Paper's median C-index (Table 2 linear kernel / Table 3 clinical kernel).
PAPER = {
    ("vlc", "linear"): {"ranking": 0.62, "regression": 0.69, "hybrid": 0.69, "cox": 0.68},
    ("vlc", "clinical"): {"ranking": 0.57, "regression": 0.69, "hybrid": 0.69},
    ("gbsg2", "linear"): {"ranking": 0.62, "regression": 0.67, "hybrid": 0.67, "cox": 0.67},
    ("gbsg2", "clinical"): {"ranking": 0.62, "regression": 0.68, "hybrid": 0.68},
}


def cox_split(data, tr, te):
    prep = D.Standardizer(data.ordinal).fit(data.X.iloc[tr])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = CoxPHSurvivalAnalysis(alpha=1e-4).fit(prep.transform(data.X.iloc[tr]), data.y[tr])
    risk = m.predict(prep.transform(data.X.iloc[te]))
    t, e = data.time[te], data.event[te]
    return {"cindex": E.cindex(t, e, risk), "logrank": E.logrank_chi2(t, e, risk),
            "hazard_ratio": E.hazard_ratio(t, e, risk)}


def main(n_splits):
    RESULTS.mkdir(exist_ok=True)
    frames = []
    for name in ["vlc", "gbsg2"]:
        data = D.load(name)
        t0 = time.time()
        kernels = [cls(data.ordinal) for cls in Kn.CLASSICAL.values()]
        frames.append(E.run(data, kernels, FORMS, n_splits=n_splits, verbose=5))
        splits = E.outer_splits(data, n_splits)
        cox = Parallel(n_jobs=-1)(delayed(cox_split)(data, tr, te) for tr, te in splits)
        frames.append(pd.DataFrame([{"dataset": name, "kernel": "linear", "form": "cox", "split": i, **r}
                                    for i, r in enumerate(cox)]))
        print(f"{name}: done in {time.time() - t0:.0f}s")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "step1_splits.csv", index=False)

    summary = E.summarize(df).reset_index()
    for metric in ["logrank", "hazard_ratio"]:
        summary[metric] = E.summarize(df, metric)["text"].to_numpy()
    summary["paper"] = [PAPER.get((r.dataset, r.kernel), {}).get(r.form, np.nan) for r in summary.itertuples()]
    summary = summary.rename(columns={"text": "cindex"})[
        ["dataset", "kernel", "form", "cindex", "paper", "logrank", "hazard_ratio", "n"]]
    summary.to_csv(RESULTS / "step1_summary.csv", index=False)
    with open(RESULTS / "step1_summary.md", "w") as f:
        f.write(summary.to_markdown(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=int, default=20)
    main(p.parse_args().splits)
