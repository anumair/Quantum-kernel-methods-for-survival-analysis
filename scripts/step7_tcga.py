"""Step 5 (dataset 2): TCGA-BRCA multi-omics benchmark.

Data: UCSC Xena TCGA hub, downloaded to data/raw/:
  https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2.gz
  https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz
  https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix
  https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/survival%2FBRCA_survival.txt

Per outer split, the 8 features (3 mRNA PCs, 3 CNV PCs, age, stage) are fitted on the
training part only. *All* kernel models and the 8-feature baselines get the same 8 features.
Reality checks: Cox-LASSO on all 4000 pre-filtered genes + clinical, and on clinical only.

    uv run python scripts/step7_tcga.py [--splits 20]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import step3_diagnostics as S3  # noqa: E402

from qksm import baselines as B  # noqa: E402
from qksm import evaluate as E  # noqa: E402
from qksm import kernels as Kn  # noqa: E402
from qksm import tcga  # noqa: E402
from qksm.data import SurvivalData  # noqa: E402
from qksm.quantum import CIRCUITS  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def kernels():
    return ([cls({}) for cls in Kn.CLASSICAL.values()]
            + [Kn.QuantumKernel({}, c) for c in CIRCUITS if c != "Q-3L"]
            + [Kn.ProjectedQuantumKernel({}, "Q-2L")])


def full_view(data, tr):
    med = data.X.iloc[tr][["age", "stage"]].median()
    X = data.X.copy()
    X[["age", "stage"]] = X[["age", "stage"]].fillna(med)
    return SurvivalData(data.name, X, data.time, data.event)


def benchmark(data, splits):
    jobs = []
    for i, (tr, te) in enumerate(splits):
        red = tcga.reduced_for_split(data, tr)
        jobs += [("kernel", k.name, red, k, tr, te, i) for k in kernels()]
        jobs += [("baseline", b, red, b, tr, te, i) for b in ("cox-lasso", "rsf")]
        jobs.append(("baseline", "cox-lasso-all-genes", full_view(data, tr), "cox-lasso-hd", tr, te, i))
        jobs.append(("baseline", "cox-lasso-clinical-only", tcga.clinical_for_split(data, tr), "cox-lasso", tr, te, i))

    def one(kind, name, d, model, tr, te, i):
        if kind == "kernel":
            return E.tune_and_test(d, model, 0.5, tr, te, seed=i)
        return B.tune_and_test(d, model, tr, te, seed=i)

    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=-1, verbose=5)(delayed(one)(*j) for j in jobs)
    return pd.DataFrame([{"dataset": data.name, "kernel": j[1], "form": "hybrid" if j[0] == "kernel" else "baseline",
                          "split": j[6], **r} for j, r in zip(jobs, res)])


def diagnostics(data, splits, n_perm=100):
    jobs = []
    for s, (tr, te) in enumerate(splits):
        red = tcga.reduced_for_split(data, tr)
        X_tr, X_te = red.X.iloc[tr], red.X.iloc[te]
        refs = {"linear": Kn.LinearKernel({}).matrices(X_tr, X_te)[0],
                "rbf": Kn.RBFKernel({}).matrices(X_tr, X_te, 1.0)[0]}
        jobs += [(red, k, p, s, tr, te, refs, n_perm if s == 0 else 0) for k, p in S3.configs({})]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=-1, verbose=5)(delayed(S3.one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    df["gap"] = df["train_cindex"] - df["test_cindex"]
    return df


def main(n_splits):
    data = tcga.load_brca()
    print(f"TCGA-BRCA: {len(data)} patients, {int(data.event.sum())} deaths", flush=True)
    splits = E.outer_splits(data, n_splits)

    t0 = time.time()
    df = benchmark(data, splits)
    df.to_csv(RESULTS / "step7_tcga_splits.csv", index=False)
    summ = E.summarize(df).reset_index()[["kernel", "form", "median", "iqr"]].sort_values("median", ascending=False)
    print(f"benchmark done in {time.time() - t0:.0f}s\n", summ.round(3).to_string(index=False), flush=True)

    t0 = time.time()
    dg = diagnostics(data, splits[:3])
    dg.to_csv(RESULTS / "step7_tcga_diagnostics.csv", index=False)
    cfg = dg.groupby(["kernel", "param"], dropna=False).mean(numeric_only=True).reset_index()
    rho, p = spearmanr(cfg["kta"], cfg["test_cindex"])
    rho_e, p_e = spearmanr(cfg["kta_event_only"], cfg["test_cindex"])
    print(f"diagnostics done in {time.time() - t0:.0f}s")
    lines = [f"TCGA-BRCA: Spearman(kta, test C) = {rho:.3f} (p = {p:.3g}); event-only {rho_e:.3f} (p = {p_e:.3g}); {len(cfg)} configs"]
    best = cfg.loc[cfg.groupby("kernel").test_cindex.idxmax(),
                   ["kernel", "param", "offdiag_mean", "eff_rank_frac", "kta", "align_rbf", "g_vs_rbf", "gap", "test_cindex"]]
    with open(RESULTS / "step7_tcga_summary.md", "w") as f:
        f.write(f"TCGA-BRCA ({len(data)} patients, {int(data.event.sum())} deaths), median test C-index over {n_splits} splits\n\n")
        f.write(summ.round(3).to_markdown(index=False) + "\n\n" + "\n".join(lines) + "\n\nBest config per kernel (3 splits):\n\n")
        f.write(best.round(3).to_markdown(index=False) + "\n")
    print("\n".join(lines)); print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=int, default=20)
    main(p.parse_args().splits)
