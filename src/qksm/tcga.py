"""TCGA-BRCA multi-omics cohort (UCSC Xena TCGA hub) and fold-safe reduction to 8 features.

Raw files in data/raw/ (see scripts/step7_tcga.py for the download URLs):
  HiSeqV2.gz                                  mRNA, log2(RSEM + 1), genes x samples
  Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz   copy number (GISTIC2), genes x samples
  BRCA_clinicalMatrix                         clinical (age, pathologic stage)
  BRCA_survival.txt                           curated overall survival (OS, OS.time; TCGA-CDR)

Cohort: primary tumours (-01) present in all four files with OS available.

Leakage control: the only step done on the whole cohort is a *label-free* pre-filter
(top-N variance genes per omic, to keep the data small). Everything that is fitted
-- gene selection, standardization, PCA, imputation -- happens in OmicsReducer.fit()
on the training part of each split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from qksm.data import SurvivalData

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

STAGE = {"Stage I": 1, "Stage IA": 1, "Stage IB": 1,
         "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2,
         "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
         "Stage IV": 4}           # Stage X / missing / [Discrepancy] -> NaN, imputed per fold


def _omic(path, prefix, keep, n_top):
    M = pd.read_csv(path, sep="\t", index_col=0)
    M = M.loc[:, M.columns.isin(keep)].T.astype(np.float32)        # patients x genes
    M = M.loc[:, M.notna().all()]
    M = M[M.var().nlargest(n_top).index]                              # label-free pre-filter
    M.columns = [f"{prefix}:{g}" for g in M.columns]
    return M


def load_brca(n_top=2000, cache=True):
    """SurvivalData with columns age, stage (NaN allowed), mrna:<gene>..., cnv:<gene>..."""
    path = PROCESSED / f"brca_top{n_top}.pkl"
    if cache and path.exists():
        return pd.read_pickle(path)
    surv = pd.read_csv(RAW / "BRCA_survival.txt", sep="\t", index_col=0).dropna(subset=["OS", "OS.time"])
    surv = surv[surv["OS.time"] > 0]
    clin = pd.read_csv(RAW / "BRCA_clinicalMatrix", sep="\t", index_col=0)
    mrna_cols = pd.read_csv(RAW / "HiSeqV2.gz", sep="\t", index_col=0, nrows=0).columns
    cnv_cols = pd.read_csv(RAW / "Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz", sep="\t", index_col=0, nrows=0).columns
    keep = sorted({s for s in mrna_cols if s.endswith("-01")} & set(cnv_cols) & set(surv.index) & set(clin.index))

    mrna = _omic(RAW / "HiSeqV2.gz", "mrna", keep, n_top)
    cnv = _omic(RAW / "Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz", "cnv", keep, n_top)
    clinical = pd.DataFrame({
        "age": pd.to_numeric(clin.loc[keep, "age_at_initial_pathologic_diagnosis"], errors="coerce"),
        "stage": clin.loc[keep, "pathologic_stage"].map(STAGE).astype(float),
    })
    X = pd.concat([clinical, mrna.loc[keep], cnv.loc[keep]], axis=1).reset_index(drop=True)
    data = SurvivalData("tcga-brca", X, surv.loc[keep, "OS.time"].to_numpy(float),
                        surv.loc[keep, "OS"].to_numpy().astype(bool))
    data.patients = keep
    PROCESSED.mkdir(exist_ok=True)
    pd.to_pickle(data, path)
    return data


class OmicsReducer:
    """Fit on training patients: per omic, top-variance genes -> z-score -> PCA.

    Output columns: mrna_pc1..k, cnv_pc1..k, age, stage  (k = n_pcs; 3+3+2 = 8 features).
    Missing age/stage are imputed with the training median.
    """

    def __init__(self, n_pcs=3, n_genes=1000):
        self.n_pcs, self.n_genes = n_pcs, n_genes

    def fit(self, X):
        self.parts_ = {}
        for omic in ("mrna", "cnv"):
            cols = [c for c in X.columns if c.startswith(omic + ":")]
            top = X[cols].var().nlargest(self.n_genes).index
            mu, sd = X[top].mean(), X[top].std() + 1e-8
            pca = PCA(self.n_pcs, random_state=0).fit((X[top] - mu) / sd)
            self.parts_[omic] = (top, mu, sd, pca)
        self.med_ = X[["age", "stage"]].median()
        return self

    def transform(self, X):
        out = {}
        for omic, (top, mu, sd, pca) in self.parts_.items():
            Z = pca.transform((X[top] - mu) / sd)
            for k in range(self.n_pcs):
                out[f"{omic}_pc{k + 1}"] = Z[:, k]
        out["age"] = X["age"].fillna(self.med_["age"]).to_numpy()
        out["stage"] = X["stage"].fillna(self.med_["stage"]).to_numpy()
        return pd.DataFrame(out, index=X.index)


def reduced_for_split(data, tr, **kw):
    """8-feature SurvivalData for one split: reducer fitted on `tr` only, applied to everyone."""
    red = OmicsReducer(**kw).fit(data.X.iloc[tr])
    return SurvivalData(data.name, red.transform(data.X), data.time, data.event)


def clinical_for_split(data, tr):
    """Clinical-only (age, stage) view with training-median imputation."""
    med = data.X.iloc[tr][["age", "stage"]].median()
    return SurvivalData(data.name, data.X[["age", "stage"]].fillna(med), data.time, data.event)
