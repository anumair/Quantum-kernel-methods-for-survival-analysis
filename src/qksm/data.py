"""Dataset loaders and fold-safe preprocessing.

Every loader returns a :class:`SurvivalData` with the raw feature DataFrame
(categoricals kept as pandas categoricals, so the clinical kernel can treat
them correctly) plus survival time and event indicator.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.datasets import load_gbsg2, load_veterans_lung_cancer
from sksurv.util import Surv


@dataclass
class SurvivalData:
    name: str
    X: pd.DataFrame          # raw features (categoricals as pandas Categorical)
    time: np.ndarray         # observed time (event or censoring)
    event: np.ndarray        # True = event observed, False = right-censored
    ordinal: dict = field(default_factory=dict)  # column -> ordered category labels

    @property
    def y(self):
        """Structured array in the format scikit-survival expects."""
        return Surv.from_arrays(event=self.event, time=self.time)

    def __len__(self):
        return len(self.time)


def load(name):
    """Load a dataset by name: 'vlc' or 'gbsg2'."""
    if name == "vlc":
        X, y = load_veterans_lung_cancer()
        return SurvivalData("vlc", X, y["Survival_in_days"], y["Status"])
    if name == "gbsg2":
        X, y = load_gbsg2()
        return SurvivalData("gbsg2", X, y["time"], y["cens"],
                            ordinal={"tgrade": ["I", "II", "III"]})
    raise ValueError(f"unknown dataset {name!r}")


def to_numeric(X, ordinal):
    """Encode features as numbers: ordinal -> 0..k-1, nominal -> one-hot (drop first)."""
    X = X.copy()
    for col, levels in ordinal.items():
        X[col] = pd.Categorical(X[col], categories=levels, ordered=True).codes.astype(float)
    nominal = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    return pd.get_dummies(X, columns=nominal, drop_first=True, dtype=float)


class Standardizer:
    """Numeric encoding + z-scoring, fitted on the training part only."""

    def __init__(self, ordinal):
        self.ordinal = ordinal

    def fit(self, X_train):
        self.columns_ = to_numeric(X_train, self.ordinal).columns
        self.scaler_ = StandardScaler().fit(self._encode(X_train))
        return self

    def transform(self, X):
        return self.scaler_.transform(self._encode(X))

    def _encode(self, X):
        # reindex so a category missing from one split still gets its column
        return to_numeric(X, self.ordinal).reindex(columns=self.columns_, fill_value=0.0).to_numpy()
