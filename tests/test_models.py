import warnings

import numpy as np
import pytest
from sksurv.svm import FastKernelSurvivalSVM

from qksm import data as D
from qksm import evaluate as E
from qksm import kernels as Kn
from qksm.models import FORMS, KernelSurvivalSVM, empirical_kernel_map


@pytest.fixture(scope="module")
def vlc_split():
    d = D.load("vlc")
    tr, te = E.outer_splits(d, 1)[0]
    K_tr, K_te = Kn.RBFKernel(d.ordinal).matrices(d.X.iloc[tr], d.X.iloc[te], 1.0)
    return d, tr, te, K_tr, K_te


def test_empirical_kernel_map_reconstructs_kernel(vlc_split):
    _, _, _, K_tr, _ = vlc_split
    phi = K_tr @ empirical_kernel_map(K_tr)
    assert np.allclose(phi @ phi.T, K_tr, atol=1e-8)


@pytest.mark.parametrize("rank_ratio", FORMS.values())
def test_matches_kernel_svm(vlc_split, rank_ratio):
    """Linear SVM on the empirical kernel map == scikit-survival's kernel SVM."""
    d, tr, te, K_tr, K_te = vlc_split
    ref = FastKernelSurvivalSVM(alpha=1.0, rank_ratio=rank_ratio, fit_intercept=rank_ratio < 1,
                                kernel="precomputed", max_iter=200, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref.fit(K_tr, d.y[tr])
    expected = ref.predict(K_te) * (1 if rank_ratio == 1 else -1)
    ours = KernelSurvivalSVM(1.0, rank_ratio).fit(K_tr, d.y[tr]).risk(K_te)
    assert np.corrcoef(expected, ours)[0, 1] > 0.9999


@pytest.mark.parametrize("rank_ratio", FORMS.values())
def test_risk_orientation(vlc_split, rank_ratio):
    """Higher risk must mean shorter survival in every form (sign fix for rank_ratio < 1)."""
    d, tr, te, K_tr, K_te = vlc_split
    risk = KernelSurvivalSVM(1.0, rank_ratio).fit(K_tr, d.y[tr]).risk(K_te)
    assert E.cindex(d.time[te], d.event[te], risk) > 0.55
