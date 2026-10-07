"""Kernel diagnostics computed *before* training (Objective 2) and the evidence
measurements used to explain the results (R1-R4 in docs/PLAN.md, Step 5b).

All functions take a training kernel matrix K (n x n) and, where needed, the
training survival labels. None of them fits a survival model.
"""

import numpy as np
from scipy.linalg import eigh, sqrtm
from scipy.stats import rankdata

# ---------------------------------------------------------------- helpers


def center(K):
    """H K H with H = I - 11^T/n (feature-space centering)."""
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def comparable_mask(time, event):
    """M[i, j] = True if the order of i and j is known (paper Eq. 16, symmetrized).

    Pair known if the earlier time is an observed event. Two censored patients,
    or an event after a censoring, cannot be ordered.
    """
    t, e = np.asarray(time), np.asarray(event, bool)
    earlier_i = (t[:, None] < t[None, :]) & e[:, None]
    M = earlier_i | earlier_i.T
    np.fill_diagonal(M, False)
    return M


# ---------------------------------------------------------------- health check (R1)


def health(K):
    """Is the kernel informative? Off-diagonal statistics and effective rank.

    offdiag_mean ~ 0 / std ~ 0  -> K ~ identity: every patient looks different
    offdiag_mean ~ 1 / std ~ 0  -> K ~ all ones: every patient looks the same
    eff_rank = (sum lam)^2 / sum lam^2   (participation ratio, 1 ... n)
    """
    off = K[~np.eye(len(K), dtype=bool)]
    lam = np.clip(np.linalg.eigvalsh(K), 0, None)
    return {
        "offdiag_mean": float(off.mean()),
        "offdiag_std": float(off.std()),
        "eff_rank": float(lam.sum() ** 2 / (lam ** 2).sum()),
        "eff_rank_frac": float(lam.sum() ** 2 / (lam ** 2).sum() / len(K)),
    }


def spectrum(K):
    """Eigenvalues normalized to sum 1, largest first (R2)."""
    lam = np.clip(np.linalg.eigvalsh(K)[::-1], 0, None)
    return lam / lam.sum()


# ---------------------------------------------------------------- survival KTA (Objective 2)


def survival_target(time):
    """T[i, j] = 1 - |r_i - r_j| / (n - 1): patients with similar survival ranks are 'similar'.

    For censored patients the rank is only a lower bound; survival_kta() therefore
    uses only comparable pairs, and kta_event_only() is the cross-check.
    """
    r = rankdata(time)
    return 1.0 - np.abs(r[:, None] - r[None, :]) / (len(r) - 1)


def survival_kta(K, time, event):
    """Centered kernel-target alignment restricted to comparable pairs (in [-1, 1])."""
    M = comparable_mask(time, event)
    Kc, Tc = center(K)[M], center(survival_target(time))[M]
    return float((Kc * Tc).sum() / np.sqrt((Kc ** 2).sum() * (Tc ** 2).sum()))


def kta_event_only(K, time, event):
    """Cross-check: KTA on patients with an observed event (exact ranks, all pairs comparable)."""
    idx = np.flatnonzero(np.asarray(event, bool))
    return survival_kta(K[np.ix_(idx, idx)], np.asarray(time)[idx], np.ones(len(idx), bool))


def kta_permutation_test(K, time, event, n_perm=200, seed=0):
    """Shuffle (time, event) pairs together; p = fraction of shuffles with KTA >= observed."""
    rng = np.random.default_rng(seed)
    obs = survival_kta(K, time, event)
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(len(time))
        null[b] = survival_kta(K, np.asarray(time)[p], np.asarray(event)[p])
    return obs, float((1 + (null >= obs).sum()) / (1 + n_perm)), float(null.mean()), float(null.std())


# ---------------------------------------------------------------- comparing kernels (R3, R4)


def kernel_alignment(K1, K2):
    """Centered kernel alignment between two kernels (1 = same geometry)."""
    A, B = center(K1), center(K2)
    return float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))


def geometric_difference(K_classical, K_quantum, reg=1e-3):
    """g(K_C || K_Q) of Huang et al. (2021), 'Power of data in quantum machine learning'.

    g = sqrt(|| sqrt(K_Q) (K_C + reg I)^-1 sqrt(K_Q) ||_spectral), both normalized to trace n.
    g ~ 1: the classical kernel can learn anything the quantum kernel can on this data,
    so no quantum advantage is possible. Large g (~ sqrt(n)): advantage is *possible* (not guaranteed).
    """
    n = len(K_classical)
    Kc = K_classical * n / np.trace(K_classical)
    Kq = K_quantum * n / np.trace(K_quantum)
    sq = np.real(sqrtm(Kq))
    M = sq @ np.linalg.solve(Kc + reg * np.eye(n), sq)
    return float(np.sqrt(eigh((M + M.T) / 2, eigvals_only=True).max()))
