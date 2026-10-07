"""Kernels as plug-in parts.

Every kernel follows the same interface so the survival models never know
which kernel they get:

    spec.grid                      -> list of hyperparameter values to tune over
    spec.matrices(X_tr, X_te, p)   -> (K_tr, K_te) with shapes (n_tr, n_tr), (n_te, n_tr)

All fitted state (scalers, ranges, bandwidth heuristics) comes from X_tr only.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sksurv.kernels import ClinicalKernelTransform

from qksm.data import Standardizer

# Same grid size for every tunable kernel (fair tuning budget, see docs/PLAN.md).
GRID_SIZE = 7


class LinearKernel:
    """k(x, z) = x·z on standardized features."""

    name = "linear"

    def __init__(self, ordinal):
        self.ordinal = ordinal
        self.grid = [None]

    def matrices(self, X_tr, X_te, param=None):
        prep = Standardizer(self.ordinal).fit(X_tr)
        A, B = prep.transform(X_tr), prep.transform(X_te)
        return A @ A.T, B @ A.T


class RBFKernel:
    """k(x, z) = exp(-gamma ||x - z||²) on standardized features.

    gamma = multiplier × (1 / median squared distance), the median heuristic,
    with multipliers 2^-3 ... 2^3 (GRID_SIZE values).
    """

    name = "rbf"

    def __init__(self, ordinal, narrow=False):
        """narrow=True: multipliers 2^1 ... 2^7 (much narrower bandwidths), a fairness check
        against A5b on high-frequency data. Labelled 'rbf-narrow', same grid size."""
        self.ordinal = ordinal
        self.grid = list(2.0 ** np.linspace(1, 7, GRID_SIZE) if narrow else 2.0 ** np.linspace(-3, 3, GRID_SIZE))
        self.name = "rbf-narrow" if narrow else "rbf"

    def matrices(self, X_tr, X_te, multiplier):
        prep = Standardizer(self.ordinal).fit(X_tr)
        A, B = prep.transform(X_tr), prep.transform(X_te)
        gamma = multiplier / np.median(pdist(A, "sqeuclidean"))
        return np.exp(-gamma * cdist(A, A, "sqeuclidean")), np.exp(-gamma * cdist(B, A, "sqeuclidean"))


class ClinicalKernel:
    """Clinical kernel of Daemen & De Moor (paper Eq. 12-13), via scikit-survival.

    Continuous/ordinal: (range - |x - z|) / range, with range from the training data.
    Nominal: 1 if equal else 0. Averaged over features.
    """

    name = "clinical"

    def __init__(self, ordinal):
        self.ordinal = ordinal
        self.grid = [None]

    def matrices(self, X_tr, X_te, param=None):
        t = ClinicalKernelTransform(ordinal_categories=self.ordinal or None).fit(X_tr)
        return t.transform(X_tr), t.transform(X_te)


CLASSICAL = {"linear": LinearKernel, "rbf": RBFKernel, "clinical": ClinicalKernel}


# ---------------------------------------------------------------- quantum

# Angle scale c: 7 log-spaced values in [0.1, 1], same grid size as the RBF bandwidth.
C_GRID = list(np.geomspace(0.1, 1.0, GRID_SIZE))
# A5b only: large angle scale, angles wrap around (labelled separately).
C_GRID_WRAP = list(np.geomspace(1.0, 8.0, GRID_SIZE))


class QuantumKernel:
    """Fidelity kernel |<psi(x)|psi(x')>|^2 for one fixed circuit (A1-A4). Tunes only c."""

    def __init__(self, ordinal, circuit, wrap=False):
        """wrap=True is approach A5b: c in [1, 8], so angles wrap around [0, pi] and the
        kernel oscillates. Separate labelled rows ('<circuit>-wrap'), same grid size."""
        from qksm.quantum import CIRCUITS
        assert circuit in CIRCUITS
        self.ordinal = ordinal
        self.circuit = circuit
        self.name = f"{circuit}-wrap" if wrap else circuit
        self.grid = C_GRID_WRAP if wrap else C_GRID

    def states(self, X_tr, X_te, c):
        from qksm.quantum import AngleScaler, statevectors
        scaler = AngleScaler(self.ordinal).fit(X_tr)
        return (statevectors(self.circuit, c * scaler.transform(X_tr)),
                statevectors(self.circuit, c * scaler.transform(X_te)))

    def matrices(self, X_tr, X_te, c):
        from qksm.quantum import fidelity_kernel
        S_tr, S_te = self.states(X_tr, X_te, c)
        return fidelity_kernel(S_tr, S_tr), fidelity_kernel(S_te, S_tr)


class HybridKernel:
    """A7: K = w * K_Q + (1 - w) * K_linear, both scaled to unit mean diagonal.

    Tests whether the quantum kernel adds information *complementary* to the linear one.
    Only w is tuned (7 values; w = 0 is pure linear, w = 1 pure quantum). The quantum
    angle scale c is chosen *before training* as the C_GRID value with the highest
    survival KTA on the training part (Objective 2 used in practice). Labels for that
    are looked up from `data` by the training rows' index, so no test labels are used.
    """

    def __init__(self, data, circuit="Q-2L"):
        self.data = data
        self.quantum = QuantumKernel(data.ordinal, circuit)
        self.linear = LinearKernel(data.ordinal)
        self.name = f"H-{circuit}"
        self.grid = list(np.linspace(0.0, 1.0, GRID_SIZE))
        self._cache = {}

    def _parts(self, X_tr, X_te):
        key = (tuple(X_tr.index), tuple(X_te.index))
        if key not in self._cache:
            from qksm.diagnostics import survival_kta
            pos = self.data.X.index.get_indexer(X_tr.index)
            t, e = self.data.time[pos], self.data.event[pos]
            best = max(self.quantum.grid,
                       key=lambda c: survival_kta(self.quantum.matrices(X_tr, X_tr.iloc[:1], c)[0], t, e))
            Q_tr, Q_te = self.quantum.matrices(X_tr, X_te, best)
            L_tr, L_te = self.linear.matrices(X_tr, X_te)
            s = np.mean(np.diag(L_tr))
            self._cache = {key: (Q_tr, Q_te, L_tr / s, L_te / s, best)}
        return self._cache[key]

    def matrices(self, X_tr, X_te, w):
        Q_tr, Q_te, L_tr, L_te, self.c_ = self._parts(X_tr, X_te)
        return w * Q_tr + (1 - w) * L_tr, w * Q_te + (1 - w) * L_te


class ProjectedQuantumKernel(QuantumKernel):
    """Projected quantum kernel (A6, Huang et al. 2021).

    Measure <X>,<Y>,<Z> on every qubit, then an RBF kernel on those 3d numbers.
    gamma is fixed by the median heuristic (not tuned), so the only tuned knob is c
    and the tuning budget equals the fidelity kernels' and the RBF's.
    """

    def __init__(self, ordinal, circuit="Q-2L"):
        super().__init__(ordinal, circuit)
        self.name = f"P-{circuit}"

    def matrices(self, X_tr, X_te, c):
        from qksm.quantum import pauli_expectations
        S_tr, S_te = self.states(X_tr, X_te, c)
        A, B = pauli_expectations(S_tr), pauli_expectations(S_te)
        d2 = cdist(A, A, "sqeuclidean")
        med = np.median(d2[np.triu_indices_from(d2, 1)])
        gamma = 1.0 / med if med > 0 else 1.0
        return np.exp(-gamma * d2), np.exp(-gamma * cdist(B, A, "sqeuclidean"))
