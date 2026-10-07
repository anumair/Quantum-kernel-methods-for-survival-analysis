"""Quantum feature maps and kernels, simulated exactly with Qiskit statevectors.

Each patient x (d features, rescaled to angles in [0, pi]) becomes a d-qubit
state |psi(x)> = U(x)|0...0>. Kernels compare these states:

    fidelity kernel   k(x, x') = |<psi(x)|psi(x')>|^2                 (A1-A5)
    projected kernel  k(x, x') = exp(-gamma * sum_q ||rho_q(x) - rho_q(x')||^2)   (A6)

Why the 1-layer circuit is classical: any data-independent gates after the last
encoding layer cancel in <psi(x)|psi(x')> (C^dagger C = I). So every entangling
circuit here re-encodes the data *after* its CNOTs. See docs/PLAN.md, Step 2.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qksm.data import to_numeric

CIRCUITS = ("Q-1L", "Q-2L", "Q-3L", "Q-ZZ", "Q-Z")


# ---------------------------------------------------------------- circuits


def _encode(qc, theta):
    for q, t in enumerate(theta):
        qc.ry(t, q)


def _cnot_ring(qc):
    n = qc.num_qubits
    if n < 2:
        return
    for q in range(n - 1):
        qc.cx(q, q + 1)
    if n > 2:
        qc.cx(n - 1, 0)


def build_circuit(name, theta, trailing_ring=False):
    """Feature-map circuit for one patient. `theta` = angles (already c * x in [0, pi]).

    trailing_ring=True appends a final CNOT ring. It exists only so the tests can
    show that it has no effect on the kernel.
    """
    n = len(theta)
    qc = QuantumCircuit(n)
    if name == "Q-1L":                       # A1: product state, no entanglement
        _encode(qc, theta)
    elif name in ("Q-2L", "Q-3L"):           # A2/A3: encode -> ring -> encode [-> ring -> encode]
        layers = 2 if name == "Q-2L" else 3
        _encode(qc, theta)
        for _ in range(layers - 1):
            _cnot_ring(qc)
            _encode(qc, theta)
    elif name in ("Q-ZZ", "Q-Z"):            # A4: ZZ feature map, 2 repetitions
        # A4b "Q-Z" is the ablation without the ZZ entanglers (a product state, so classical):
        # it separates the effect of the H+RZ encoding from the effect of entanglement.
        pairs = [(q, q + 1) for q in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
        for _ in range(2):
            qc.h(range(n))
            for q, t in enumerate(theta):
                qc.rz(t, q)
            if name == "Q-ZZ":
                for a, b in pairs:
                    qc.rzz(theta[a] * theta[b] / np.pi, a, b)   # max c*pi <= pi: no wrap-around
    else:
        raise ValueError(f"unknown circuit {name!r}")
    if trailing_ring:
        _cnot_ring(qc)
    return qc


def statevectors(name, angles, trailing_ring=False):
    """Statevectors for all patients: array of shape (n_patients, 2**n_qubits)."""
    return np.array([Statevector(build_circuit(name, th, trailing_ring)).data for th in angles])


# ---------------------------------------------------------------- kernels


def fidelity_kernel(S_a, S_b):
    """K[i, j] = |<psi_a_i | psi_b_j>|^2 for all pairs, in one matrix product."""
    return np.abs(S_a.conj() @ S_b.T) ** 2


def pauli_expectations(S):
    """<X>, <Y>, <Z> on every qubit (single-qubit reduced density matrices).

    Returns shape (n_patients, 3 * n_qubits). Qiskit is little-endian: qubit q is
    tensor axis (n - 1 - q) of the reshaped statevector.
    """
    N, dim = S.shape
    n = int(np.log2(dim))
    T = S.reshape((N,) + (2,) * n)
    feats = []
    for q in range(n):
        axis = 1 + (n - 1 - q)
        M = np.moveaxis(T, axis, 1).reshape(N, 2, -1)
        rho = np.einsum("nik,njk->nij", M, M.conj())       # 2x2 reduced density matrix
        feats += [2 * rho[:, 0, 1].real, -2 * rho[:, 0, 1].imag, (rho[:, 0, 0] - rho[:, 1, 1]).real]
    return np.stack(feats, axis=1)


# ---------------------------------------------------------------- preprocessing


class AngleScaler:
    """Numeric encoding + min-max to [0, pi], fitted on training data; test values clipped."""

    def __init__(self, ordinal):
        self.ordinal = ordinal

    def fit(self, X_train):
        A = to_numeric(X_train, self.ordinal)
        self.columns_ = A.columns
        self.lo_, self.hi_ = A.min().to_numpy(), A.max().to_numpy()
        return self

    def transform(self, X):
        A = to_numeric(X, self.ordinal).reindex(columns=self.columns_, fill_value=0.0).to_numpy()
        span = np.where(self.hi_ > self.lo_, self.hi_ - self.lo_, 1.0)
        return np.clip((A - self.lo_) / span, 0.0, 1.0) * np.pi
