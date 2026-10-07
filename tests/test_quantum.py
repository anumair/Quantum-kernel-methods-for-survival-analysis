import numpy as np
import pytest

from qksm import data as D
from qksm import kernels as Kn
from qksm.quantum import CIRCUITS, fidelity_kernel, pauli_expectations, statevectors

rng = np.random.default_rng(0)
ANGLES = rng.uniform(0, np.pi, size=(12, 5))  # 12 patients, 5 qubits


def product_of_cosines(A, B):
    """Analytic kernel of RY-only product states: prod_k cos^2((a_k - b_k) / 2)."""
    return np.prod(np.cos((A[:, None, :] - B[None, :, :]) / 2) ** 2, axis=2)


def kernel(name, angles=ANGLES, trailing_ring=False):
    S = statevectors(name, angles, trailing_ring)
    return fidelity_kernel(S, S)


# --- the trailing-CNOT bug (docs/PLAN.md, Step 2) ---------------------------------


def test_1L_equals_product_of_cosines():
    """Test 1: the no-entanglement control is exactly the classical cos^2 kernel."""
    assert np.allclose(kernel("Q-1L"), product_of_cosines(ANGLES, ANGLES), atol=1e-10)


@pytest.mark.parametrize("name", CIRCUITS)
def test_trailing_cnot_ring_has_no_effect(name):
    """Test 2: data-independent gates after the last encoding cancel in the overlap."""
    assert np.allclose(kernel(name), kernel(name, trailing_ring=True), atol=1e-10)


def test_1L_plus_trailing_ring_is_still_classical():
    """The old bug: 'RY then CNOT ring' is still just the product-of-cosines kernel."""
    assert np.allclose(kernel("Q-1L", trailing_ring=True), product_of_cosines(ANGLES, ANGLES), atol=1e-10)


@pytest.mark.parametrize("name", ["Q-2L", "Q-3L", "Q-ZZ"])
def test_entangled_circuits_differ_from_product(name):
    """Test 3: entanglement + re-encoding actually changes the kernel."""
    assert np.abs(kernel(name) - product_of_cosines(ANGLES, ANGLES)).max() > 1e-2


# --- kernel sanity ----------------------------------------------------------------


@pytest.mark.parametrize("name", CIRCUITS)
def test_fidelity_kernel_is_valid(name):
    K = kernel(name)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-10      # positive semi-definite
    assert K.min() >= -1e-12 and K.max() <= 1 + 1e-12


def test_pauli_expectations_product_state():
    """RY(t)|0> has <X> = sin t, <Y> = 0, <Z> = cos t on each qubit."""
    E = pauli_expectations(statevectors("Q-1L", ANGLES))
    assert np.allclose(E[:, 0::3], np.sin(ANGLES), atol=1e-10)
    assert np.allclose(E[:, 1::3], 0.0, atol=1e-10)
    assert np.allclose(E[:, 2::3], np.cos(ANGLES), atol=1e-10)


@pytest.mark.parametrize("make", [lambda o: Kn.QuantumKernel(o, "Q-2L"), lambda o: Kn.ProjectedQuantumKernel(o)])
def test_kernel_interface_on_vlc(make):
    d = D.load("vlc")
    k = make(d.ordinal)
    K_tr, K_te = k.matrices(d.X.iloc[:30], d.X.iloc[30:40], k.grid[3])
    assert K_tr.shape == (30, 30) and K_te.shape == (10, 30)
    assert np.allclose(np.diag(K_tr), 1.0)
