import numpy as np

from qksm import diagnostics as G

rng = np.random.default_rng(0)


def test_comparable_mask_follows_eq16():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([True, False, True])
    M = G.comparable_mask(time, event)
    assert M[0, 1] and M[0, 2]          # 0 is an event before both
    assert not M[1, 2]                  # 1 censored before 2: order unknown
    assert (M == M.T).all() and not M.diagonal().any()


def test_kta_perfect_and_null():
    n = 60
    time = rng.exponential(size=n)
    event = rng.random(n) < 0.7
    T = G.survival_target(time)
    assert G.survival_kta(T, time, event) > 0.999           # target aligned with itself
    noise = rng.random((n, n)); noise = noise @ noise.T
    assert abs(G.survival_kta(noise, time, event)) < 0.5


def test_permutation_test_detects_signal():
    n = 80
    x = rng.normal(size=n)
    time = np.exp(-x + 0.1 * rng.normal(size=n))
    event = np.ones(n, bool)
    K = np.exp(-(x[:, None] - x[None, :]) ** 2)
    obs, p, _, _ = G.kta_permutation_test(K, time, event, n_perm=100)
    assert obs > 0.3 and p < 0.05


def test_geometric_difference_of_kernel_with_itself_is_about_one():
    X = rng.normal(size=(40, 3))
    K = np.exp(-((X[:, None] - X[None]) ** 2).sum(-1) / 3)
    assert 0.9 < G.geometric_difference(K, K) <= 1.0 + 1e-6


def test_health_extremes():
    assert G.health(np.eye(30))["offdiag_std"] == 0.0
    assert abs(G.health(np.eye(30))["eff_rank"] - 30) < 1e-9
    assert abs(G.health(np.ones((30, 30)))["eff_rank"] - 1) < 1e-9
