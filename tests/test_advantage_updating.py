"""Advantage Updating trader: learns a trivial policy, keeps max_a A near 0, deterministic."""
import numpy as np

from src.advantage_updating import AUTrader, run_policy, train_au


def toy_problem(T=400, seed=0):
    """Paper Sec. V-C example: binomial returns, the next return's sign is known in advance.
    Optimal policy: F_t = x_t = sign(r_{t+1})."""
    rng = np.random.default_rng(seed)
    r = 0.02 * rng.choice([-1.0, 1.0], size=T + 1)
    X = np.zeros((T + 1, 2))
    X[:-1, 0] = np.sign(r[1:])          # x_t tells the sign of r_{t+1}
    X[:, 1] = rng.standard_normal(T + 1)  # a useless noise input
    return X, r, np.zeros(T + 1)


def sharpe(R, rf):
    return np.mean(R - rf) / (np.std(R - rf) + 1e-12)


def _train(seed, gammas=(0.0,), epochs=(15,)):
    X, r, rf = toy_problem()
    rng = np.random.default_rng(seed)
    agent = AUTrader(X.shape[1], rng)
    agent = train_au(agent, X, r, rf, np.arange(0, 300), (300, 399), 0.0, rng,
                     gammas=gammas, epochs=epochs, score=sharpe)
    return agent, X, r, rf


def test_learns_known_sign_policy():
    agent, X, r, rf = _train(0, gammas=(0.0, 0.75), epochs=(10, 10))
    F, _ = run_policy(agent, X, r, rf, 300, 399, 0.0)
    assert np.mean(F == X[300:399, 0]) > 0.95


def test_normalization_keeps_max_advantage_near_zero():
    agent, X, _, _ = _train(1)
    S = np.column_stack([X[:300], np.ones(300)])
    A = agent.advantages(S)
    gap = np.abs(A[:, 1] - A[:, 0]).mean()
    assert np.abs(A.max(axis=1)).mean() < 0.2 * gap


def test_deterministic():
    a1, X, r, rf = _train(2)
    a2, *_ = _train(2)
    np.testing.assert_array_equal(run_policy(a1, X, r, rf, 300, 399, 0.0)[0],
                                  run_policy(a2, X, r, rf, 300, 399, 0.0)[0])
