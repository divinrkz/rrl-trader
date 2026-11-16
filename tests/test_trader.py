"""Required tests 2-4: zero-cost sanity, no look-ahead, determinism. Plus return formulas."""
import numpy as np
import pytest

from experiments.exp1_synthetic import make_inputs, run_trial
from src.data_synthetic import generate_prices, lagged_inputs, price_differences
from src.returns import additive_return, multiplicative_return
from src.rrl import TraderState, init_theta, run_online


# ---- return formulas -------------------------------------------------------

def test_additive_return_formula():
    R, dF, dFp = additive_return(F_prev=1.0, F=-1.0, r=0.02, delta=0.005, mu=2.0)
    assert R == pytest.approx(2.0 * (0.02 - 0.005 * 2))
    assert dF == pytest.approx(2.0 * 0.005)          # -mu delta sign(-2)
    assert dFp == pytest.approx(2.0 * 0.02 - 2.0 * 0.005)


def test_multiplicative_long_and_short_legs():
    r, rf = 0.03, 0.004
    assert multiplicative_return(1.0, 1.0, r, rf, 0.005)[0] == pytest.approx(r)
    assert multiplicative_return(-1.0, -1.0, r, rf, 0.005)[0] == pytest.approx(2 * rf - r)
    R = multiplicative_return(1.0, -1.0, r, rf, 0.005)[0]   # long -> short pays 2 * delta
    assert R == pytest.approx((1 + r) * (1 - 0.01) - 1)


# ---- test 2: zero-cost sanity ----------------------------------------------

def test_zero_cost_trader_learns_positive_sharpe():
    out = run_trial(seed=7, delta=0.0, epochs=5, n=4000)
    assert out["sharpe"] > 0.1


# ---- test 3: no look-ahead ---------------------------------------------------

def test_lagged_inputs_use_only_past():
    r = np.arange(20.0)
    X = lagged_inputs(r, 8)
    np.testing.assert_array_equal(X[10], r[3:11])     # r_{t-7}, ..., r_t
    assert np.all(X[:8] == 0)


def _online_run(r, X, seed=0, delta=0.005):
    theta = init_theta(X.shape[1], np.random.default_rng(seed), 0.1)
    state = TraderState(len(theta))
    _, _, hist = run_online(theta, X, r, state, t_start=8, t_end=len(r), delta=delta,
                            rho=0.01, discrete=True, warmup=20)
    return hist


def test_future_data_does_not_change_past_positions_or_returns():
    z = generate_prices(1500, np.random.default_rng(3))
    r = price_differences(z)
    scale = r[1:500].std()
    t_cut = 900
    r2 = r.copy()
    r2[t_cut + 1:] = np.random.default_rng(99).normal(0, r.std(), len(r) - t_cut - 1)
    X, X2 = lagged_inputs(r, 8) / scale, lagged_inputs(r2, 8) / scale
    h1, h2 = _online_run(r, X), _online_run(r2, X2)
    k = t_cut - 8 + 1                                  # history index of time t_cut, inclusive
    np.testing.assert_array_equal(h1["F"][:k], h2["F"][:k])
    np.testing.assert_array_equal(h1["R"][:k], h2["R"][:k])
    assert not np.array_equal(h1["F"][k:], h2["F"][k:])


def test_return_uses_previous_position_and_current_price_change():
    z = generate_prices(1200, np.random.default_rng(4))
    r, X = make_inputs(z, n_train=500)
    delta = 0.005
    h = _online_run(r, X, delta=delta)
    F = np.concatenate(([0.0], h["F"]))                # F_7 = 0, then F_8, F_9, ...
    expected = F[:-1] * r[8:] - delta * np.abs(F[1:] - F[:-1])
    np.testing.assert_allclose(h["R"], expected)


# ---- test 4: determinism -----------------------------------------------------

def test_same_seed_same_result():
    a = run_trial(seed=11, delta=0.005, epochs=3, n=3000)
    b = run_trial(seed=11, delta=0.005, epochs=3, n=3000)
    c = run_trial(seed=12, delta=0.005, epochs=3, n=3000)
    np.testing.assert_array_equal(a["F"], b["F"])
    np.testing.assert_array_equal(a["R"], b["R"])
    assert a["sharpe"] == b["sharpe"]
    assert not np.array_equal(a["R"], c["R"])
