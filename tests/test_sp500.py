"""Experiment 2 plumbing: walk-forward windows, no look-ahead, baselines."""
import os

import numpy as np
import pandas as pd
import pytest

from experiments.exp2_sp500 import (RAW_DIR, committee_vote, normalized_inputs,
                                    positions_to_returns, windows, year_rows)
from src.q_trader import QNet, run_policy, train_q

HAVE_DATA = os.path.exists(os.path.join(RAW_DIR, "GSPC_monthend_close.csv"))


@pytest.fixture(scope="module")
def df():
    if not HAVE_DATA:
        pytest.skip("raw data not downloaded")
    from src.data_sp500 import build_dataset
    return build_dataset(RAW_DIR)


def test_windows_precede_test_year(df):
    for year in [1970, 1985, 1994]:
        t0, t1, t2 = windows(df, year)
        y0, _ = year_rows(df, year)
        assert t0 < t1 < t2 == y0
        assert df.index[t0].year == year - 20 and df.index[t2 - 1].year == year - 1
        assert t1 - t0 == 120 and t2 - t1 == 120


def test_normalization_uses_window_only(df):
    t0, _, t2 = windows(df, 1980)
    X = normalized_inputs(df, t0, t2)
    df2 = df.copy()
    df2.iloc[t2:, 2:] = 1e6                        # corrupt everything after the window
    np.testing.assert_array_equal(X[:t2], normalized_inputs(df2, t0, t2)[:t2])


@pytest.mark.parametrize("start,end,cut", [("1950-01", "1994-12", "1980-06"),
                                           ("1974-01", "2018-12", "2005-06")])
def test_features_do_not_use_future_prices(start, end, cut):
    """Rebuild the dataset with prices after `cut` altered; earlier rows must not change."""
    if not HAVE_DATA:
        pytest.skip("raw data not downloaded")
    import src.data_sp500 as D
    orig = D._month_end_close
    full = D.build_dataset(RAW_DIR, start, end)

    def altered(raw_dir, end_):
        P = orig(raw_dir, end_).copy()
        P[P.index > pd.Period(cut, "M")] *= 3.0
        return P
    D._month_end_close = altered
    try:
        alt = D.build_dataset(RAW_DIR, start, end)
    finally:
        D._month_end_close = orig
    pd.testing.assert_frame_equal(full.loc[:cut], alt.loc[:cut])
    nxt = str(pd.Period(cut, "M") + 1)
    assert not full.loc[nxt].equals(alt.loc[nxt])


def test_buy_and_hold_returns_equal_index_returns():
    r, rf = np.array([0.01, -0.02, 0.03]), np.array([0.004, 0.004, 0.004])
    R = positions_to_returns(np.ones(3), r, rf, F_before=1.0)
    np.testing.assert_allclose(R, r)


def test_committee_vote_majority_and_ties():
    F = np.array([[1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1], [-1, 1, -1, 1]], float)
    np.testing.assert_array_equal(committee_vote(F), [1, 1, -1, -1])


def test_q_trader_learns_predictable_market():
    rng = np.random.default_rng(0)
    T = 400
    X = rng.normal(size=(T, 3))
    r = np.zeros(T)
    r[1:] = 0.03 * np.sign(X[:-1, 0])               # next return = sign of today's input 0
    rf = np.full(T, 0.003)
    net = QNet(5, rng)
    score = lambda R, rf_: np.mean(R)
    net = train_q(net, X, r, rf, np.arange(0, 199), (200, 300), 0.005, rng, score=score,
                  iterations=(2, 0))
    F, R = run_policy(net, X, r, rf, 300, 399, 0.005)
    assert np.mean(F[:-1] == np.sign(X[300:398, 0])) > 0.9
