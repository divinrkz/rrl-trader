"""Experiment 3 plumbing: half-hourly bars from ticks, bid/ask cost, no look-ahead."""
import os
import lzma

import numpy as np
import pandas as pd
import pytest

from src.data_fx import (half_hourly, lagged_returns, load_histdata_ticks, read_hour,
                         read_histdata_month)
from src.returns import additive_return


def test_read_hour_decodes_dukascopy_format(tmp_path):
    rec = np.array([(1000, 182300, 182280, 1.0, 2.0), (2000, 182310, 182290, 1.0, 1.0)],
                   dtype=[("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("av", ">f4"), ("bv", ">f4")])
    p = tmp_path / "13h_ticks.bi5"
    p.write_bytes(lzma.compress(rec.tobytes()))
    df = read_hour(p, "2004-01-06 13:00")
    assert df.index[0] == pd.Timestamp("2004-01-06 13:00:01")
    np.testing.assert_allclose(df.ask, [1.823, 1.8231])
    np.testing.assert_allclose(df.bid, [1.8228, 1.8229])


def test_half_hourly_uses_last_quote_and_drops_empty_bars():
    idx = pd.to_datetime(["2004-01-05 00:05", "2004-01-05 00:25",       # bar ending 00:30
                          "2004-01-05 00:40",                             # bar ending 01:00
                          "2004-01-05 02:10"])                            # 01:00-02:00 empty
    ticks = pd.DataFrame({"bid": [1.0, 1.1, 1.2, 1.3], "ask": [1.002, 1.102, 1.202, 1.302]},
                         index=idx)
    bars = half_hourly(ticks)
    assert list(bars.index) == list(pd.to_datetime(["2004-01-05 00:30", "2004-01-05 01:00",
                                                    "2004-01-05 02:30"]))
    np.testing.assert_allclose(bars.bid, [1.1, 1.2, 1.3])
    np.testing.assert_allclose(bars.delta, 0.001 / bars.mid)
    np.testing.assert_allclose(bars.r.iloc[1], bars.mid.iloc[1] / bars.mid.iloc[0] - 1)


def test_spread_cost_equals_trading_at_bid_and_ask():
    """Buy at the ask, hold one bar, sell at the bid: profit / mid equals the additive
    return with delta_t = half the relative spread (up to second-order terms)."""
    bid0, ask0, bid1, ask1 = 1.8000, 1.8004, 1.8100, 1.8104
    m0, m1 = (bid0 + ask0) / 2, (bid1 + ask1) / 2
    d0, d1 = (ask0 - bid0) / (2 * m0), (ask1 - bid1) / (2 * m1)
    R_open, _, _ = additive_return(0.0, 1.0, 0.0, d0)          # enter long at t0
    R_hold, _, _ = additive_return(1.0, 0.0, m1 / m0 - 1, d1)   # hold, then exit at t1
    exact = (bid1 - ask0) / m0
    assert R_open + R_hold == pytest.approx(exact, rel=1e-3)


def test_lagged_returns_use_only_the_past():
    r = np.random.default_rng(0).normal(size=50)
    X = lagged_returns(r, 8, 1.0)
    r2 = r.copy()
    r2[30:] = 99.0
    np.testing.assert_array_equal(X[:30], lagged_returns(r2, 8, 1.0)[:30])
    np.testing.assert_allclose(X[20], r[13:21])


def test_market_closed_hours_are_empty_in_cached_data():
    """The weekend hours we skip downloading really have no ticks in the files we have."""
    import glob
    from datetime import datetime
    from src.data_fx import market_closed
    root = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "dukascopy", "GBPUSD")
    checked = 0
    for p in glob.glob(os.path.join(root, "*", "*", "*", "*.bi5")):
        y, m, d, h = p.split(os.sep)[-4:]
        if market_closed(datetime(int(y), int(m), int(d), int(h[:2]))):
            assert os.path.getsize(p) == 0, p
            checked += 1
    if checked == 0:
        pytest.skip("no cached FX data")


def _write_histdata(tmp_path, yyyymm, rows):
    import zipfile
    p = tmp_path / f"HISTDATA_COM_ASCII_GBPUSD_T{yyyymm}.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(f"DAT_ASCII_GBPUSD_T_{yyyymm}.csv", "\n".join(rows) + "\n")
    return p


def test_read_histdata_month_parses_and_converts_est_to_utc(tmp_path):
    p = _write_histdata(tmp_path, "201001", ["20100103 170029000,1.610900,1.611600,0",
                                             "20100103 170031250,1.610600,1.611200,0"])
    df = read_histdata_month(p)
    assert df.index[0] == pd.Timestamp("2010-01-03 22:00:29")        # 17:00 EST = 22:00 UTC
    assert df.index[1] == pd.Timestamp("2010-01-03 22:00:31.250")
    np.testing.assert_allclose(df.bid, [1.6109, 1.6106])
    np.testing.assert_allclose(df.ask, [1.6116, 1.6112])


def test_load_histdata_picks_files_by_est_month_and_drops_crossed(tmp_path):
    _write_histdata(tmp_path, "201001", ["20100131 235900000,1.60,1.61,0",
                                         "20100131 235930000,1.61,1.61,0"])   # bid == ask: dropped
    _write_histdata(tmp_path, "201002", ["20100201 000100000,1.62,1.63,0"])
    t = load_histdata_ticks(tmp_path, "2010-02-01 04:00", "2010-02-01 06:00")
    assert list(t.index) == list(pd.to_datetime(["2010-02-01 04:59", "2010-02-01 05:01"]))
    with pytest.raises(FileNotFoundError):
        load_histdata_ticks(tmp_path, "2010-02-01", "2010-03-15")
