"""SUBSTITUTE half-hourly GBP/USD bid/ask series for Experiment 3 (paper Sec. IV-B).

The paper used Olsen HFDF96 half-hourly USD/GBP quotes (Jan-Aug 1996, commercial).
We build the same kind of series from free tick data with real bid AND ask quotes.

Source used for the results: HistData.com GBP/USD tick data, Sep 2009 - Aug 2010
(monthly zips in data/raw/histdata/, downloaded by hand; see NOTES.md):
  * HISTDATA_COM_ASCII_GBPUSD_T{YYYYMM}.zip holds DAT_ASCII_GBPUSD_T_{YYYYMM}.csv
  * rows "YYYYMMDD HHMMSSNNN,bid,ask,volume" (volume is always 0)
  * times are EST WITHOUT daylight saving (HistData spec), so UTC = EST + 5 h, and
    each monthly file covers its EST calendar month

Earlier source (abandoned: the server throttled the download to days):
Dukascopy hourly .bi5 files (experiments/fetch_fx.py):

  * each hourly .bi5 file is LZMA-compressed; each tick is 20 bytes, big-endian:
    uint32 ms since the hour, uint32 ask, uint32 bid, float32 ask vol, float32 bid vol;
    prices are in units of 1e-5
  * a half-hourly series takes the LAST bid and ask quote in each 30-minute bar
    (the quote standing at the bar's close)
  * bars with no ticks (weekends, holidays) are dropped, giving the
    "24 hour, 5 days a week" market of the paper

Derived series, row t = close of bar t:
  mid_t   = (bid_t + ask_t) / 2
  r_t     = mid_t / mid_{t-1} - 1             return of holding 1 unit over bar t
  delta_t = (ask_t - bid_t) / (2 mid_t)       half the relative spread at time t
With the additive return R_t = F_{t-1} r_t - delta_t |F_t - F_{t-1}|, a trade
from neutral to long pays half the spread, and long -> short pays the full spread:
exactly the cost of trading at the bid/ask.
"""
import lzma
import os
import struct
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TICK = struct.Struct(">3I2f")
POINT = 1e-5


def read_hour(path, hour_start):
    """Ticks of one hour as a DataFrame indexed by UTC time, columns bid, ask."""
    blob = open(path, "rb").read()
    if not blob:
        return None
    raw = lzma.decompress(blob)
    ticks = np.frombuffer(raw, dtype=np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"),
                                               ("av", ">f4"), ("bv", ">f4")]))
    idx = pd.Timestamp(hour_start) + pd.to_timedelta(ticks["ms"].astype(np.int64), unit="ms")
    return pd.DataFrame({"bid": ticks["bid"] * POINT, "ask": ticks["ask"] * POINT}, index=idx)


def market_closed(t):
    """True for the weekend hours Fri 22:00 - Sun 20:59 UTC, when FX does not trade.
    These hours were empty in every one of the 17 weeks first downloaded
    (Sep-Dec 2003), so fetch_fx.py does not request them (NOTES.md)."""
    wd, h = t.weekday(), t.hour
    return (wd == 4 and h >= 22) or wd == 5 or (wd == 6 and h <= 20)


def load_ticks(raw_dir, start, end):
    """All ticks in [start, end) from the cached hourly files (closed weekend hours may
    be absent)."""
    frames, t = [], pd.Timestamp(start)
    missing = []
    while t < pd.Timestamp(end):
        path = os.path.join(raw_dir, f"{t:%Y}", f"{t:%m}", f"{t:%d}", f"{t:%H}h_ticks.bi5")
        if not os.path.exists(path):
            if not market_closed(t):
                missing.append(str(t))
        else:
            df = read_hour(path, t)
            if df is not None:
                frames.append(df)
        t += pd.Timedelta(hours=1)
    if missing:
        raise FileNotFoundError(f"{len(missing)} hourly files missing, e.g. {missing[:3]}; "
                                "run experiments/fetch_fx.py")
    return pd.concat(frames)


HISTDATA_UTC_OFFSET = pd.Timedelta(hours=5)       # EST, no daylight saving


def read_histdata_month(path):
    """Ticks of one HistData monthly zip, indexed by UTC time, columns bid, ask."""
    import zipfile
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        d = pd.read_csv(z.open(name), header=None, names=["t", "bid", "ask", "vol"],
                        dtype={"t": str})
    idx = pd.to_datetime(d.t, format="%Y%m%d %H%M%S%f") + HISTDATA_UTC_OFFSET
    return pd.DataFrame({"bid": d.bid.to_numpy(), "ask": d.ask.to_numpy()},
                        index=pd.DatetimeIndex(idx))


def load_histdata_ticks(raw_dir, start, end, pair="GBPUSD"):
    """All HistData ticks in [start, end) (UTC). Quotes with bid >= ask are dropped."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    months = pd.period_range(start - HISTDATA_UTC_OFFSET,
                             end - HISTDATA_UTC_OFFSET - pd.Timedelta("1ms"), freq="M")
    frames, missing = [], []
    for m in months:
        path = os.path.join(raw_dir, f"HISTDATA_COM_ASCII_{pair}_T{m:%Y%m}.zip")
        if os.path.exists(path):
            frames.append(read_histdata_month(path))
        else:
            missing.append(os.path.basename(path))
    if missing:
        raise FileNotFoundError(f"missing HistData files {missing} in {raw_dir}")
    ticks = pd.concat(frames).sort_index(kind="stable")
    ticks = ticks[(ticks.index >= start) & (ticks.index < end)]
    return ticks[ticks.ask > ticks.bid]


def build_fx_histdata(raw_dir, start, end):
    """Half-hourly GBP/USD bid/ask series for [start, end) from HistData ticks."""
    return half_hourly(load_histdata_ticks(raw_dir, start, end))


def half_hourly(ticks):
    """Last bid/ask in each 30-minute bar, bars without ticks dropped."""
    bars = ticks.resample("30min", label="right", closed="left").last().dropna()
    bars = bars[bars.ask > bars.bid]                      # drop crossed / locked quotes
    out = pd.DataFrame(index=bars.index)
    out["bid"], out["ask"] = bars.bid, bars.ask
    out["mid"] = (bars.bid + bars.ask) / 2
    out["r"] = out.mid.pct_change().fillna(0.0)
    out["delta"] = (bars.ask - bars.bid) / (2 * out.mid)
    return out


def build_fx(raw_dir, start, end):
    """Half-hourly GBP/USD bid/ask series for [start, end)."""
    return half_hourly(load_ticks(raw_dir, start, end))


def lagged_returns(r, n_lags, scale):
    """X[t] = [r_{t-n_lags+1}, ..., r_t] / scale; rows without full history are zero.
    Only data up to t is used."""
    T = len(r)
    X = np.zeros((T, n_lags))
    for j in range(n_lags):
        lag = n_lags - 1 - j
        X[lag:, j] = r[: T - lag]
    X[:n_lags] = 0.0
    return X / scale
