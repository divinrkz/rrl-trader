"""SUBSTITUTE monthly S&P 500 / T-bill dataset for Experiment 2 (paper Sec. IV-C).

The paper used 84 Citibase series that are not available to us. This module
builds a small substitute from public data (sources and download date in
NOTES.md):

  * Yahoo Finance ^GSPC daily closes -> month-end S&P 500 price P_t
  * Shiller "ie_data.xls": dividends D, earnings E, CPI, 10-year yield GS10
  * FRED: TB3MS (3-month T-bill), AAA, BAA, INDPRO, UNRATE

Timing: row t is the END of month t. Everything in X[t] must be known then.
  r[t]  = (P_t + D_t / 12) / P_{t-1} - 1     S&P total return over month t
  rf[t] = TB3MS_{t-1} / 1200                 T-bill return over month t, known at t-1
Macro series are lagged one month (reporting delay, as in the paper). Shiller's
D and E are interpolated from quarterly figures, so we lag them three months.
"""
import os

import numpy as np
import pandas as pd

MACRO_LAG = 1
SHILLER_DE_LAG = 3


def _fred(raw_dir, name):
    s = pd.read_csv(os.path.join(raw_dir, f"{name}.csv"), index_col=0, parse_dates=True)
    s.index = s.index.to_period("M")
    return pd.to_numeric(s.iloc[:, 0], errors="coerce").rename(name)


def _shiller(raw_dir):
    df = pd.read_excel(os.path.join(raw_dir, "ie_data.xls"), sheet_name="Data",
                       header=None, skiprows=8, engine="xlrd")
    df = df.iloc[:, [0, 2, 3, 4, 6]]
    df.columns = ["date", "D", "E", "CPI", "GS10"]
    df = df[pd.to_numeric(df.date, errors="coerce").notna()]
    # Shiller dates are floats like 1950.1 meaning October 1950.
    year = df.date.astype(float).astype(int)
    month = np.rint((df.date.astype(float) - year) * 100).astype(int)
    df.index = pd.PeriodIndex.from_fields(year=year, month=month, freq="M")
    return df.drop(columns="date").apply(pd.to_numeric, errors="coerce")


def _month_end_close(raw_dir, end):
    # The original download stops in 1995. A second download (identical on the
    # overlap) extends it to 2019 for the later test period; use it only if needed.
    name = "GSPC_monthend_close.csv"
    if pd.Period(end, "M") > pd.Period("1995-12", "M"):
        name = "GSPC_monthend_close_1948_2019.csv"
    s = pd.read_csv(os.path.join(raw_dir, name), index_col=0,
                    parse_dates=True).iloc[:, 0]
    s.index = s.index.to_period("M")
    return pd.to_numeric(s, errors="coerce").rename("P")


def build_dataset(raw_dir, start="1950-01", end="1994-12"):
    """Return a DataFrame indexed by month with columns r, rf and the input features.

    Default range is the paper's 1950-1994. Two extra years before `start` are
    loaded so that 12-month changes and lags are defined from `start` on.
    """
    P = _month_end_close(raw_dir, end)
    sh = _shiller(raw_dir)
    fred = pd.concat([_fred(raw_dir, n) for n in ["TB3MS", "AAA", "BAA", "INDPRO", "UNRATE"]],
                     axis=1)
    first = str(pd.Period(start, "M") - 24)
    df = pd.concat([P, sh, fred], axis=1).loc[first:end]

    out = pd.DataFrame(index=df.index)
    out["r"] = (df.P + df.D / 12) / df.P.shift(1) - 1
    out["rf"] = df.TB3MS.shift(1) / 1200

    # --- price-based inputs (known at the end of month t) ---
    logp = np.log(df.P)
    out["ret_1m"] = out.r
    out["ret_1m_lag1"] = out.r.shift(1)
    out["ret_3m"] = logp.diff(3)
    out["ret_6m"] = logp.diff(6)
    out["ret_12m"] = logp.diff(12)

    # --- valuation (Shiller D, E lagged three months) ---
    out["div_yield"] = (df.D / df.P).shift(SHILLER_DE_LAG)
    out["earn_yield"] = (df.E / df.P).shift(SHILLER_DE_LAG)

    # --- rates and macro (lagged one month) ---
    m = df[["TB3MS", "AAA", "BAA", "GS10", "CPI", "INDPRO", "UNRATE"]].shift(MACRO_LAG)
    out["tbill"] = m.TB3MS
    out["tbill_diff6"] = m.TB3MS.diff(6)        # "6 month difference in T-bill yield"
    out["aaa_diff6"] = m.AAA.diff(6)            # "6 month difference in AAA bond yield"
    out["yield_curve"] = m.GS10 - m.TB3MS       # "yield curve slope"
    out["credit_spread"] = m.BAA - m.AAA
    out["inflation_12m"] = np.log(m.CPI).diff(12)
    out["indpro_12m"] = np.log(m.INDPRO).diff(12)
    out["unrate_diff6"] = m.UNRATE.diff(6)
    out["excess_tbill_vs_div"] = m.TB3MS / 100 - out.div_yield
    out = out.loc[start:end]
    if out.isna().any().any():
        raise ValueError(f"missing data in {start}..{end}: {out.columns[out.isna().any()].tolist()}")
    return out


FEATURES = ["ret_1m", "ret_1m_lag1", "ret_3m", "ret_6m", "ret_12m", "div_yield",
            "earn_yield", "tbill", "tbill_diff6", "aaa_diff6", "yield_curve",
            "credit_spread", "inflation_12m", "indpro_12m", "unrate_diff6",
            "excess_tbill_vs_div"]
