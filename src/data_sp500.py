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

    year = df.date.astype(float).astype(int)
    month = np.rint((df.date.astype(float) - year) * 100).astype(int)
    df.index = pd.PeriodIndex.from_fields(year=year, month=month, freq="M")
    return df.drop(columns="date").apply(pd.to_numeric, errors="coerce")


def _month_end_close(raw_dir, end):
    name = "GSPC_monthend_close.csv"
    if pd.Period(end, "M") > pd.Period("1995-12", "M"):
        name = "GSPC_monthend_close_1948_2019.csv"
    s = pd.read_csv(os.path.join(raw_dir, name), index_col=0,
                    parse_dates=True).iloc[:, 0]
    s.index = s.index.to_period("M")
    return pd.to_numeric(s, errors="coerce").rename("P")


def build_dataset(raw_dir, start="1950-01", end="1994-12"):
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

    out["div_yield"] = (df.D / df.P).shift(SHILLER_DE_LAG)
    out["earn_yield"] = (df.E / df.P).shift(SHILLER_DE_LAG)

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
