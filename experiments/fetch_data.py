"""Download the SUBSTITUTE raw data for Experiment 2 into data/raw/ (see NOTES.md).

  * Shiller ie_data.xls  (dividends, earnings, CPI, 10-year yield)
  * FRED: TB3MS, AAA, BAA, INDPRO, UNRATE
  * Yahoo Finance ^GSPC daily closes -> month-end closes (needs `pip install yfinance`)
"""
import os
import urllib.request

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
SHILLER = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    urllib.request.urlretrieve(SHILLER, os.path.join(RAW, "ie_data.xls"))
    for s in ["TB3MS", "AAA", "BAA", "INDPRO", "UNRATE"]:
        urllib.request.urlretrieve(FRED.format(s), os.path.join(RAW, f"{s}.csv"))
    import yfinance as yf
    d = yf.download("^GSPC", start="1948-01-01", end="1996-01-01", interval="1d",
                    auto_adjust=False, progress=False)
    d["Close"].resample("ME").last().to_csv(os.path.join(RAW, "GSPC_monthend_close.csv"))
    print("saved to", os.path.abspath(RAW))
