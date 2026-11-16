"""Collect all Exp. 2 runs (periods x Q-method x base seed) into one table.

Reads results/**/exp2_summary.json and writes results/exp2_comparison.csv.
Usage: python experiments/exp2_compare.py
"""
import glob
import json
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "**", "exp2_summary.json"),
                             recursive=True)):
        s = json.load(open(path))
        c = s["config"]
        period, method = c.get("test_period", "1970-1994"), c.get("q_method", "q")
        for name in ["Buy and hold", "RRL committee", "Q committee"]:
            label = name.replace("Q committee", f"{method.upper()} committee")
            rows.append(dict(period=period, q_method=method, seed=c["seed"], strategy=label,
                             final_equity=round(s[name]["final_equity"], 1),
                             sharpe=round(s[name]["sharpe"], 2),
                             max_dd_pct=round(100 * s[name]["max_drawdown"]),
                             trades_pct=round(s[name]["trade_freq_pct"], 1)))
        for kind in ["RRL", "Q"]:
            ind = s[f"{kind} individual traders"]
            label = "RRL" if kind == "RRL" else method.upper()
            rows.append(dict(period=period, q_method=method, seed=c["seed"],
                             strategy=f"{label} traders (median)",
                             final_equity=round(ind["final_equity"]["median"], 1),
                             sharpe=round(ind["sharpe"]["median"], 2),
                             max_dd_pct=round(100 * ind["max_drawdown"]["median"]),
                             trades_pct=round(ind["trade_freq_pct"]["median"], 1)))
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(os.path.join(ROOT, "results", "exp2_comparison.csv"), index=False)
    with pd.option_context("display.width", 140, "display.max_rows", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
