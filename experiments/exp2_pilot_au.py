"""Pick Advantage Updating settings on PRE-1970 data only (pseudo walk-forward).

Same pilot protocol as used for the RRL and Q settings (NOTES.md):
  pseudo-test years 1965-69; for year Y train on 1950..Y-1 (bootstrap), validate /
  early-stop on Y-5..Y-1; pilot seeds 100-107 (disjoint from the reported seeds 0-29).
Score: mean over seeds of the annualized excess Sharpe over the 60 pseudo-test months.
No data after 1969 is used.

Usage: python experiments/exp2_pilot_au.py --seed 100
"""
import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from experiments.exp2_sp500 import (AU_SETTINGS, DELTA, RAW_DIR, excess_sharpe,
                                    normalized_inputs, positions_to_returns, year_rows)
from src import advantage_updating as au
from src.data_sp500 import FEATURES, build_dataset

PSEUDO_TEST = range(1965, 1970)
GRID = dict(alpha=[0.01, 0.03, 0.1], omega=[0.03, 0.1, 0.3])


def pilot_trader(args):
    seed, settings, df = args
    r, rf = df.r.to_numpy(), df.rf.to_numpy()
    rng = np.random.default_rng(seed)
    agent = au.AUTrader(len(FEATURES), rng)
    t0 = year_rows(df, 1950)[0]
    F_prev, F_all = 0.0, []
    for year in PSEUDO_TEST:
        t1, t2 = year_rows(df, year - 5)[0], year_rows(df, year - 1)[1]
        X = normalized_inputs(df, t0, t2)
        agent = au.train_au(agent, X, r, rf, np.arange(t0, t2 - 1), (t1, t2), DELTA, rng,
                            score=excess_sharpe, **settings)
        y0, y1 = year_rows(df, year)
        F, _ = au.run_policy(agent, X, r, rf, y0, y1, DELTA, F_prev=F_prev)
        F_prev = F[-1]
        F_all.append(F)
    y0, y1 = year_rows(df, PSEUDO_TEST[0])[0], year_rows(df, PSEUDO_TEST[-1])[1]
    R = positions_to_returns(np.concatenate(F_all), r[y0:y1], rf[y0:y1], 0.0)
    return excess_sharpe(R, rf[y0:y1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=100, help="first pilot seed")
    ap.add_argument("--n-seeds", type=int, default=8)
    args = ap.parse_args()
    print(f"seed={args.seed} n_seeds={args.n_seeds}")
    df = build_dataset(RAW_DIR, "1950-01", "1969-12")      # nothing after 1969
    seeds = range(args.seed, args.seed + args.n_seeds)
    configs = [dict(AU_SETTINGS, alpha=a, beta=a, omega=w)
               for a, w in itertools.product(GRID["alpha"], GRID["omega"])]
    jobs = [(s, c, df) for c in configs for s in seeds]
    t = time.time()
    with Pool(os.cpu_count()) as pool:
        out = np.array(pool.map(pilot_trader, jobs)).reshape(len(configs), len(seeds))
    print(f"took {time.time() - t:.0f}s")
    yr0, yr1 = year_rows(df, 1965)[0], year_rows(df, 1969)[1]
    bh = positions_to_returns(np.ones(yr1 - yr0), df.r.to_numpy()[yr0:yr1],
                              df.rf.to_numpy()[yr0:yr1], 1.0)
    print(f"buy-and-hold pseudo-test excess Sharpe: {excess_sharpe(bh, df.rf.to_numpy()[yr0:yr1]):.3f}")
    print("alpha=beta  omega   mean   median   min")
    for c, row in zip(configs, out):
        print(f"{c['alpha']:<10} {c['omega']:<6} {row.mean():6.3f} {np.median(row):7.3f} {row.min():6.3f}")
    best = configs[int(np.argmax(out.mean(axis=1)))]
    print("best by mean:", {k: best[k] for k in ("alpha", "beta", "omega")})


if __name__ == "__main__":
    main()
