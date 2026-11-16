"""Pick rho and epochs for Experiment 3 on PILOT data only (Sep-Dec 2009).

Runs the full Exp. 3 protocol (train 2000 bars, trade 480, slide) on Sep-Dec 2009,
which is entirely before the Jan-Aug 2010 test data. Pilot seeds 100-107.
Rule (fixed before running): pick the setting with the highest mean annualized Sharpe.

Usage: python experiments/exp3_pilot.py --seed 100
"""
import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from experiments.exp3_fx import (DATA_LABEL, PILOT_PERIOD, TRAIN_BARS, bars_per_year, load_fx,
                                 metrics, walk_forward)

GRID = dict(rho=[0.003, 0.01, 0.03, 0.1], epochs=[3, 10, 30])


def job(args):
    fx, seed, rho, epochs = args
    F, R = walk_forward(fx, seed, rho, epochs)
    return metrics(F, R, bars_per_year(fx))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=8)
    args = ap.parse_args()
    print(f"seed={args.seed} n_seeds={args.n_seeds} [{DATA_LABEL}; pilot period]")
    fx = load_fx(PILOT_PERIOD)
    print(f"pilot data: {len(fx)} bars {fx.index[0]} .. {fx.index[-1]}, "
          f"pseudo-test {len(fx) - TRAIN_BARS} bars")
    seeds = range(args.seed, args.seed + args.n_seeds)
    configs = list(itertools.product(GRID["rho"], GRID["epochs"]))
    t = time.time()
    with Pool(os.cpu_count()) as pool:
        res = pool.map(job, [(fx, s, rho, ep) for rho, ep in configs for s in seeds])
    print(f"took {time.time() - t:.0f}s")
    print("rho     epochs  mean_sharpe  min_sharpe  mean_ann_ret%  hours_between_trades")
    scores = []
    for k, (rho, ep) in enumerate(configs):
        m = res[k * len(seeds):(k + 1) * len(seeds)]
        sh = np.array([x["sharpe"] for x in m])
        scores.append(sh.mean())
        print(f"{rho:<7} {ep:<7} {sh.mean():11.2f} {sh.min():11.2f} "
              f"{np.mean([x['ann_return_pct'] for x in m]):14.1f} "
              f"{np.median([x['hours_between_trades'] for x in m]):20.1f}")
    rho, ep = configs[int(np.argmax(scores))]
    print(f"best by mean Sharpe: rho={rho} epochs={ep}")


if __name__ == "__main__":
    main()
