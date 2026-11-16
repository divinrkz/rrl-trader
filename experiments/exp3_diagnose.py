"""Diagnose Experiment 3: where do the losses come from? (diagnosis only, no tuning)

Re-runs the 25 traders with the frozen settings and splits each trader's return into
  gross  = F_{t-1} r_t                      (what the positions earned)
  cost   = delta_t |F_t - F_{t-1}|          (bid/ask spread paid)
and also runs the same traders with zero cost, to see whether the signal is there at all.
The zero-cost numbers are NOT a result; they only locate the problem.

*** SUBSTITUTE data: HistData GBP/USD bid/ask, 2010 (see exp3_fx.py) ***

Usage: python experiments/exp3_diagnose.py --seed 0
"""
import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from experiments.exp3_fx import (DATA_LABEL, RESULTS_DIR, TEST_PERIOD, TRAIN_BARS, bars_per_year,
                                 load_fx, walk_forward)


def _job(args):
    fx, seed = args
    return walk_forward(fx, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-traders", type=int, default=25)
    args = ap.parse_args()
    fx = load_fx(TEST_PERIOD)
    bpy = bars_per_year(fx)
    r = fx.r.to_numpy()[TRAIN_BARS:]
    fx0 = fx.copy()
    fx0["delta"] = 0.0
    seeds = range(args.seed, args.seed + args.n_traders)
    with Pool(os.cpu_count()) as pool:
        runs = pool.map(_job, [(fx, s) for s in seeds])
        runs0 = pool.map(_job, [(fx0, s) for s in seeds])

    out = [f"seed={args.seed} n_traders={args.n_traders} [{DATA_LABEL}]",
           "Annualized %, medians over traders."]
    g, c, n = [], [], []
    for F, R in runs:
        gross = np.concatenate(([0.0], F[:-1])) * r
        g.append(100 * gross.mean() * bpy)
        c.append(100 * (gross - R).mean() * bpy)
        n.append(100 * R.mean() * bpy)
    out.append(f"with spread costs: gross {np.median(g):.1f}%  spread cost {np.median(c):.1f}%  "
               f"net {np.median(n):.1f}%  (positive gross: {np.sum(np.array(g) > 0)}/{len(g)})")
    s0 = [np.sqrt(bpy) * R.mean() / R.std() for _, R in runs0]
    f0 = [100 * np.mean(np.diff(np.concatenate(([0.0], F))) != 0) for F, _ in runs0]
    out.append(f"zero cost (train and test, diagnostic only): median Sharpe {np.median(s0):.2f}, "
               f"positive {np.sum(np.array(s0) > 0)}/{len(s0)}, trades on {np.median(f0):.1f}% of bars")
    d = 1e4 * fx.delta.iloc[TRAIN_BARS:]
    out.append("median half-spread (bp) by month: "
               + str(d.groupby(d.index.month).median().round(2).to_dict()))
    text = "\n".join(out)
    print(text)
    out_dir = os.path.join(RESULTS_DIR, "exp3_fx")
    with open(os.path.join(out_dir, "exp3_cost_decomposition.txt"), "w") as f:
        f.write(text + "\n")
    df = pd.DataFrame(dict(seed=list(seeds), gross_pct=g, cost_pct=c, net_pct=n,
                           zero_cost_sharpe=s0))
    df.to_csv(os.path.join(out_dir, "exp3_cost_decomposition.csv"), index=False)
    plot_decomposition(df, os.path.join(out_dir, "exp3_why_it_fails.png"))


def plot_decomposition(df, out):
    """Left: annualized return split into position profit, spread cost and net (bars =
    medians, dots = individual traders). Right: Sharpe of the same traders with zero cost."""
    from experiments.plot_style import INK, INK_2, SERIES, plt
    rng = np.random.default_rng(0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[2.2, 1]))
    cols = [("gross_pct", "Profit from\npositions", SERIES[0], 1),
            ("cost_pct", "Spread cost\npaid", SERIES[1], -1),
            ("net_pct", "Net result", INK_2, 1)]
    for i, (c, label, color, sign) in enumerate(cols):
        vals = sign * df[c].to_numpy()
        med = np.median(vals)
        a.bar(i, med, width=0.55, color=color, alpha=0.9)
        a.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=7, color=INK, alpha=0.45,
                  zorder=3, linewidths=0)
        a.annotate(f"median {med:+.0f}%", (i + 0.3, med), xytext=(4, 0),
                   textcoords="offset points", ha="left", va="center", fontsize=8.5, color=INK,
                   fontweight="bold", zorder=4,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    a.axhline(0, color=INK_2, lw=0.8)
    a.set_xticks(range(3), [c[1] for c in cols])
    a.set_ylabel("Annualized return (%)")
    a.set_title("Where the money went (bars: median; dots: 25 traders)")
    a.grid(axis="x", visible=False)

    z = df.zero_cost_sharpe.to_numpy()
    b.scatter(rng.uniform(-0.12, 0.12, len(z)), z, s=12, color=SERIES[0], zorder=3)
    b.hlines(np.median(z), -0.25, 0.25, color=INK, lw=1.5)
    b.annotate(f"median {np.median(z):.2f}", (0.27, np.median(z)), va="center", fontsize=8,
               color=INK)
    b.axhline(0, color=INK_2, lw=0.8)
    b.axhline(2.3, color=SERIES[1], ls="--", lw=1.1)
    b.annotate("paper (with costs): 2.3", (-0.45, 2.3), xytext=(0, 4),
               textcoords="offset points", fontsize=7.5, color=INK_2)
    b.set_xlim(-0.5, 0.75)
    b.set_xticks([])
    b.set_ylabel("Annualized Sharpe")
    b.set_title("Same traders, zero trading cost")
    fig.suptitle(f"Why the FX trader fails, Mar–Aug 2010 out-of-sample  [{DATA_LABEL}]",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
