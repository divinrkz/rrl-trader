"""Experiment 3: half-hourly GBP/USD {long, short, neutral} trader (paper Sec. IV-B).

*** Uses SUBSTITUTE data: HistData.com GBP/USD bid/ask ticks, Jan-Aug 2010, not the
    paper's Olsen HFDF96 USD/GBP quotes for Jan-Aug 1996. NOT a replication on the
    original data. ***

Protocol (paper Sec. IV-B):
  * three positions {long, short, neutral}; trades go through the bid/ask spread
  * RRL trained to maximize the differential downside deviation ratio (Eq. 24)
  * train on the first 2000 half-hour bars, trade the next 480 (2 weeks), shift the
    window forward by 480 bars, retrain, and repeat to the end of the data
Choices the paper does not give (NOTES.md): inputs = last 8 returns scaled by the
training-window std; weights carry over between windows; fixed weights while
trading each 2-week block; rho and epochs picked on Sep-Dec 2009 pilot data.

Usage: python experiments/exp3_fx.py --seed 0
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from src.data_fx import build_fx_histdata, lagged_returns
from src.dsr import ema_sharpe
from src.rrl import TraderState, init_theta, run_online

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR = os.path.join(ROOT, "data", "raw", "histdata")
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_LABEL = "SUBSTITUTE data: HistData GBP/USD bid/ask, 2010"

# UTC; 05:00 UTC = midnight EST, the boundary of HistData's monthly files
TEST_PERIOD = ("2010-01-01 05:00", "2010-09-01 05:00")    # 8 months, like Jan-Aug 1996
PILOT_PERIOD = ("2009-09-01 05:00", "2010-01-01 05:00")   # settings chosen here only
TRAIN_BARS, BLOCK_BARS = 2000, 480             # paper
N_LAGS = 8                                     # as in Exp. 1 (not given for FX)
ETA = 0.01                                     # paper (DSR/DDR adaptation rate)
WARMUP = 100
INIT_SCALE = 0.1
RHO, EPOCHS = 0.03, 30                         # chosen by the pilot (results/exp3_pilot_log.txt)
N_TRADERS = 25


def load_fx(period):
    return build_fx_histdata(RAW_DIR, *period)


def walk_forward(fx, seed, rho=RHO, epochs=EPOCHS, objective="ddr"):
    """Paper Sec. IV-B protocol. Returns test-period positions F and returns R
    (arrays aligned with fx rows TRAIN_BARS..end)."""
    r, delta = fx.r.to_numpy(), fx.delta.to_numpy()
    rng = np.random.default_rng(seed)
    theta = init_theta(N_LAGS, rng, INIT_SCALE)
    live = TraderState(len(theta))                 # carries the position between blocks
    F_all, R_all = [], []
    for s in range(0, len(fx) - TRAIN_BARS, BLOCK_BARS):
        t0, t1 = s, s + TRAIN_BARS
        t2 = min(t1 + BLOCK_BARS, len(fx))
        X = lagged_returns(r, N_LAGS, r[t0 + 1:t1].std())    # training-window scale only
        state = TraderState(len(theta))
        for _ in range(epochs):
            state.reset_position()
            theta, state, _ = run_online(theta, X, r, state, t_start=t0 + N_LAGS, t_end=t1,
                                         rho=rho, eta=ETA, delta=delta, discrete="ternary",
                                         warmup=WARMUP, objective=objective)
        # trade the next 2 weeks with fixed weights
        _, live, hist = run_online(theta, X, r, live, t_start=t1, t_end=t2, delta=delta,
                                   discrete="ternary", eta=ETA, objective=objective)
        F_all.append(hist["F"])
        R_all.append(hist["R"])
    return np.concatenate(F_all), np.concatenate(R_all)


def bars_per_year(fx):
    """Annualization: observed bars per week x 52 (about 48 bars/day x 5 days)."""
    weeks = (fx.index[-1] - fx.index[0]) / pd.Timedelta(weeks=1)
    return len(fx) / weeks * 52


def metrics(F, R, bpy):
    """Annualized return (additive, R in units of the position value), Sharpe and
    downside deviation ratio, and the trading frequency."""
    dd = np.sqrt(np.mean(np.minimum(R, 0) ** 2))
    trades = np.mean(np.diff(np.concatenate(([0.0], F))) != 0)
    return dict(ann_return_pct=100 * R.mean() * bpy,
                sharpe=np.sqrt(bpy) * R.mean() / R.std(),
                ddr=np.sqrt(bpy) * R.mean() / dd,
                trade_freq_pct=100 * trades,
                hours_between_trades=0.5 / trades if trades > 0 else np.inf,
                pct_long=100 * np.mean(F > 0), pct_short=100 * np.mean(F < 0),
                pct_neutral=100 * np.mean(F == 0),
                total_return_pct=100 * R.sum())


def _job(args):
    fx, seed, rho, epochs = args
    return walk_forward(fx, seed, rho, epochs)


def plot_single(fx_test, F, R, out):
    from experiments.plot_style import INK_2, SERIES, plt
    t = fx_test.index
    fig, ax = plt.subplots(4, 1, figsize=(8, 7.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.3, 0.8, 1.2, 1]))
    ax[0].plot(t, fx_test.mid, color=SERIES[0], lw=0.7)
    ax[0].set_ylabel("GBP/USD mid")
    ax[0].set_title(f"RRL {{long, short, neutral}} trader, out-of-sample  [{DATA_LABEL}]",
                    fontsize=9)
    ax[1].step(t, F, where="post", color=INK_2, lw=0.6)
    ax[1].set_yticks([-1, 0, 1], ["short", "neutral", "long"])
    ax[1].set_ylim(-1.4, 1.4)
    ax[1].set_ylabel("Position")
    ax[2].plot(t, 100 * np.cumsum(R), color=SERIES[0], lw=1.1)
    ax[2].axhline(0, color=INK_2, lw=0.6)
    ax[2].set_ylabel("Cumulative\nreturn (%)")
    ax[3].plot(t, ema_sharpe(R, ETA), color=SERIES[0], lw=0.9)
    ax[3].axhline(0, color=INK_2, lw=0.6)
    ax[3].set_ylabel("Moving Sharpe\n(η = 0.01)")
    fig.align_ylabels(ax)
    fig.autofmt_xdate()
    fig.savefig(out)
    plt.close(fig)


def plot_distribution(df, out):
    from experiments.plot_style import INK_2, SERIES, plt
    cols = [("ann_return_pct", "Annualized return (%)"), ("sharpe", "Annualized Sharpe"),
            ("hours_between_trades", "Hours between trades")]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (c, title) in zip(axes, cols):
        bp = ax.boxplot(df[c], widths=0.45, patch_artist=True,
                        medianprops=dict(color="white", lw=1.5),
                        flierprops=dict(marker="o", ms=3, mec=INK_2, mfc="none"))
        bp["boxes"][0].set(facecolor=SERIES[0], edgecolor=SERIES[0], alpha=0.85)
        ax.scatter(np.ones(len(df)) + np.random.default_rng(0).uniform(-0.12, 0.12, len(df)),
                   df[c], s=8, color=INK_2, zorder=3)
        ax.set_xticks([])
        ax.set_title(title)
        if c != "hours_between_trades":
            ax.axhline(0, color=INK_2, lw=0.6)
    paper = {"ann_return_pct": 15, "sharpe": 2.3, "hours_between_trades": 5}
    for ax, (c, _) in zip(axes, cols):
        ax.axhline(paper[c], color=SERIES[1], ls="--", lw=1.2, label=f"paper: {paper[c]}")
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(f"{len(df)} traders (different seeds), out-of-sample  [{DATA_LABEL}]",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0, help="base seed; trader i uses seed+i")
    ap.add_argument("--n-traders", type=int, default=N_TRADERS)
    ap.add_argument("--rho", type=float, default=RHO)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    out_dir = os.path.join(RESULTS_DIR, "exp3_fx")
    os.makedirs(out_dir, exist_ok=True)
    print(f"seed={args.seed} n_traders={args.n_traders} rho={args.rho} epochs={args.epochs} "
          f"[{DATA_LABEL}]")

    fx = load_fx(TEST_PERIOD)
    bpy = bars_per_year(fx)
    test = fx.iloc[TRAIN_BARS:]
    print(f"{len(fx)} half-hour bars {fx.index[0]} .. {fx.index[-1]}; test {len(test)} bars "
          f"from {test.index[0]}; {bpy:.0f} bars/year; median spread "
          f"{1e4 * (fx.ask - fx.bid).median():.1f} pips")

    t = time.time()
    seeds = [args.seed + i for i in range(args.n_traders)]
    with Pool(os.cpu_count()) as pool:
        runs = pool.map(_job, [(fx, s, args.rho, args.epochs) for s in seeds])
    print(f"training took {time.time() - t:.0f}s")

    rows = [dict(seed=s, **metrics(F, R, bpy)) for s, (F, R) in zip(seeds, runs)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "exp3_traders.csv"), index=False)
    long_only = metrics(np.ones(len(test)),
                        np.concatenate(([-test.delta.iloc[0]], test.r.to_numpy()[1:])), bpy)
    summary = dict(median={c: float(df[c].median()) for c in df.columns if c != "seed"},
                   frac_positive_sharpe=float((df.sharpe > 0).mean()),
                   seed0=rows[0], long_only_gbp=long_only,
                   paper=dict(ann_return_pct=15, sharpe=2.3, hours_between_trades=5),
                   config=dict(seed=args.seed, n_traders=args.n_traders, rho=args.rho,
                               epochs=args.epochs, eta=ETA, n_lags=N_LAGS,
                               train_bars=TRAIN_BARS, block_bars=BLOCK_BARS,
                               bars_per_year=bpy, objective="ddr", data=DATA_LABEL))
    with open(os.path.join(out_dir, "exp3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: summary[k] for k in ("median", "frac_positive_sharpe", "long_only_gbp")},
                     indent=2, default=float))
    plot_single(test, *runs[0], os.path.join(out_dir, "exp3_fig6_single_run.png"))
    plot_distribution(df, os.path.join(out_dir, "exp3_distribution.png"))


if __name__ == "__main__":
    main()
