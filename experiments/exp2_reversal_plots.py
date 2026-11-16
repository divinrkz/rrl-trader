"""Figures: the paper's ranking holds on 1970-1994 but reverses on 1994-2018.

Reads saved positions/summaries from experiments/exp2_sp500.py runs (no retraining):
  results/exp2_reversal_equity.png   equity curves of all strategies, both periods (seed 0)
  results/exp2_reversal_ranking.png  committee final wealth for 3 base seeds, both periods
  results/exp2_reversal_why.png      1994-2002 zoom: RRL shorted the boom; dividend-yield z-score

SUBSTITUTE data (Yahoo, Shiller, FRED), not the paper's Citibase series.
Usage: python experiments/exp2_reversal_plots.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from experiments.exp2_sp500 import (DATA_LABEL, RAW_DIR, RESULTS_DIR, committee_vote,
                                    excess_sharpe, normalized_inputs, positions_to_returns,
                                    windows, year_rows)
from experiments.plot_style import GRID, INK, INK_2, SERIES, plt
from src.data_sp500 import FEATURES, build_dataset
from src.metrics import equity_curve

PERIODS = {"1970-1994": (1970, 1994), "1994-2018": (1994, 2018)}
STRATS = ["Buy and hold", "RRL", "Advantage Updating", "Plain Q-learning"]
STYLE = {"Buy and hold": dict(color=INK_2, ls=":", lw=1.6),
         "RRL": dict(color=SERIES[0], ls="-", lw=1.6),
         "Advantage Updating": dict(color=SERIES[1], ls="--", lw=1.3),
         "Plain Q-learning": dict(color=SERIES[2], ls="-.", lw=1.2)}


def run_dir(period, method, seed):
    if period == "1970-1994" and method == "q":
        return RESULTS_DIR if seed == 0 else os.path.join(RESULTS_DIR, f"exp2_seed{seed}")
    d = os.path.join(RESULTS_DIR, f"exp2_{period}_{method}")
    return d if seed == 0 else f"{d}_seed{seed}"


def load_period(period):
    y0, y1 = PERIODS[period]
    df = build_dataset(RAW_DIR, f"{y0 - 20}-01", f"{y1}-12")
    test = df.loc[f"{y0}-01":f"{y1}-12"]
    return df, test


def committee_returns(period, seed, test):
    """Monthly returns of every strategy's committee for one base seed."""
    r, rf = test.r.to_numpy(), test.rf.to_numpy()
    au = np.load(os.path.join(run_dir(period, "au", seed), "exp2_positions.npz"))
    q = np.load(os.path.join(run_dir(period, "q", seed), "exp2_positions.npz"))
    pos = {"Buy and hold": np.ones(len(r)), "RRL": committee_vote(au["F_rrl"]),
           "Advantage Updating": committee_vote(au["F_q"]),
           "Plain Q-learning": committee_vote(q["F_q"])}
    rets = {k: positions_to_returns(F, r, rf, 1.0 if k == "Buy and hold" else 0.0)
            for k, F in pos.items()}
    return pos, rets


def fig_equity(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (period, (test, rets)) in zip(axes, data.items()):
        dates = test.index.to_timestamp()
        ranked = sorted(STRATS, key=lambda k: -equity_curve(rets[k])[-1])
        finals = {k: equity_curve(rets[k])[-1] for k in STRATS}
        prev_y = None
        for k in sorted(STRATS, key=lambda k: -finals[k]):      # top to bottom
            eq = equity_curve(rets[k])
            ax.plot(dates, eq, **STYLE[k])
            # nudge labels that would overlap (log scale: compare ratios)
            dy = -9 if prev_y is not None and prev_y / eq[-1] < 1.15 else 0
            prev_y = eq[-1]
            ax.annotate(f"{eq[-1]:.1f}×", (dates[-1], eq[-1]), xytext=(4, dy),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
        ax.set_yscale("log")
        ax.set_title(f"Test {period}   ranking: " + " > ".join(
            {"Buy and hold": "B&H", "RRL": "RRL", "Advantage Updating": "AU",
             "Plain Q-learning": "Q"}[k] for k in ranked))
        ax.axhline(1, color=INK_2, lw=0.6)
    axes[0].set_ylabel("Wealth (log scale, start = 1)")
    handles = [plt.Line2D([], [], **STYLE[k]) for k in STRATS]
    fig.legend(handles, STRATS, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Same method, same settings, two periods: the paper's ranking reverses  "
                 f"[{DATA_LABEL}; committees, base seed 0]", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_ranking(table, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
    for ax, period in zip(axes, PERIODS):
        sub = table[table.period == period]
        for i, k in enumerate(STRATS):
            v = sub[sub.strategy == k].final_equity.to_numpy()
            y = np.full(len(v), len(STRATS) - 1 - i) + np.linspace(-0.12, 0.12, len(v))
            ax.scatter(v, y, s=40, color=STYLE[k]["color"], edgecolor="white", lw=1.5, zorder=3)
            ax.plot([v.min(), v.max()], [len(STRATS) - 1 - i] * 2, color=STYLE[k]["color"],
                    lw=1, alpha=0.5)
        bh = sub[sub.strategy == "Buy and hold"].final_equity.iloc[0]
        ax.axvline(bh, color=INK_2, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yticks(range(len(STRATS)), STRATS[::-1])
        ax.set_title(f"Test {period}")
        ax.set_xlabel("Committee final wealth (log scale)")
        ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5×", "1×", "2×", "5×", "10×", "20×", "50×"])
    fig.suptitle("Committee final wealth for 3 base seeds (0, 100, 200); dotted line = buy and hold"
                 f"  [{DATA_LABEL}]", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_why(df, test, pos, rets, out):
    """1994-2002: RRL short through the boom, while the dividend yield was ~2 std below
    its training-window mean (each year normalized with its own 20-year window)."""
    j = FEATURES.index("div_yield")
    z, zmin = [], []
    for year in range(1994, 2003):
        t0, _, t2 = windows(df, year)
        X = normalized_inputs(df, t0, t2)
        y0, y1 = year_rows(df, year)
        z.append(X[y0:y1, j])
        zmin.append(np.full(y1 - y0, X[t0:t2, j].min()))   # lowest value seen in training
    z, zmin = np.concatenate(z), np.concatenate(zmin)
    n = len(z)
    dates = test.index.to_timestamp()[:n]

    fig, ax = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 0.9, 1.3], hspace=0.25))
    for k in ["Buy and hold", "RRL"]:
        ax[0].plot(dates, equity_curve(rets[k][:n]), **STYLE[k], label=k)
    ax[0].set_yscale("log")
    ax[0].set_yticks([0.5, 1, 2, 3], ["0.5×", "1×", "2×", "3×"])
    ax[0].minorticks_off()
    ax[0].set_ylabel("Wealth (log)")
    ax[0].legend(loc="upper left")
    ax[0].set_title("1994–2002: the RRL committee is short through the late-1990s boom")

    ax[1].step(dates, pos["RRL"][:n], where="post", color=SERIES[0], lw=1.1)
    ax[1].set_yticks([-1, 1], ["short", "long"])
    ax[1].set_ylim(-1.5, 1.5)
    ax[1].set_ylabel("RRL position")

    ax[2].fill_between(dates, zmin, -3, step="post", color=GRID, lw=0)
    ax[2].step(dates, zmin, where="post", color=INK_2, lw=1, ls="--",
               label="lowest value in that year's 20-year training window")
    ax[2].plot(dates, z, color=INK, lw=1.1, label="dividend yield (test month)")
    ax[2].axhline(0, color=INK_2, lw=0.8)
    ax[2].set_ylim(-2.6, 1)
    ax[2].set_ylabel("Dividend yield\n(z-score)")
    ax[2].legend(loc="upper left", fontsize=8)
    ax[2].set_title("Dividend yield vs its 20-year training window (low yield → 'overvalued' signal)",
                    fontsize=9)
    fig.align_ylabels(ax)
    fig.savefig(out)
    plt.close(fig)


def main():
    rows, data = [], {}
    for period in PERIODS:
        df, test = load_period(period)
        rf = test.rf.to_numpy()
        for seed in (0, 100, 200):
            pos, rets = committee_returns(period, seed, test)
            for k in STRATS:
                rows.append(dict(period=period, seed=seed, strategy=k,
                                 final_equity=equity_curve(rets[k])[-1],
                                 sharpe=excess_sharpe(rets[k], rf)))
            if seed == 0:
                data[period] = (test, rets)
                if period == "1994-2018":
                    why = (df, test, pos, rets)
    table = pd.DataFrame(rows)
    print(table.pivot_table(index="strategy", columns=["period", "seed"],
                            values="final_equity").round(1).to_string())
    fig_equity(data, os.path.join(RESULTS_DIR, "exp2_reversal_equity.png"))
    fig_ranking(table, os.path.join(RESULTS_DIR, "exp2_reversal_ranking.png"))
    fig_why(*why, os.path.join(RESULTS_DIR, "exp2_reversal_why.png"))
    print("wrote results/exp2_reversal_{equity,ranking,why}.png")


if __name__ == "__main__":
    main()
