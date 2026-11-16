"""Figures for Experiment 2 (compare with paper Figs. 8 and 9)."""
import os

import numpy as np

from experiments.plot_style import INK, INK_2, SERIES, plt
from src.metrics import equity_curve

COLORS = {"RRL committee": SERIES[0], "Q committee": SERIES[1], "Buy and hold": INK_2}


def plot_all(test, series, returns, ind, F_rrl, F_q, out_dir, label, q_name="Q"):
    """q_name: display name of the value-function trader ("Q" or "AU" = Advantage Updating)."""
    dates = test.index.to_timestamp()
    y0, y1 = test.index[0].year, test.index[-1].year
    disp = lambda name: name.replace("Q committee", f"{q_name} committee")

    # Fig. 8: distribution of individual traders' final equity and Sharpe.
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, col, title in [(ax[0], "final_equity", f"Final equity (W_{y1} / W_{y0})"),
                          (ax[1], "sharpe", "Annualized excess Sharpe")]:
        data = [ind[ind.method == "Q"][col], ind[ind.method == "RRL"][col]]
        bp = a.boxplot(data, notch=True, widths=0.45, patch_artist=True,
                       medianprops=dict(color="white", lw=1.5),
                       flierprops=dict(marker="o", ms=3, mec=INK_2, mfc="none"))
        for patch, c in zip(bp["boxes"], [SERIES[1], SERIES[0]]):
            patch.set(facecolor=c, edgecolor=c, alpha=0.85)
        stat = {n: (equity_curve(R)[-1] if col == "final_equity" else None)
                for n, R in returns.items()}
        for name, ls in [("Buy and hold", ":"), ("Q committee", "--"), ("RRL committee", "-")]:
            if col == "final_equity":
                v = stat[name]
            else:
                R = returns[name]
                ex = R - test.rf.to_numpy()
                v = np.sqrt(12) * ex.mean() / ex.std()
            a.axhline(v, color=COLORS[name], ls=ls, lw=1.2, label=f"{disp(name)} ({v:.2f})")
        a.set_xticks([1, 2], [f"{q_name}-traders (n={len(data[0])})", f"RRL-traders (n={len(data[1])})"])
        a.set_title(title)
        a.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"S&P 500 / T-bill, test {y0}-{y1}, cost 0.5%  [{label}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp2_fig8_boxplots.png"))
    plt.close(fig)

    # Fig. 9: equity curves of the committees and buy-and-hold, plus signals.
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1, 1]))
    for name in ["RRL committee", "Q committee", "Buy and hold"]:
        eq = equity_curve(returns[name])
        ax[0].plot(dates, eq, color=COLORS[name], lw=1.2,
                   ls={"Buy and hold": ":", "Q committee": "--"}.get(name, "-"),
                   label=f"{disp(name)}: {eq[-1]:.1f}x")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Equity (log scale)")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title(f"Committee trading, {y0}-{y1}  [{label}]", fontsize=9)
    for a, name in [(ax[1], "RRL committee"), (ax[2], "Q committee")]:
        a.step(dates, series[name], color=COLORS[name], lw=0.9, where="post")
        a.set_yticks([-1, 1])
        a.set_ylim(-1.4, 1.4)
        a.set_ylabel(disp(name).replace(" committee", ""))
    fig.align_ylabels(ax)
    fig.savefig(os.path.join(out_dir, "exp2_fig9_equity.png"))
    plt.close(fig)
