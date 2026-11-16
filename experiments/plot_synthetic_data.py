"""Visualize the artificial price series of paper Sec. IV-A.1 (no trading).

Figure A (results/synthetic_data_overview.png)
  1. One full 10,000-step series, with the 1,000-step training window shaded.
  2. A 300-step zoom of the price.
  3. The hidden trend beta(t) over the same 300 steps (not seen by the trader).

Figure B (results/synthetic_data_properties.png)
  1. Three independent series (different seeds): no two look alike.
  2. Autocorrelation of returns, measured over 100 series vs theory.
  3. Size of one-step price moves |r_t| vs the cost of a long->short reversal.

Usage: python experiments/plot_synthetic_data.py --seed 0
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.plot_style import GRID, INK, INK_2, SERIES, plt  # noqa: E402
from src.data_synthetic import generate_latent, price_differences  # noqa: E402

N, N_TRAIN = 10_000, 1_000
ALPHA, K = 0.9, 3.0
COSTS = [0.002, 0.005, 0.01]          # delta for 0.2%, 0.5%, 1%
LAGS = np.arange(1, 31)


def make_series(seed):
    p, beta = generate_latent(N, np.random.default_rng(seed), ALPHA, K)
    z = np.exp(p / (p.max() - p.min()))
    return z, beta


def autocorr(x, lag):
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


def theory_autocorr(lag):
    """Corr of log-price increments: alpha^lag * var(beta) / (var(beta) + k^2)."""
    var_beta = 1.0 / (1.0 - ALPHA**2)
    return ALPHA**lag * var_beta / (var_beta + K**2)


def figure_overview(seed, out):
    z, beta = make_series(seed)
    t = np.arange(N)
    z0, z1 = 4_000, 4_300                       # zoom window (inside the test set)

    fig = plt.figure(figsize=(9, 6.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1, 0.8], hspace=0.45)

    ax = fig.add_subplot(gs[0])
    ax.axvspan(0, N_TRAIN, color=GRID, lw=0)
    ax.plot(t, z, color=SERIES[0], lw=0.6)
    ax.axvspan(z0, z1, fill=False, ec=INK_2, lw=0.8, ls="--")
    ax.text(N_TRAIN / 2, 0.97, "train\n(1,000)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=INK_2)
    ax.text(7_000, 0.97, "test (9,000), online learning continues",
            transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK_2)
    ax.set(title=f"Synthetic price z(t), one full series (seed {seed})",
           xlim=(0, N), ylabel="price z(t)")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t[z0:z1], z[z0:z1], color=SERIES[0], lw=1.2)
    ax2.set(title=f"Zoom: steps {z0}–{z1} (dashed box above)", ylabel="price z(t)",
            xlim=(z0, z1))

    ax3 = fig.add_subplot(gs[2], sharex=ax2)
    ax3.axhline(0, color=INK_2, lw=0.8)
    ax3.fill_between(t[z0:z1], beta[z0:z1], 0, where=beta[z0:z1] > 0,
                     color=SERIES[2], alpha=0.35, lw=0, label="up-trend")
    ax3.fill_between(t[z0:z1], beta[z0:z1], 0, where=beta[z0:z1] < 0,
                     color=SERIES[1], alpha=0.35, lw=0, label="down-trend")
    ax3.plot(t[z0:z1], beta[z0:z1], color=INK, lw=1.0)
    ax3.set(title="Hidden trend β(t), which drives the price but is NOT seen by the trader",
            xlabel="time step t", ylabel="β(t)")
    ax3.legend(loc="upper right", ncol=2)

    fig.suptitle("Artificial data, paper Sec. IV-A (Eqs. 42–44): "
                 "p(t) = p(t−1) + β(t−1) + 3ε(t),  β(t) = 0.9β(t−1) + ν(t)",
                 fontsize=9, color=INK_2, y=0.995)
    fig.savefig(out)
    plt.close(fig)


def figure_properties(seed, out, n_series=100):
    rng_seeds = [seed + i for i in range(n_series)]
    acs, abs_r = [], []
    for s in rng_seeds:
        z, _ = make_series(s)
        dp = np.diff(np.log(z))
        acs.append([autocorr(dp, L) for L in LAGS])
        abs_r.append(np.abs(price_differences(z)[1:]))
    acs = np.array(acs)
    abs_r = np.concatenate(abs_r)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"wspace": 0.3})

    ax = axes[0]
    for i in range(3):
        z, _ = make_series(seed + i)
        ax.plot(z, color=SERIES[i], lw=0.6, label=f"seed {seed + i}")
    ax.set(title="Three independent series", xlabel="time step t", ylabel="price z(t)",
           xlim=(0, N))
    ax.legend(loc="upper left")

    ax = axes[1]
    mean, lo, hi = acs.mean(0), *np.percentile(acs, [5, 95], axis=0)
    ax.fill_between(LAGS, lo, hi, color=SERIES[0], alpha=0.2, lw=0, label="5–95% of series")
    ax.plot(LAGS, mean, "o", color=SERIES[0], ms=3.5, label=f"measured ({n_series} series)")
    ax.plot(LAGS, theory_autocorr(LAGS), color=INK, lw=1.0, label="theory")
    ax.axvspan(0.5, 8.5, color=GRID, lw=0, zorder=0)
    ax.text(4.5, 0.02, "8 lags fed\nto the trader", ha="center", va="bottom", color=INK_2)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(title="Returns are predictable only short-term", xlabel="lag (steps)",
           ylabel="autocorrelation of returns", xlim=(0.5, LAGS[-1] + 0.5))
    ax.legend(loc="center right")

    ax = axes[2]
    bins = np.linspace(0, 0.025, 60)
    ax.hist(abs_r, bins=bins, color=SERIES[0], alpha=0.6, density=True)
    med = np.median(abs_r)
    ax.axvline(med, color=INK, lw=1.0)
    ax.text(med, 0.97, f" median |r_t|\n = {med:.4f}", transform=ax.get_xaxis_transform(),
            va="top", ha="left", color=INK)
    for d, ls, y in zip(COSTS, [":", "--", "-."], [0.75, 0.55, 0.35]):
        ax.axvline(2 * d, color=INK_2, lw=1.0, ls=ls)
        ax.text(2 * d, y, f" 2δ at {d * 100:g}%\n = {2 * d / med:.0f}× median",
                transform=ax.get_xaxis_transform(), va="top", ha="left", color=INK_2,
                fontsize=8)
    ax.set(title="One reversal costs many typical moves", xlabel="|r_t| = |z_t − z_{t−1}|",
           ylabel="density", xlim=(0, bins[-1]))
    ax.set_yticks([])

    fig.savefig(out)
    plt.close(fig)
    return med, acs.mean(0)[:5]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0, help="seed of the first series")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"seed={args.seed}")

    a = os.path.join(args.out, "synthetic_data_overview.png")
    b = os.path.join(args.out, "synthetic_data_properties.png")
    figure_overview(args.seed, a)
    med, ac = figure_properties(args.seed, b)
    print(f"median |r_t| = {med:.4f}; autocorr lags 1-5 = {np.round(ac, 3)}")
    print(f"wrote {a}\nwrote {b}")


if __name__ == "__main__":
    main()
