"""Experiment 1: RRL trader on artificial price series (paper Sec. IV-A).

Reproduces
  * Fig. 2/3: one trader learning on-line from scratch over 10,000 periods,
    transaction cost 0.5% (price, signal, cumulative profit, EMA Sharpe).
  * Fig. 5:   100 trials per cost level (0.2%, 0.5%, 1%). Each trial: a fresh
    price series and fresh random weights; 100 epochs of on-line RRL on the
    first 1,000 samples, then on-line learning continues through the last
    9,000 samples, on which all statistics are computed.

Usage:  python experiments/exp1_synthetic.py --seed 0 [--trials 100] [--quick]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from src.data_synthetic import generate_prices, lagged_inputs, price_differences
from src.dsr import ema_sharpe
from src.metrics import sharpe, trading_frequency
from src.rrl import TraderState, init_theta, run_online

# ---- Settings. Paper values are marked; the rest are our choices (see NOTES.md).
N_SAMPLES = 10_000          # paper
N_TRAIN = 1_000             # paper
N_LAGS = 8                  # paper: "the previous eight returns"
EPOCHS = 100                # paper
ETA = 0.01                  # paper (DSR adaptation rate)
COSTS = [0.002, 0.005, 0.01]  # paper: 0.2%, 0.5%, 1%
RHO = 0.001                 # learning rate (not in paper; chosen by pilot on seeds 1000-1015, NOTES.md)
RHO_ONLINE = 0.01           # learning rate for the from-scratch Fig. 2 run (pilot, NOTES.md)
INIT_SCALE = 0.1            # std of initial weights (not in paper)
WARMUP = 100                # DSR warm-up steps = 1/eta (not in paper)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def make_inputs(z, n_lags=N_LAGS, n_train=N_TRAIN):
    """Price differences r_t and lagged-return inputs scaled by the std of the
    training-window returns (training statistics only)."""
    r = price_differences(z)
    scale = r[1:n_train].std()
    return r, lagged_inputs(r, n_lags) / scale


def run_trial(seed, delta, rho=RHO, epochs=EPOCHS, n=N_SAMPLES, n_train=N_TRAIN,
              init_scale=INIT_SCALE, eta=ETA):
    """One Fig. 5 trial. Returns a dict of test-period statistics and series."""
    rng = np.random.default_rng(seed)
    z = generate_prices(n, rng)
    r, X = make_inputs(z, n_train=n_train)
    theta = init_theta(N_LAGS, rng, init_scale)
    state = TraderState(len(theta))
    kw = dict(delta=delta, rho=rho, eta=eta, discrete=True, warmup=WARMUP)

    for _ in range(epochs):                 
        state.reset_position()
        theta, state, _ = run_online(theta, X, r, state, t_start=N_LAGS, t_end=n_train, **kw)

    F_before = state.F_prev
    theta, state, test = run_online(theta, X, r, state, t_start=n_train, t_end=n, **kw)

    F_with_prev = np.concatenate(([F_before], test["F"]))
    return {
        "seed": seed,
        "delta": delta,
        "sharpe": sharpe(test["R"]),
        "profit_pct": 100 * test["R"].sum(),
        "trade_freq_pct": 100 * trading_frequency(F_with_prev),
        "z": z, "F": test["F"], "R": test["R"],
    }


def run_single_online(seed, delta=0.005, rho=RHO_ONLINE, n=N_SAMPLES, init_scale=INIT_SCALE):
    """Fig. 2: a trader learning on-line from scratch over the whole series (no pre-training)."""
    rng = np.random.default_rng(seed)
    z = generate_prices(n, rng)
    r, X = make_inputs(z)
    theta = init_theta(N_LAGS, rng, init_scale)
    state = TraderState(len(theta))
    _, _, hist = run_online(theta, X, r, state, t_start=N_LAGS, t_end=n, delta=delta,
                            rho=rho, eta=ETA, discrete=True, warmup=WARMUP)
    return z, hist

PAPER_FIG5 = {0.002: dict(trade_freq_pct=6.0, profit_pct=590, sharpe=0.213),
              0.005: dict(trade_freq_pct=4.2, profit_pct=430, sharpe=0.137),
              0.01: dict(trade_freq_pct=3.1, profit_pct=225, sharpe=0.059)}

SYNTH_LABEL = "artificial data (paper Eqs. 42-44, regenerated)"

def _trial_job(args):
    seed, delta = args
    out = run_trial(seed, delta)
    return {k: out[k] for k in ("seed", "delta", "sharpe", "profit_pct", "trade_freq_pct")}

def plot_single_run(z, hist, path, title, window=None):
    """Price, signal, cumulative profit and EMA Sharpe (paper Figs. 2 and 3).
    The curves are computed over the whole run, then restricted to `window`."""
    from experiments.plot_style import INK, plt
    t = np.arange(N_LAGS, N_LAGS + len(hist["F"]))
    profit = 100 * np.cumsum(hist["R"])
    ema = ema_sharpe(hist["R"], ETA)
    w = slice(None) if window is None else window
    fig, ax = plt.subplots(4, 1, figsize=(7, 6.5), sharex=True)
    ax[0].plot(t[w], z[t][w], color=INK, lw=0.6)
    ax[0].set_ylabel("Price")
    ax[1].step(t[w], hist["F"][w], color=INK, lw=0.4, where="post")
    ax[1].set_ylabel("Signal")
    ax[1].set_yticks([-1, 0, 1])
    ax[2].plot(t[w], profit[w], color=INK, lw=0.8)
    ax[2].set_ylabel("Profit (%)")
    ax[3].plot(t[w], ema[w], color=INK, lw=0.6)
    ax[3].set_ylabel("Sharpe (EMA, eta=0.01)")
    ax[3].set_xlabel("time")
    ax[3].set_ylim(-0.3, 0.3)
    ax[0].set_title(title)
    fig.align_ylabels(ax)
    fig.savefig(path)
    plt.close(fig)


def plot_boxes(df, path):
    from experiments.plot_style import INK_2, SERIES, plt
    cols = [("trade_freq_pct", "Trading frequency (%)"),
            ("profit_pct", "Cumulative profit (%)"),
            ("sharpe", "Sharpe ratio (per period)")]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for a, (col, label) in zip(ax, cols):
        data = [df[df.delta == d][col].values for d in COSTS]
        bp = a.boxplot(data, notch=True, widths=0.5, patch_artist=True,
                       medianprops=dict(color="white", lw=1.5),
                       flierprops=dict(marker="o", ms=3, mec=INK_2, mfc="none"))
        for patch, c in zip(bp["boxes"], SERIES):
            patch.set(facecolor=c, edgecolor=c)
        paper = [PAPER_FIG5[d][col] for d in COSTS]
        a.plot(range(1, 4), paper, "D", color=INK_2, ms=5, mfc="white",
               label="paper median (Fig. 5)")
        a.set_xticks(range(1, 4), [f"{100 * d:g}%" for d in COSTS])
        a.set_xlabel("Transaction cost")
        a.set_title(label)
    ax[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"100 trials per cost level, statistics on the last 9,000 periods; {SYNTH_LABEL}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; trial i uses seed+i (same series at each cost level)")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"seed={args.seed} trials={args.trials} rho={RHO} rho_single_run={RHO_ONLINE} "
          f"eta={ETA} init_scale={INIT_SCALE}")

    # Fig. 2 / 3: one trader learning on-line from scratch, 0.5% cost.
    z, hist = run_single_online(args.seed, delta=0.005)
    plot_single_run(z, hist, os.path.join(RESULTS_DIR, "exp1_fig2_single_run.png"),
                    f"On-line RRL from scratch, cost 0.5%, seed {args.seed} ({SYNTH_LABEL})")
    plot_single_run(z, hist, os.path.join(RESULTS_DIR, "exp1_fig3_last1000.png"),
                    "Last 1,000 periods of the run above", window=slice(-1000, None))

    # Fig. 5: ensembles of trials.
    t0 = time.time()
    jobs = [(args.seed + i, d) for d in COSTS for i in range(args.trials)]
    with Pool(args.workers) as pool:
        rows = pool.map(_trial_job, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp1_trials.csv"), index=False)
    plot_boxes(df, os.path.join(RESULTS_DIR, "exp1_fig5_boxplots.png"))

    summary = {}
    for d in COSTS:
        g = df[df.delta == d]
        summary[f"{100 * d:g}%"] = {
            "median_sharpe": round(g.sharpe.median(), 4),
            "median_profit_pct": round(g.profit_pct.median(), 1),
            "median_trade_freq_pct": round(g.trade_freq_pct.median(), 2),
            "frac_positive_sharpe": round((g.sharpe > 0).mean(), 3),
            "paper_median": PAPER_FIG5[d],
        }
    summary["config"] = dict(seed=args.seed, trials=args.trials, rho=RHO,
                             rho_single_run=RHO_ONLINE, eta=ETA,
                             init_scale=INIT_SCALE, warmup=WARMUP, epochs=EPOCHS)
    with open(os.path.join(RESULTS_DIR, "exp1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"trials took {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
