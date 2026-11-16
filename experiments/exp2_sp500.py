"""Experiment 2: S&P 500 / T-bill asset allocation, RRL vs Q-learning (paper Sec. IV-C).

*** Uses SUBSTITUTE public data (Yahoo, Shiller, FRED), not the paper's 84
    Citibase series. Results are NOT a replication on the original data. ***

Protocol (paper Sec. IV-C.1):
  * Long/short S&P 500. Long earns the S&P total return; short earns 2x the
    T-bill rate minus the S&P return (2:1 leverage). Cost 0.5%. Multiplicative
    profits (Eq. 7), profits reinvested.
  * For each test year 1970..1994: train on the previous 20 years, first 10 for
    training, last 10 for validation/early stopping. 1970 starts from random
    weights; later years start from the previous year's weights.
  * RRL: single tanh unit (thresholded when trading), weight decay 0.01,
    differential Sharpe ratio. 30 traders; committee = majority vote.
  * Q-trader: 30-unit tanh network, bootstrap of the training window,
    gamma 0 then 0.75. 10 traders; committee = majority vote.

Options beyond the paper (see NOTES.md):
  --q-method au        use Advantage Updating (the paper's choice) instead of plain Q-learning
  --test-start/--test-end   other test periods, e.g. 1994-2018. ALL settings stay frozen
                            at the values chosen on pre-1970 data; nothing is re-tuned.

Usage:  python experiments/exp2_sp500.py --seed 0
        python experiments/exp2_sp500.py --seed 0 --q-method au
        python experiments/exp2_sp500.py --seed 0 --q-method au --test-start 1994 --test-end 2018
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

from src.data_sp500 import FEATURES, build_dataset
from src.metrics import annualized_sharpe, equity_curve, max_drawdown, trading_frequency
from src import advantage_updating as au
from src.q_trader import QNet, run_policy, train_q
from src.returns import multiplicative_return
from src.rrl import TraderState, init_theta, run_online

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR = os.path.join(ROOT, "data", "raw")
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_LABEL = "SUBSTITUTE data: Yahoo ^GSPC, Shiller, FRED"

DELTA = 0.005             # paper: 0.5%
TEST_YEARS = range(1970, 1995)
N_RRL, N_Q = 30, 10       # paper
WEIGHT_DECAY = 0.01       # paper
ETA = 0.01                # DSR adaptation rate (paper value for Exp. 1; not restated here)
RRL_RHO = 0.03            # not in paper; chosen on the 1970 window only (NOTES.md)
RRL_EPOCHS = 100          # max epochs per year; best epoch picked on validation
RRL_INIT_SCALE = 0.1
Q_SETTINGS = dict(fit_epochs=30, weight_decay=1e-2)   # not in paper; pre-1970 pilot (NOTES.md)
RRL_WARMUP = 120          # first pass over the training window only warms up A, B
AU_SETTINGS = dict(alpha=0.01, beta=0.01, omega=0.3, epochs=(10, 20), batch=32,
                   weight_decay=1e-3)  # not in paper; pre-1970 pilot (NOTES.md)


def excess_sharpe(R, rf):
    return annualized_sharpe(np.asarray(R) - np.asarray(rf))


def normalized_inputs(df, t0, t1):
    """Standardize features with statistics of rows t0..t1-1 (the 20-year window)."""
    X = df[FEATURES].to_numpy()
    mu, sd = X[t0:t1].mean(axis=0), X[t0:t1].std(axis=0)
    return (X - mu) / sd


def year_rows(df, year):
    idx = np.flatnonzero(df.index.year == year)
    return idx[0], idx[-1] + 1


def windows(df, year):
    """(train_start, val_start, val_end) row indices for test year `year`."""
    t0 = year_rows(df, year - 20)[0]
    t1 = year_rows(df, year - 10)[0]
    t2 = year_rows(df, year - 1)[1]
    return t0, t1, t2


# ---------------------------------------------------------------- RRL trader

def train_rrl_year(theta, X, r, rf, t0, t1, t2, rho=RRL_RHO, epochs=RRL_EPOCHS):
    """Online RRL passes over [t0, t1); keep the epoch with the best validation
    Sharpe on [t1, t2) (early stopping)."""
    state = TraderState(len(theta))
    kw = dict(delta=DELTA, eta=ETA, discrete=True, rf=rf)
    best_theta, best = theta.copy(), -np.inf
    for _ in range(epochs):
        state.reset_position()
        theta, state, _ = run_online(theta, X, r, state, t_start=t0, t_end=t1, rho=rho,
                                     weight_decay=WEIGHT_DECAY, warmup=RRL_WARMUP, **kw)
        _, _, val = run_online(theta, X, r, TraderState(len(theta)), t_start=t1, t_end=t2,
                               **kw)
        s = excess_sharpe(val["R"], rf[t1:t2])
        if s > best:
            best, best_theta = s, theta.copy()
    return best_theta


def run_rrl_trader(seed, df=None, rho=RRL_RHO, years=TEST_YEARS):
    """Walk forward through the test years. Returns test positions (one per month)."""
    df = build_dataset(RAW_DIR) if df is None else df
    r, rf = df.r.to_numpy(), df.rf.to_numpy()
    rng = np.random.default_rng(seed)
    theta = init_theta(len(FEATURES), rng, RRL_INIT_SCALE)
    live = TraderState(len(theta))
    F_all = []
    for year in years:
        t0, t1, t2 = windows(df, year)
        X = normalized_inputs(df, t0, t2)
        theta = train_rrl_year(theta, X, r, rf, t0, t1, t2, rho=rho)
        y0, y1 = year_rows(df, year)
        _, live, hist = run_online(theta, X, r, live, t_start=y0, t_end=y1, delta=DELTA,
                                   discrete=True, rf=rf)
        F_all.append(hist["F"])
    return np.concatenate(F_all)


# ---------------------------------------------------------------- Q trader

def run_q_trader(seed, df=None, years=TEST_YEARS, method="q", **q_kwargs):
    """Value-function trader. method "q": plain fitted Q-learning; "au": Advantage Updating."""
    df = build_dataset(RAW_DIR) if df is None else df
    r, rf = df.r.to_numpy(), df.rf.to_numpy()
    rng = np.random.default_rng(seed)
    if method == "au":
        net = au.AUTrader(len(FEATURES), rng)
        train, policy, settings = au.train_au, au.run_policy, AU_SETTINGS
    else:
        net = QNet(len(FEATURES) + 2, rng)
        train, policy, settings = train_q, run_policy, Q_SETTINGS
    F_prev, F_all = 0.0, []
    for year in years:
        t0, t1, t2 = windows(df, year)
        X = normalized_inputs(df, t0, t2)
        # Paper: bootstrap sample of the whole 20-year window for training, final
        # 10 years for early stopping (so validation overlaps training, as in the paper).
        # Decision times whose reward month is still inside the window:
        train_times = np.arange(t0, t2 - 1)
        net = train(net, X, r, rf, train_times, (t1, t2), DELTA, rng, score=excess_sharpe,
                    **{**settings, **q_kwargs})
        y0, y1 = year_rows(df, year)
        F, _ = policy(net, X, r, rf, y0, y1, DELTA, F_prev=F_prev)
        F_prev = F[-1]
        F_all.append(F)
    return np.concatenate(F_all)


# ---------------------------------------------------------------- evaluation

def positions_to_returns(F, r, rf, F_before):
    """Returns of holding F over the test months, with the paper's accounting:
    R_t uses F_{t-1} and r_t, and pays the cost of moving to F_t."""
    Fp = np.concatenate(([F_before], F[:-1]))
    return np.array([multiplicative_return(a, b, rt, rft, DELTA)[0]
                     for a, b, rt, rft in zip(Fp, F, r, rf)])


def committee_vote(F_matrix):
    """Majority vote of +-1 positions; a tie keeps the previous committee position."""
    s = F_matrix.sum(axis=0)
    out, prev = np.empty(F_matrix.shape[1]), 1.0
    for i, v in enumerate(s):
        prev = np.sign(v) if v != 0 else prev
        out[i] = prev
    return out


def stats(F, r, rf, F_before):
    R = positions_to_returns(F, r, rf, F_before)
    eq = equity_curve(R)
    return dict(final_equity=eq[-1], sharpe=excess_sharpe(R, rf), max_drawdown=max_drawdown(eq),
                trade_freq_pct=100 * trading_frequency(np.concatenate(([F_before], F)))), R


def _job(args):
    kind, seed, df, years, q_method = args
    if kind == "rrl":
        return run_rrl_trader(seed, df=df, years=years)
    return run_q_trader(seed, df=df, years=years, method=q_method)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0, help="base seed; trader i uses seed+i")
    ap.add_argument("--n-rrl", type=int, default=N_RRL)
    ap.add_argument("--n-q", type=int, default=N_Q)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--q-method", choices=["q", "au"], default="q",
                    help="value-function baseline: plain Q-learning or Advantage Updating")
    ap.add_argument("--test-start", type=int, default=TEST_YEARS[0])
    ap.add_argument("--test-end", type=int, default=TEST_YEARS[-1])
    ap.add_argument("--results-dir", default=None,
                    help="default: results/ for the paper's setup, else results/exp2_<period>_<method>")
    args = ap.parse_args()
    years = range(args.test_start, args.test_end + 1)
    paper_period = (args.test_start, args.test_end) == (1970, 1994)
    RESULTS = args.results_dir or (
        RESULTS_DIR if paper_period and args.q_method == "q"
        else os.path.join(RESULTS_DIR, f"exp2_{args.test_start}-{args.test_end}_{args.q_method}"))
    os.makedirs(RESULTS, exist_ok=True)
    q_name = {"q": "Q", "au": "AU"}[args.q_method]
    print(f"seed={args.seed} n_rrl={args.n_rrl} n_q={args.n_q} rho={RRL_RHO} "
          f"q_method={args.q_method} test={args.test_start}-{args.test_end} [{DATA_LABEL}]")

    df = build_dataset(RAW_DIR, f"{args.test_start - 20}-01", f"{args.test_end}-12")
    y0, y1 = year_rows(df, years[0])[0], year_rows(df, years[-1])[1]
    test = df.iloc[y0:y1]
    r, rf = test.r.to_numpy(), test.rf.to_numpy()

    t = time.time()
    jobs = [("rrl", args.seed + i, df, years, None) for i in range(args.n_rrl)] + \
           [("q", args.seed + i, df, years, args.q_method) for i in range(args.n_q)]
    with Pool(args.workers) as pool:
        Fs = pool.map(_job, jobs)
    F_rrl, F_q = np.array(Fs[:args.n_rrl]), np.array(Fs[args.n_rrl:])
    print(f"training took {time.time() - t:.0f}s")

    # Individual traders start flat (F = 0) at the start of 1970, as do committees.
    rows = []
    for kind, Fm in [("RRL", F_rrl), ("Q", F_q)]:
        for i, F in enumerate(Fm):
            s, _ = stats(F, r, rf, 0.0)
            rows.append(dict(method=kind, trader=i, seed=args.seed + i, **s))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "exp2_traders.csv"), index=False)

    series = {"Buy and hold": np.ones(len(r)), "RRL committee": committee_vote(F_rrl),
              "Q committee": committee_vote(F_q)}
    summary, returns = {}, {}
    for name, F in series.items():
        F_before = 1.0 if name == "Buy and hold" else 0.0
        summary[name], returns[name] = stats(F, r, rf, F_before)
    ind = pd.DataFrame(rows)
    for kind in ["RRL", "Q"]:
        g = ind[ind.method == kind]
        summary[f"{kind} individual traders"] = {
            c: dict(median=g[c].median(), q25=g[c].quantile(0.25), q75=g[c].quantile(0.75))
            for c in ["final_equity", "sharpe", "max_drawdown", "trade_freq_pct"]}
    if paper_period:   # the paper's Q-trader used Advantage Updating
        summary["paper"] = {"Buy and hold": dict(final_equity=13.48, sharpe=0.34),
                            "Q committee": dict(final_equity=33.59, sharpe=0.63),
                            "RRL committee": dict(final_equity=58.60, sharpe=0.83)}
    summary["config"] = dict(seed=args.seed, n_rrl=args.n_rrl, n_q=args.n_q, rho=RRL_RHO,
                             q_method=args.q_method,
                             q=AU_SETTINGS if args.q_method == "au" else Q_SETTINGS,
                             test_period=f"{args.test_start}-{args.test_end}",
                             eta=ETA, weight_decay=WEIGHT_DECAY, delta=DELTA,
                             data=DATA_LABEL)
    with open(os.path.join(RESULTS, "exp2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    np.savez(os.path.join(RESULTS, "exp2_positions.npz"), F_rrl=F_rrl, F_q=F_q,
             months=np.array(test.index.astype(str), dtype="U7"))
    print(json.dumps({k: summary[k] for k in series}, indent=2, default=float))

    from experiments.exp2_plots import plot_all
    plot_all(test, series, returns, ind, F_rrl, F_q, RESULTS, DATA_LABEL, q_name=q_name)


if __name__ == "__main__":
    main()
