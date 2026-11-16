"""Performance metrics.

Annualization convention (used everywhere in this repo):
  * Synthetic data (Exp. 1): Sharpe ratios are PER PERIOD, not annualized,
    matching the scale of the paper's Figs. 3-5 (values around 0.05-0.2).
  * Monthly S&P data (Exp. 2): Sharpe = sqrt(12) * mean / std of monthly
    returns in excess of the T-bill return, as in the paper (footnote 13).
"""
import numpy as np


def sharpe(R):
    """Per-period Sharpe ratio mean(R) / std(R)."""
    R = np.asarray(R)
    return R.mean() / R.std()


def annualized_sharpe(R, periods_per_year=12):
    return np.sqrt(periods_per_year) * sharpe(R)


def trading_frequency(F):
    """Fraction of periods in which the position changes (F_t != F_{t-1})."""
    F = np.asarray(F)
    return np.mean(F[1:] != F[:-1])


def equity_curve(R):
    """Multiplicative wealth W_t / W_0 = prod (1 + R_s)."""
    return np.cumprod(1.0 + np.asarray(R))


def max_drawdown(equity):
    """Largest peak-to-trough loss, as a fraction of the peak."""
    equity = np.asarray(equity)
    peak = np.maximum.accumulate(equity)
    return np.max(1.0 - equity / peak)
