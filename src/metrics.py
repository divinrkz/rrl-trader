import numpy as np


def sharpe(R):
    """Per-period Sharpe ratio mean(R) / std(R)."""
    R = np.asarray(R)
    return R.mean() / R.std()


def annualized_sharpe(R, periods_per_year=12):
    return np.sqrt(periods_per_year) * sharpe(R)


def trading_frequency(F):
    """Fraction of periods in which the position changes.
    """
    F = np.asarray(F)
    return np.mean(F[1:] != F[:-1])


def equity_curve(R):
    """Multiplicative wealth
    """
    return np.cumprod(1.0 + np.asarray(R))


def max_drawdown(equity):
    """Largest peak-to-trough loss, as a fraction of the peak."""
    equity = np.asarray(equity)
    peak = np.maximum.accumulate(equity)
    return np.max(1.0 - equity / peak)
