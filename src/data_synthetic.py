"""Artificial price series, paper Sec. IV-A.1, Eqs. (42)-(44).

    p(t)    = p(t-1) + beta(t-1) + k * eps(t)
    beta(t) = alpha * beta(t-1) + nu(t)
    z(t)    = exp(p(t) / R),   R = max(p) - min(p)
    eps, nu ~ N(0, 1);  alpha = 0.9, k = 3

Log prices are a random walk with an autoregressive trend: trending on short
time scales, with a high level of noise.
"""
import numpy as np


def generate_latent(n, rng, alpha=0.9, k=3.0):
    """Return the log-price p[0..n-1] and the hidden trend beta[0..n-1].

    p(0) = 0 and beta(0) = 0. The trader never sees beta; it is returned for plots.
    """
    eps = rng.standard_normal(n)
    nu = rng.standard_normal(n)
    p = np.zeros(n)
    beta = np.zeros(n)
    for t in range(1, n):
        p[t] = p[t - 1] + beta[t - 1] + k * eps[t]
        beta[t] = alpha * beta[t - 1] + nu[t]
    return p, beta


def generate_prices(n, rng, alpha=0.9, k=3.0):
    """Return prices z[0..n-1] = exp(p / R). z(0) = 1."""
    p, _ = generate_latent(n, rng, alpha, k)
    R = p.max() - p.min()
    return np.exp(p / R)


def price_differences(z):
    """r_t = z_t - z_{t-1} (additive price returns, NOT log returns). r[0] = 0."""
    r = np.zeros_like(z)
    r[1:] = np.diff(z)
    return r


def lagged_inputs(r, n_lags):
    """X[t] = [r_{t-n_lags+1}, ..., r_t]: the n_lags most recent returns known at time t.

    Rows with t < n_lags (not enough history) are zero. Uses only data up to t.
    """
    T = len(r)
    X = np.zeros((T, n_lags))
    for j in range(n_lags):           # column j holds r_{t-(n_lags-1-j)}
        lag = n_lags - 1 - j
        X[lag:, j] = r[: T - lag]
    X[:n_lags] = 0.0
    return X
