"""Trading returns R_t and their partial derivatives w.r.t. the positions.

Timing convention (paper Sec. II-A): the position F_{t-1} is held over the
interval (t-1, t], earning the price return r_t; at time t the trader switches
to F_t and pays the transaction cost delta * |F_t - F_{t-1}|.

Each function returns (R_t, dR_t/dF_t, dR_t/dF_{t-1}), which is all the RRL
gradient needs (paper Eq. 25 / 28).
"""
import numpy as np


def additive_return(F_prev, F, r, delta, mu=1.0):
    """Additive profits, paper Eq. (6), with r_t = z_t - z_{t-1} (price DIFFERENCES).

    R_t = mu * (F_{t-1} * r_t - delta * |F_t - F_{t-1}|)
    """
    s = np.sign(F - F_prev)
    R = mu * (F_prev * r - delta * abs(F - F_prev))
    dR_dF = -mu * delta * s
    dR_dFprev = mu * r + mu * delta * s
    return R, dR_dF, dR_dFprev


def multiplicative_return(F_prev, F, r, rf, delta):
    """Multiplicative profits, paper Eq. (7), leverage nu = 1.

    r_t  = z_t / z_{t-1} - 1          (simple return of the risky asset)
    rf_t = risk-free return over the same period
    1 + R_t = {1 + (1 - F_{t-1}) rf_t + F_{t-1} r_t} * {1 - delta |F_t - F_{t-1}|}

    With F = +1 the trader earns r_t; with F = -1 it earns 2 rf_t - r_t, i.e. a
    short sale with 2:1 leverage earning twice the T-bill rate (Sec. IV-C).
    """
    s = np.sign(F - F_prev)
    gross = 1.0 + (1.0 - F_prev) * rf + F_prev * r
    cost = 1.0 - delta * abs(F - F_prev)
    R = gross * cost - 1.0
    dR_dF = gross * (-delta * s)
    dR_dFprev = (r - rf) * cost + gross * delta * s
    return R, dR_dF, dR_dFprev
