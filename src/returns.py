"""Trading returns R_t and their partial derivatives w.r.t. the positions.
"""
import numpy as np


def additive_return(F_prev, F, r, delta, mu=1.0):
    """Additive profits.
    """
    s = np.sign(F - F_prev)
    R = mu * (F_prev * r - delta * abs(F - F_prev))
    dR_dF = -mu * delta * s
    dR_dFprev = mu * r + mu * delta * s
    return R, dR_dF, dR_dFprev


def multiplicative_return(F_prev, F, r, rf, delta):
    """Multiplicative profits.
    """
    s = np.sign(F - F_prev)
    gross = 1.0 + (1.0 - F_prev) * rf + F_prev * r
    cost = 1.0 - delta * abs(F - F_prev)
    R = gross * cost - 1.0
    dR_dF = gross * (-delta * s)
    dR_dFprev = (r - rf) * cost + gross * delta * s
    return R, dR_dF, dR_dFprev
