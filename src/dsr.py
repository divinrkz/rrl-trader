"""Differential Sharpe ratio (paper Eqs. 14-16).

A_t and B_t are exponential moving estimates of the first and second moments
of the trading returns. D_t is the first-order effect of R_t on the moving
Sharpe ratio. A_{t-1}, B_{t-1} are treated as constants when differentiating.
"""
import numpy as np

# Below this variance estimate D_t is considered undefined (start-up guard).
MIN_VARIANCE = 1e-12


def differential_sharpe(R, A_prev, B_prev):
    """Return (D_t, dD_t/dR_t). Both are 0 while the variance estimate is degenerate."""
    var = B_prev - A_prev**2
    if var <= MIN_VARIANCE:
        return 0.0, 0.0
    denom = var**1.5
    dA = R - A_prev
    dB = R**2 - B_prev
    D = (B_prev * dA - 0.5 * A_prev * dB) / denom
    dD_dR = (B_prev - A_prev * R) / denom
    return D, dD_dR


def update_moments(R, A_prev, B_prev, eta):
    """Eq. (15): A_t = A_{t-1} + eta (R_t - A_{t-1}),  B_t = B_{t-1} + eta (R_t^2 - B_{t-1})."""
    A = A_prev + eta * (R - A_prev)
    B = B_prev + eta * (R**2 - B_prev)
    return A, B


def ema_sharpe(R, eta=0.01):
    """Moving-average Sharpe ratio A_t / sqrt(B_t - A_t^2), as plotted in Figs. 2-3."""
    A = B = 0.0
    out = np.full(len(R), np.nan)
    for t, Rt in enumerate(R):
        A, B = update_moments(Rt, A, B, eta)
        var = B - A**2
        if var > MIN_VARIANCE:
            out[t] = A / np.sqrt(var)
    return out


# ---------------------------------------------------------------------------
# Differential Downside Deviation Ratio (paper Eqs. 19-24), used in Exp. 3 (FX).
#
# Moving estimates (Eq. 21):
#   A_t    = A_{t-1}    + eta (R_t - A_{t-1})
#   DD2_t  = DD2_{t-1}  + eta (min(R_t, 0)^2 - DD2_{t-1})
# DDR_t = A_t / DD_t. Its first-order effect in eta (Eqs. 22-24):
#   D_t = (R_t - A/2) / DD                              if R_t > 0
#   D_t = (DD^2 (R_t - A/2) - (A/2) R_t^2) / DD^3       if R_t <= 0
# with A = A_{t-1}, DD = sqrt(DD2_{t-1}) treated as constants, so
#   dD_t/dR_t = 1 / DD                                  if R_t > 0
#   dD_t/dR_t = (DD^2 - A R_t) / DD^3                   if R_t <= 0
# Gains are not penalized at all; only losses count as risk.
# ---------------------------------------------------------------------------

def differential_ddr(R, A_prev, DD2_prev):
    """Return (D_t, dD_t/dR_t). Both are 0 while the downside estimate is degenerate."""
    if DD2_prev <= MIN_VARIANCE:
        return 0.0, 0.0
    DD = np.sqrt(DD2_prev)
    if R > 0:
        return (R - 0.5 * A_prev) / DD, 1.0 / DD
    D = (DD2_prev * (R - 0.5 * A_prev) - 0.5 * A_prev * R**2) / DD**3
    return D, (DD2_prev - A_prev * R) / DD**3


def update_ddr_moments(R, A_prev, DD2_prev, eta):
    """Eq. (21)."""
    A = A_prev + eta * (R - A_prev)
    DD2 = DD2_prev + eta * (min(R, 0.0) ** 2 - DD2_prev)
    return A, DD2
