import numpy as np

MIN_VARIANCE = 1e-12

def differential_sharpe(R, A_prev, B_prev):
    """Return (D_t, dD_t/dR_t)."""
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
    """A_t and B_t updates."""
    A = A_prev + eta * (R - A_prev)
    B = B_prev + eta * (R**2 - B_prev)
    return A, B


def ema_sharpe(R, eta=0.01):
    """Moving-average Sharpe ratio."""
    A = B = 0.0
    out = np.full(len(R), np.nan)
    for t, Rt in enumerate(R):
        A, B = update_moments(Rt, A, B, eta)
        var = B - A**2
        if var > MIN_VARIANCE:
            out[t] = A / np.sqrt(var)
    return out


def differential_ddr(R, A_prev, DD2_prev):
    """Return (D_t, dD_t/dR_t)."""
    if DD2_prev <= MIN_VARIANCE:
        return 0.0, 0.0
    DD = np.sqrt(DD2_prev)
    if R > 0:
        return (R - 0.5 * A_prev) / DD, 1.0 / DD
    D = (DD2_prev * (R - 0.5 * A_prev) - 0.5 * A_prev * R**2) / DD**3
    return D, (DD2_prev - A_prev * R) / DD**3


def update_ddr_moments(R, A_prev, DD2_prev, eta):
    A = A_prev + eta * (R - A_prev)
    DD2 = DD2_prev + eta * (min(R, 0.0) ** 2 - DD2_prev)
    return A, DD2
