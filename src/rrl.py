from dataclasses import dataclass, field

import numpy as np

from src.dsr import differential_ddr, differential_sharpe, update_ddr_moments, update_moments
from src.returns import additive_return, multiplicative_return


@dataclass
class TraderState:
    """Everything the online trader carries from one time step to the next."""
    n_params: int
    F_prev: float = 0.0                       # F_{t-1}; F_0 = 0 (flat)
    dF_prev: np.ndarray = field(default=None)  # dF_{t-1}/dtheta (total derivative)
    A: float = 0.0                            # EMA of R_t
    B: float = 0.0                            # EMA of R_t^2
    n_seen: int = 0                           # steps seen, for the DSR warm-up

    def __post_init__(self):
        if self.dF_prev is None:
            self.dF_prev = np.zeros(self.n_params)

    def reset_position(self):
        """Start a new pass over a sequence: flat position, no gradient history.
        The moment estimates A, B are kept."""
        self.F_prev = 0.0
        self.dF_prev = np.zeros(self.n_params)


def position(y, discrete):
    """Map the tanh output y to a position. discrete: False, True (= {-1,+1}) or "ternary"."""
    if discrete == "ternary":
        return 1.0 if y > 0.5 else (-1.0 if y < -0.5 else 0.0)
    if discrete:
        return 1.0 if y >= 0 else -1.0
    return y


def _at(delta, t):
    """Transaction cost at time t: a constant, or a per-step array (bid/ask spread)."""
    return delta[t] if np.ndim(delta) else delta


def init_theta(n_inputs, rng, scale=0.1):
    """Random initial weights for [bias, inputs..., u]."""
    return rng.normal(0.0, scale, size=n_inputs + 2)


def trader_step(theta, x_t, F_prev, dF_prev, r_t, rf_t, delta, mu, discrete):
    """One forward step of the trader plus the recurrent gradient.
    """
    inp = np.empty(len(theta))
    inp[0] = 1.0
    inp[1:-1] = x_t
    inp[-1] = F_prev
    y = np.tanh(theta @ inp)
    F = position(y, discrete)

    # Total derivative, paper Eq. (27):
    #   dF_t/dtheta = dF_t/dtheta|direct + dF_t/dF_{t-1} * dF_{t-1}/dtheta
    # with dy/da = 1 - y^2 and dF_t/dF_{t-1}|direct = (1 - y^2) * u.
    dF = (1.0 - y * y) * (inp + theta[-1] * dF_prev)

    if rf_t is None:
        R, dR_dF, dR_dFprev = additive_return(F_prev, F, r_t, delta, mu)
    else:
        R, dR_dF, dR_dFprev = multiplicative_return(F_prev, F, r_t, rf_t, delta)

    # Paper Eq. (25): dR_t/dtheta = dR_t/dF_t dF_t/dtheta + dR_t/dF_{t-1} dF_{t-1}/dtheta
    dR = dR_dF * dF + dR_dFprev * dF_prev
    return F, dF, R, dR


def run_online(theta, X, r, state, *, t_start, t_end, delta, rho=0.0, eta=0.01,
               discrete=False, mu=1.0, rf=None, weight_decay=0.0, warmup=100,
               record_grad=False, objective="sharpe"):
    """Online (stochastic) RRL, paper Eqs. (28)-(31), over times t_start..t_end-1.

    X[t] is the input vector at time t, r[t] the price return over (t-1, t],
    rf[t] the risk-free return (multiplicative profits) or None (additive).
    The parameters are updated *during* the forward pass:
        theta += rho * (dD_t/dR_t * dR_t/dtheta - weight_decay * theta)
    With rho = 0 the trader just trades with fixed weights.
    For multiplicative profits the DSR is computed on excess returns R_t - rf_t.

    The first `warmup` steps only warm up the moment estimates A, B (no update).

    objective: "sharpe" (differential Sharpe ratio, A/B = first/second moment) or
    "ddr" (differential downside deviation ratio; state.B then holds DD^2).
    delta may be a per-step array (e.g. half the bid/ask spread).

    Returns (theta, state, history) where history has arrays F, R, D (and the
    per-step gradient dD_t/dtheta as "grad" if record_grad, for testing).
    """
    theta = theta.copy()
    n = t_end - t_start
    hist = {"F": np.empty(n), "R": np.empty(n), "D": np.empty(n)}
    if record_grad:
        hist["grad"] = np.zeros((n, len(theta)))
    decay_mask = np.ones_like(theta)
    decay_mask[0] = 0.0                     # no weight decay on the bias

    diff, update = {"sharpe": (differential_sharpe, update_moments),
                    "ddr": (differential_ddr, update_ddr_moments)}[objective]
    for i, t in enumerate(range(t_start, t_end)):
        rf_t = None if rf is None else rf[t]
        F, dF, R, dR = trader_step(theta, X[t], state.F_prev, state.dF_prev,
                                   r[t], rf_t, _at(delta, t), mu, discrete)
        R_obj = R if rf is None else R - rf_t
        D, dD_dR = diff(R_obj, state.A, state.B)
        state.A, state.B = update(R_obj, state.A, state.B, eta)
        state.n_seen += 1

        if state.n_seen > warmup:
            if record_grad:
                hist["grad"][i] = dD_dR * dR
            if rho > 0:
                theta += rho * (dD_dR * dR - weight_decay * decay_mask * theta)

        state.F_prev, state.dF_prev = F, dF
        hist["F"][i], hist["R"][i], hist["D"][i] = F, R, D
    return theta, state, hist


def batch_gradient(theta, X, r, *, t_start, t_end, delta, discrete=False,
                   mu=1.0, rf=None):
    """Batch RRL
    """
    F_prev, dF_prev = 0.0, np.zeros(len(theta))
    Rs, dRs = [], []
    for t in range(t_start, t_end):
        rf_t = None if rf is None else rf[t]
        F, dF, R, dR = trader_step(theta, X[t], F_prev, dF_prev, r[t], rf_t,
                                   _at(delta, t), mu, discrete)
        Rs.append(R)
        dRs.append(dR)
        F_prev, dF_prev = F, dF
    R = np.array(Rs)
    dR = np.array(dRs)
    T = len(R)
    A, B = R.mean(), (R**2).mean()
    K = B - A**2
    U = A / np.sqrt(K)
    dU_dR = (B - A * R) / (T * K**1.5)
    return U, dU_dR @ dR, R


def train_batch(theta, X, r, *, t_start, t_end, delta, rho, epochs, **kw):
    """Batch gradient ascent on the Sharpe ratio, paper Eq. (26)."""
    theta = theta.copy()
    for _ in range(epochs):
        _, grad, _ = batch_gradient(theta, X, r, t_start=t_start, t_end=t_end,
                                    delta=delta, **kw)
        theta += rho * grad
    return theta
