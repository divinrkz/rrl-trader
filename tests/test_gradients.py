"""Required test 1: hand-derived gradients vs finite differences and autograd BPTT."""
import numpy as np
import pytest
import torch

from src import rrl_torch
from src.dsr import differential_sharpe
from src.rrl import TraderState, batch_gradient, run_online, trader_step


def random_problem(seed, T=60, n_inputs=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, n_inputs))
    r = 0.01 * rng.normal(size=T)
    rf = 0.003 * rng.random(T)
    theta = rng.normal(0, 0.5, size=n_inputs + 2)
    return X, r, rf, theta


def finite_diff(f, theta, h=1e-6):
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = h
        g[i] = (f(theta + e) - f(theta - e)) / (2 * h)
    return g


@pytest.mark.parametrize("multiplicative", [False, True])
def test_batch_gradient_matches_finite_differences(multiplicative):
    X, r, rf, theta = random_problem(0)
    kw = dict(t_start=0, t_end=len(r), delta=0.01, rf=rf if multiplicative else None)
    _, grad, _ = batch_gradient(theta, X, r, **kw)
    fd = finite_diff(lambda th: batch_gradient(th, X, r, **kw)[0], theta)
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("multiplicative", [False, True])
@pytest.mark.parametrize("discrete", [False, True])
def test_batch_gradient_matches_autograd_bptt(multiplicative, discrete):
    X, r, rf, theta = random_problem(1)
    rf_ = rf if multiplicative else None
    kw = dict(t_start=0, t_end=len(r), delta=0.01, rf=rf_, discrete=discrete)
    U, grad, _ = batch_gradient(theta, X, r, **kw)

    th = torch.tensor(theta, requires_grad=True)
    Rs, _ = rrl_torch.forward(th, X, r, **kw)
    U_t = rrl_torch.sharpe(Rs)
    U_t.backward()
    assert U == pytest.approx(U_t.item(), rel=1e-12)
    np.testing.assert_allclose(grad, th.grad.numpy(), rtol=1e-5, atol=1e-10)


def test_dropping_recurrent_term_gives_wrong_gradient():
    """Common pitfall: ignoring dF_{t-1}/dtheta. The naive gradient disagrees with FD."""
    X, r, _, theta = random_problem(2)
    kw = dict(t_start=0, t_end=len(r), delta=0.01)
    _, grad, _ = batch_gradient(theta, X, r, **kw)

    F_prev, Rs, dRs = 0.0, [], []
    for t in range(len(r)):    # same forward pass, but dF_{t-1}/dtheta forced to 0
        F, _, R, dR = trader_step(theta, X[t], F_prev, np.zeros_like(theta), r[t],
                                  None, 0.01, 1.0, False)
        F_prev = F
        Rs.append(R)
        dRs.append(dR)
    R, dR = np.array(Rs), np.array(dRs)
    A, B = R.mean(), (R**2).mean()
    naive = ((B - A * R) / (len(R) * (B - A**2) ** 1.5)) @ dR

    rel_err = np.linalg.norm(naive - grad) / np.linalg.norm(grad)
    assert rel_err > 0.05


@pytest.mark.parametrize("discrete", [False, True])
def test_online_gradient_matches_autograd_with_detached_moments(discrete):
    """With rho = 0, the online per-step updates dD_t/dR_t * dR_t/dtheta sum to the
    BPTT gradient of sum_t D_t where A_{t-1}, B_{t-1} are held constant."""
    X, r, _, theta = random_problem(3, T=80)
    eta, warmup, delta = 0.05, 10, 0.01
    state = TraderState(len(theta))
    _, _, hist = run_online(theta, X, r, state, t_start=0, t_end=len(r), delta=delta,
                            rho=0.0, eta=eta, discrete=discrete, warmup=warmup,
                            record_grad=True)
    online_grad = hist["grad"].sum(axis=0)

    th = torch.tensor(theta, requires_grad=True)
    _, Ds = rrl_torch.forward(th, X, r, t_start=0, t_end=len(r), delta=delta,
                              discrete=discrete, eta=eta)
    # run_online starts updating at step index `warmup`; torch only appends D once
    # the variance is non-degenerate, which here is from step 1 on.
    assert len(Ds) == len(r) - 1
    torch.stack(Ds[warmup - 1:]).sum().backward()
    np.testing.assert_allclose(online_grad, th.grad.numpy(), rtol=1e-5, atol=1e-10)


def test_dsr_derivative_and_definition():
    A, B, R = 0.01, 0.0004, 0.03
    D, dD = differential_sharpe(R, A, B)
    h = 1e-7
    fd = (differential_sharpe(R + h, A, B)[0] - differential_sharpe(R - h, A, B)[0]) / (2 * h)
    assert dD == pytest.approx(fd, rel=1e-6)
    # Eq. (13)-(14): D_t = dS_t/deta at eta = 0, where S = A / sqrt(B - A^2).
    eta = 1e-7
    S_prev = A / np.sqrt(B - A**2)
    A1, B1 = A + eta * (R - A), B + eta * (R**2 - B)
    S_new = A1 / np.sqrt(B1 - A1**2)
    assert (S_new - S_prev) / eta == pytest.approx(D, rel=1e-5)


# ---------------------------------------------------------------- Exp. 3 additions

def test_ddr_derivative_and_definition():
    """dD/dR matches finite differences, and D_t = dDDR_t/deta at eta = 0 (Eq. 22),
    for both a gain and a loss."""
    from src.dsr import differential_ddr
    A, DD2 = 0.002, 0.0001
    for R in (0.03, -0.02):
        D, dD = differential_ddr(R, A, DD2)
        h = 1e-7
        fd = (differential_ddr(R + h, A, DD2)[0] - differential_ddr(R - h, A, DD2)[0]) / (2 * h)
        assert dD == pytest.approx(fd, rel=1e-6)
        eta = 1e-7
        A1, DD2_1 = A + eta * (R - A), DD2 + eta * (min(R, 0) ** 2 - DD2)
        assert (A1 / np.sqrt(DD2_1) - A / np.sqrt(DD2)) / eta == pytest.approx(D, rel=1e-5)


def test_ternary_batch_gradient_matches_autograd_with_varying_cost():
    """Three-position trader with a per-step cost (bid/ask spread): hand-derived
    straight-through gradient == autograd."""
    X, r, _, theta = random_problem(4)
    delta = 0.002 + 0.003 * np.random.default_rng(4).random(len(r))
    kw = dict(t_start=0, t_end=len(r), delta=delta, discrete="ternary")
    U, grad, _ = batch_gradient(theta, X, r, **kw)
    th = torch.tensor(theta, requires_grad=True)
    Rs, _ = rrl_torch.forward(th, X, r, **kw)
    rrl_torch.sharpe(Rs).backward()
    np.testing.assert_allclose(grad, th.grad.numpy(), rtol=1e-5, atol=1e-10)


def test_ternary_trader_uses_all_three_positions():
    from src.rrl import position
    assert [position(y, "ternary") for y in (-0.9, -0.2, 0.0, 0.3, 0.8)] == [-1, 0, 0, 0, 1]


@pytest.mark.parametrize("discrete", [True, "ternary"])
def test_online_ddr_gradient_matches_autograd(discrete):
    """Online DDR updates sum to the BPTT gradient of sum_t D_t with A, DD^2 detached."""
    X, r, _, theta = random_problem(5, T=80)
    delta = 0.001 + 0.002 * np.random.default_rng(5).random(len(r))
    eta, warmup = 0.05, 10
    _, _, hist = run_online(theta, X, r, TraderState(len(theta)), t_start=0, t_end=len(r),
                            delta=delta, rho=0.0, eta=eta, discrete=discrete, warmup=warmup,
                            record_grad=True, objective="ddr")
    th = torch.tensor(theta, requires_grad=True)
    _, Ds = rrl_torch.forward(th, X, r, t_start=0, t_end=len(r), delta=delta,
                              discrete=discrete, eta=eta, objective="ddr")
    assert len(Ds) == len(r)
    torch.stack(Ds[warmup:]).sum().backward()
    np.testing.assert_allclose(hist["grad"].sum(axis=0), th.grad.numpy(), rtol=1e-5, atol=1e-10)
