"""PyTorch autograd reference for the RRL trader. Used ONLY in tests to verify
the hand-derived gradients in src/rrl.py via backpropagation through time."""
import torch


def forward(theta, X, r, *, t_start, t_end, delta, mu=1.0, rf=None,
            discrete=False, eta=None, objective="sharpe"):
    """Re-implements the trader forward pass with autograd.

    Returns the list of R_t tensors, and (if eta is given) the list of D_t with
    A_{t-1}, B_{t-1} detached, mirroring how the online update treats them.

    For the discrete trader a straight-through estimator is used: the forward
    value is sign(y_t) but the gradient flows through y_t, which is exactly the
    surrogate gradient used by src/rrl.py.
    """
    F_prev = torch.zeros((), dtype=torch.float64)
    Rs, Ds = [], []
    A = B = torch.zeros((), dtype=torch.float64)
    for t in range(t_start, t_end):
        inp = torch.cat([torch.ones(1, dtype=torch.float64),
                         torch.as_tensor(X[t], dtype=torch.float64),
                         F_prev.reshape(1)])
        y = torch.tanh(theta @ inp)
        if discrete == "ternary":
            hard = torch.where(y > 0.5, 1.0, torch.where(y < -0.5, -1.0, 0.0)).to(torch.float64)
            F = y + (hard - y).detach()
        elif discrete:
            hard = torch.where(y >= 0, 1.0, -1.0).to(torch.float64)
            F = y + (hard - y).detach()
        else:
            F = y
        d = delta[t] if hasattr(delta, "__len__") else delta
        if rf is None:
            R = mu * (F_prev * r[t] - d * torch.abs(F - F_prev))
        else:
            gross = 1.0 + (1.0 - F_prev) * rf[t] + F_prev * r[t]
            R = gross * (1.0 - d * torch.abs(F - F_prev)) - 1.0
        Rs.append(R)
        if eta is not None and objective == "ddr":
            # differential downside deviation ratio; B holds DD^2 (Eqs. 21-24)
            Ad, Bd = A.detach(), B.detach()
            if Bd > 1e-12:
                DD = torch.sqrt(Bd)
                Ds.append(torch.where(R > 0, (R - 0.5 * Ad) / DD,
                                      (Bd * (R - 0.5 * Ad) - 0.5 * Ad * R**2) / DD**3))
            else:
                Ds.append(torch.zeros((), dtype=torch.float64))   # keeps Ds aligned with t
            A = Ad + eta * (R - Ad)
            B = Bd + eta * (torch.clamp(R, max=0.0) ** 2 - Bd)
        elif eta is not None:
            Ad, Bd = A.detach(), B.detach()
            var = Bd - Ad**2
            if var > 1e-12:
                Ds.append((Bd * (R - Ad) - 0.5 * Ad * (R**2 - Bd)) / var**1.5)
            A = Ad + eta * (R - Ad)
            B = Bd + eta * (R**2 - Bd)
        F_prev = F
    return Rs, Ds


def sharpe(Rs):
    R = torch.stack(Rs)
    A, B = R.mean(), (R**2).mean()
    return A / torch.sqrt(B - A**2)
