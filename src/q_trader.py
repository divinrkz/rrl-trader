import numpy as np

from src.returns import multiplicative_return

ACTIONS = (-1.0, 1.0)


class QNet:
    """Q(x, F_prev, a) = w2 . tanh(W1 [x, F_prev, a] + b1) + b2."""

    def __init__(self, n_in, rng, n_hidden=30):
        self.W1 = rng.normal(0, 1 / np.sqrt(n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(n_hidden), size=n_hidden)
        self.b2 = 0.0

    def params(self):
        return [self.W1, self.b1, self.w2, np.array(self.b2)]

    def copy(self):
        new = QNet.__new__(QNet)
        new.W1, new.b1, new.w2, new.b2 = self.W1.copy(), self.b1.copy(), self.w2.copy(), self.b2
        return new

    def predict(self, Z):
        return np.tanh(Z @ self.W1 + self.b1) @ self.w2 + self.b2

    def fit(self, Z, y, epochs, lr=0.01, weight_decay=1e-4):
        """Full-batch Adam on mean squared error."""
        m = [np.zeros_like(p) for p in self.params()]
        v = [np.zeros_like(p) for p in self.params()]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in range(1, epochs + 1):
            H = np.tanh(Z @ self.W1 + self.b1)
            err = H @ self.w2 + self.b2 - y                 # dL/dq * n / 2
            n = len(y)
            g_w2 = H.T @ err / n + weight_decay * self.w2
            g_b2 = err.mean()
            dH = np.outer(err, self.w2) * (1 - H**2)
            g_W1 = Z.T @ dH / n + weight_decay * self.W1
            g_b1 = dH.mean(axis=0)
            grads = [g_W1, g_b1, g_w2, np.array(g_b2)]
            new = []
            for i, (p, g) in enumerate(zip(self.params(), grads)):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                step = lr * (m[i] / (1 - b1**k)) / (np.sqrt(v[i] / (1 - b2**k)) + eps)
                new.append(p - step)
            self.W1, self.b1, self.w2, self.b2 = new[0], new[1], new[2], float(new[3])


def _inputs(x, F_prev, a):
    return np.concatenate([x, [F_prev, a]])


def best_action(net, x, F_prev):
    q = [net.predict(_inputs(x, F_prev, a)[None])[0] for a in ACTIONS]
    return ACTIONS[int(np.argmax(q))]


def make_transitions(X, r, rf, times, delta):
    """For each decision time t (needs t+1 for the reward): all 4 (F_prev, a) pairs.
    Returns Z (inputs), reward, and the next-state inputs for both next actions."""
    Z, rew, Z_next = [], [], []
    for t in times:
        for F_prev in ACTIONS:
            for a in ACTIONS:
                R, _, _ = multiplicative_return(a, a, r[t + 1], rf[t + 1], 0.0)
                R = (1 + R) * (1 - delta * abs(a - F_prev)) - 1   # cost of moving F_prev -> a
                Z.append(_inputs(X[t], F_prev, a))
                rew.append(R)
                Z_next.append([_inputs(X[t + 1], a, b) for b in ACTIONS])
    return np.array(Z), np.array(rew), np.array(Z_next)


def run_policy(net, X, r, rf, t_start, t_end, delta, F_prev=0.0):
    """Trade greedily over t_start..t_end-1. Same timing as the RRL trader."""
    F_hist, R_hist = np.empty(t_end - t_start), np.empty(t_end - t_start)
    for i, t in enumerate(range(t_start, t_end)):
        F = best_action(net, X[t], F_prev)
        R, _, _ = multiplicative_return(F_prev, F, r[t], rf[t], delta)
        F_hist[i], R_hist[i] = F, R
        F_prev = F
    return F_hist, R_hist


def train_q(net, X, r, rf, train_times, val_range, delta, rng, *, gammas=(0.0, 0.75),
            iterations=(1, 8), fit_epochs=300, lr=0.01, weight_decay=1e-4, score=None):
    """Fitted Q-iteration with early stopping on validation trading performance.
    """
    boot = rng.choice(train_times, size=len(train_times), replace=True)
    Z, rew, Z_next = make_transitions(X, r, rf, boot, delta)
    scale = rew.std()
    best_net, best_score = net.copy(), -np.inf
    for gamma, n_iter in zip(gammas, iterations):
        for _ in range(n_iter):
            if gamma > 0:
                q_next = net.predict(Z_next.reshape(-1, Z.shape[1])).reshape(len(Z), 2)
                target = rew / scale + gamma * q_next.max(axis=1)
            else:
                target = rew / scale
            net.fit(Z, target, fit_epochs, lr=lr, weight_decay=weight_decay)
            _, R_val = run_policy(net, X, r, rf, *val_range, delta)
            s = score(R_val, rf[val_range[0]:val_range[1]])
            if s > best_score:
                best_net, best_score = net.copy(), s
    return best_net
