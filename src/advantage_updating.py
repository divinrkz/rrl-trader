import numpy as np

from src.returns import multiplicative_return

ACTIONS = np.array([-1.0, 1.0])


class MLP:
    def __init__(self, n_in, rng, n_hidden=30):
        self.W1 = rng.normal(0, 1 / np.sqrt(n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(n_hidden), size=n_hidden)
        self.b2 = 0.0

    def copy(self):
        new = MLP.__new__(MLP)
        new.W1, new.b1, new.w2, new.b2 = self.W1.copy(), self.b1.copy(), self.w2.copy(), self.b2
        return new

    def forward(self, Z):
        H = np.tanh(Z @ self.W1 + self.b1)
        return H @ self.w2 + self.b2, H

    def __call__(self, Z):
        return self.forward(Z)[0]

    def step(self, Z, err, lr, weight_decay=0.0):
        _, H = self.forward(Z)
        n = len(err)
        dH = np.outer(err, self.w2) * (1 - H**2)
        self.w2 -= lr * (H.T @ err / n + weight_decay * self.w2)
        self.b2 -= lr * err.mean()
        self.W1 -= lr * (Z.T @ dH / n + weight_decay * self.W1)
        self.b1 -= lr * dH.mean(axis=0)


class AUTrader:
    """Advantage network A(x, F_prev, a) and value network V(x, F_prev)."""

    def __init__(self, n_features, rng, n_hidden=30):
        self.A = MLP(n_features + 2, rng, n_hidden)
        self.V = MLP(n_features + 1, rng, n_hidden)

    def copy(self):
        new = AUTrader.__new__(AUTrader)
        new.A, new.V = self.A.copy(), self.V.copy()
        return new

    def advantages(self, S):
        n = len(S)
        both = np.vstack([np.column_stack([S, np.full(n, u)]) for u in ACTIONS])
        return self.A(both).reshape(2, n).T


def make_transitions(X, r, rf, times, delta):
    S, act, rew, S_next = [], [], [], []
    for t in times:
        for F_prev in ACTIONS:
            for a in ACTIONS:
                R, _, _ = multiplicative_return(a, a, r[t + 1], rf[t + 1], 0.0)
                R = (1 + R) * (1 - delta * abs(a - F_prev)) - 1   # cost of moving F_prev -> a
                S.append(np.append(X[t], F_prev))
                act.append(a)
                rew.append(R)
                S_next.append(np.append(X[t + 1], a))
    return np.array(S), np.array(act), np.array(rew), np.array(S_next)


def au_update(agent, S, a, R, S_next, gamma, alpha, beta, omega, K=1.0, weight_decay=0.0):
    A_old = agent.advantages(S)
    Amax_old = A_old.max(axis=1)
    V_s, V_next = agent.V(S), agent.V(S_next)
    # 1. advantage of the action taken
    target = Amax_old + (R + gamma * V_next - V_s) / K
    Za = np.column_stack([S, a])
    agent.A.step(Za, agent.A(Za) - target, alpha, weight_decay)
    # 2. value, driven by the change in A_max
    Amax_new = agent.advantages(S).max(axis=1)
    agent.V.step(S, -(Amax_new - Amax_old) / alpha, beta, weight_decay)
    # 3. normalization: shift both actions' advantages so that max_u A(s, u) -> 0
    A_now = agent.advantages(S)
    Amax = A_now.max(axis=1)
    for u in ACTIONS:
        Zu = np.column_stack([S, np.full(len(S), u)])
        agent.A.step(Zu, Amax, omega, 0.0)   # err = A - (A - A_max) = A_max


def run_policy(agent, X, r, rf, t_start, t_end, delta, F_prev=0.0):
    """trade greedily (argmax_a A) over t_start..t_end-1, same timing as the RRL trader."""
    T = t_end - t_start
    # Inputs don't depend on actions: precompute A for both F_prev values at once.
    A_by_prev = {Fp: agent.advantages(np.column_stack([X[t_start:t_end], np.full(T, Fp)]))
                 for Fp in (-1.0, 0.0, 1.0)}          # 0 = flat start, as in the Q-trader
    F_hist, R_hist = np.empty(T), np.empty(T)
    for i, t in enumerate(range(t_start, t_end)):
        F = ACTIONS[int(np.argmax(A_by_prev[F_prev][i]))]
        R, _, _ = multiplicative_return(F_prev, F, r[t], rf[t], delta)
        F_hist[i], R_hist[i] = F, R
        F_prev = F
    return F_hist, R_hist


def train_au(agent, X, r, rf, train_times, val_range, delta, rng, *, gammas=(0.0, 0.75),
             epochs=(10, 20), alpha=0.03, beta=0.03, omega=0.1, K=1.0, batch=32,
             weight_decay=1e-3, score=None):
    boot = rng.choice(train_times, size=len(train_times), replace=True)
    S, a, R, S_next = make_transitions(X, r, rf, boot, delta)
    R = R / R.std()
    best, best_score = agent.copy(), -np.inf
    for gamma, n_epochs in zip(gammas, epochs):
        for _ in range(n_epochs):
            order = rng.permutation(len(R))
            for k in range(0, len(R), batch):
                i = order[k:k + batch]
                au_update(agent, S[i], a[i], R[i], S_next[i], gamma, alpha, beta, omega, K,
                          weight_decay)
            _, R_val = run_policy(agent, X, r, rf, *val_range, delta)
            s = score(R_val, rf[val_range[0]:val_range[1]])
            if s > best_score:
                best, best_score = agent.copy(), s
    return best
