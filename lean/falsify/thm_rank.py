"""Theorem `rank` on random twins: the product formula by enumeration over subsets, Eq. `mu` as a
derivative of the log deployment probability in log kappa_j (central differences), the sign and
monotonicity of mu, and the rearrangement claim over all permutations."""
import itertools
import numpy as np
from _common import Counter, finish

rng = np.random.default_rng(5)
rank = Counter("thm_rank")
mu = lambda k, d: k * (np.exp(d) - 1) / (1 - k + k * np.exp(d))
for _ in range(1500):
    n = int(rng.integers(1, 8)); a0 = float(rng.normal(-2, 1))
    kap = rng.uniform(0.01, 0.99, n); D = rng.normal(0, 1.5, n)
    lhs = 0.0
    for U in itertools.product([0, 1], repeat=n):
        U = np.array(U, bool)
        lhs += np.prod(kap[U]) * np.prod(1 - kap[~U]) * np.exp(a0 + D[U].sum())
    rhs = np.exp(a0) * np.prod(1 - kap + kap * np.exp(D))
    rank.check(abs(lhs - rhs) <= 1e-10 * (1 + rhs), "product")
    j = int(rng.integers(n))
    rank.check(abs((a0 + D.sum()) - (a0 + D.sum() - D[j]) - D[j]) < 1e-12, "loo")
    def logP(u):
        k2 = kap.copy(); k2[j] = np.exp(u)
        return np.log(np.exp(a0) * np.prod(1 - k2 + k2 * np.exp(D)))
    u0, h = np.log(kap[j]), 1e-6
    num = (logP(u0 + h) - logP(u0 - h)) / (2 * h)
    rank.check(abs(num - mu(kap[j], D[j])) < 1e-6, ("dmu", num, mu(kap[j], D[j])))
    k = float(rng.uniform(0.01, 0.99)); d1, d2 = sorted(rng.normal(0, 2, 2))
    rank.check(mu(k, d1) < mu(k, d2) or d1 == d2, "strict mono")
    rank.check(abs(mu(k, 0.0)) < 1e-15 and (mu(k, -abs(d1) - 1e-3) < 0), "sign")
    m = int(rng.integers(1, 7)); Dm = np.round(rng.normal(0, 1, m), 1); v = np.sort(rng.uniform(0, 1, m))
    order = np.argsort(Dm, kind="stable"); vass = np.empty(m); vass[order] = v  # weights in the order of Delta
    best = (mu(k, Dm) * vass).sum()
    for perm in itertools.permutations(range(m)):
        rank.check((mu(k, Dm) * vass[list(perm)]).sum() <= best + 1e-12, "rearrangement")
finish([rank])
