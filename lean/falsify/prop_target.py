"""Proposition `target`, Eq. `kl-grad`, Corollaries `beta` and `budget` on random instances.

Teeth: with a < 0 and b = 0 (dropping a > 0), p* = 0 is not a minimizer over (0, 1]."""
import numpy as np
from _common import Counter, finish, softmax, TOL

rng = np.random.default_rng(2)
L = lambda a, b, q, p: -a * np.log(p) + b * (q / p + np.log(p / q) - 1)
c = lambda a, b, q, p: a - b * (1 - q / p)
def pstar(a, b, q):
    return min(1.0, q * b / (b - a)) if a < b else 1.0

tgt, grad, beta, budget, teeth = (Counter(n) for n in
    ("prop_target", "eq_kl_grad", "cor_beta", "cor_budget", "prop_target without a > 0"))
grid = np.concatenate([np.geomspace(1e-6, 1, 4000), np.linspace(0.001, 1, 4000)])
for _ in range(3000):
    a = float(rng.choice([rng.uniform(1e-4, 0.05), rng.uniform(0.05, 2)]))
    b = float(rng.choice([0.0, rng.uniform(0, 0.1), rng.uniform(0, 5)]))
    q = float(rng.choice([rng.uniform(1e-4, 0.05), rng.uniform(0.05, 1.0), 1.0]))
    ps = pstar(a, b, q)
    tgt.check(0 < ps <= 1, (a, b, q, ps))
    tgt.check(L(a, b, q, ps) <= np.min(L(a, b, q, grid)) + 1e-9 * (1 + abs(L(a, b, q, ps))), (a, b, q, ps))
    # Eq. kl-grad, scalar and coordinate forms (central differences)
    p = float(rng.uniform(1e-3, 1)); h = 1e-6 * p
    num = (L(a, b, q, p + h) - L(a, b, q, p - h)) / (2 * h)
    grad.check(abs(num - (-c(a, b, q, p) / p)) <= 1e-4 * (1 + abs(num)), ("scalar", a, b, q, p))
    V = int(rng.integers(2, 12)); z = rng.normal(0, 2, V); y = int(rng.integers(V)); v = int(rng.integers(V))
    f = lambda t: L(a, b, q, softmax(np.where(np.arange(V) == v, t, z))[y])
    num = (f(z[v] + 1e-6) - f(z[v] - 1e-6)) / 2e-6
    P = softmax(z)
    grad.check(abs(num - c(a, b, q, P[y]) * (P[v] - (v == y))) <= 1e-5 * (1 + abs(num)), ("coord", V))
    grad.check(abs(c(a, b, q, q) - a) <= TOL, "init")
    if b > 0:
        p1, p2 = sorted(rng.uniform(1e-3, 1, 2))
        grad.check(c(a, b, q, p1) > c(a, b, q, p2) or p1 == p2, "anti")
        if p1 < q:
            grad.check(c(a, b, q, p1) > a, "below ref")
    if 0 < a < b:
        pst = q * b / (b - a)
        grad.check(abs(c(a, b, q, pst)) <= 1e-9 * (1 + b * q / pst), "zero at p*")
        for p in rng.uniform(1e-3 * pst, 3 * pst, 20):
            grad.check((c(a, b, q, p) > 0) == (p < pst) or abs(p - pst) < 1e-12, ("sign", p, pst))
    # Corollary beta
    w, W, g, G = rng.uniform(0.1, 1), rng.uniform(1, 50), rng.uniform(1e-6, 1), rng.uniform(1, 50)
    bb = (w / g) * (G / W); be = float(rng.uniform(0, 3 * bb)); qq = float(rng.uniform(1e-3, 0.999))
    beta.check((w / W < be * g / G) == (bb < be), ("exists", bb, be))
    beta.check((pstar(w / W, be * g / G, qq) < 1) == (bb / (1 - qq) < be), ("binds", bb, be, qq))
    # Corollary budget
    if 0 < a < b:
        lhs = np.log(pstar(a, b, q) / q); rhs = -np.log(1 - a / b)
        budget.check(lhs <= rhs + 1e-12, ("cap", a, b, q))
        if q * b / (b - a) <= 1:
            budget.check(abs(lhs - rhs) <= 1e-9 * (1 + abs(rhs)), ("equality", a, b, q))
    if b <= a:
        budget.check(abs(np.log(pstar(a, b, q) / q) + np.log(q)) <= 1e-12, "uncapped")
    budget.check(abs((-np.log(q)) - (-np.log(ps)) - np.log(ps / q)) <= 1e-9, "shed")
    b1, b2, B = sorted(rng.uniform(0, 1, 3))
    budget.check(-np.log(1 - b1 / B) <= -np.log(1 - b2 / B) + 1e-12, "mono betabar")
    bt, B1, B2 = sorted(rng.uniform(0, 1, 3))
    budget.check(-np.log(1 - bt / B2) <= -np.log(1 - bt / B1) + 1e-12, "anti beta")
# teeth: drop a > 0
for _ in range(200):
    a, b, q = -rng.uniform(0.01, 1), 0.0, rng.uniform(0.01, 1)
    ps = pstar(a, b, q)
    teeth.check(0 < ps <= 1 and L(a, b, q, ps) <= np.min(L(a, b, q, grid)), (a, ps))
finish([tgt, grad, beta, budget], [teeth])
