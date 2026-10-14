"""Lemma `imit`, Theorem `bound-formal` (both parts), Corollary `regimes` and Theorem `bound` on
random finite models: tasks with random probabilities, a small response alphabet, budget T, random
policies at every history, rewards in [0,1] on complete response sequences, logged seed rollouts
(reward 1 on solved tasks), random localizations and late flags.

Teeth: Lemma imit with rewards above 1, and Theorem bound-formal (1) without
`solved -> R(logged) = 1`, each produce counterexamples."""
import itertools
import numpy as np
from _common import Counter, finish

rng = np.random.default_rng(6)
imit, bform, regimes, bound = (Counter(n) for n in ("lem_imit", "thm_bound_formal", "cor_regimes", "thm_bound"))
teeth_imit, teeth_bound = Counter("lem_imit with R > 1"), Counter("thm_bound_formal (1) without solved -> R(logged)=1")


def random_policy(K):
    cache = {}
    def pi(h):
        if h not in cache:
            p = rng.dirichlet(np.full(K, float(rng.choice([0.2, 1.0, 5.0]))))
            if rng.random() < 0.2:  # point masses too
                p = np.eye(K)[rng.integers(K)]
            cache[h] = p
        return cache[h]
    return pi


def expect_at(pi, t, f, h):
    if t == 0:
        return f(h)
    return sum(pi(h)[y] * expect_at(pi, t - 1, f, h + (y,)) for y in range(len(pi(h))))


tv = lambda p, q: 0.5 * np.abs(p - q).sum()


def model(rewards_above_one=False):
    nO, K, T = int(rng.integers(1, 4)), int(rng.integers(2, 4)), int(rng.integers(1, 4))
    P = rng.dirichlet(np.ones(nO))
    table = [dict() for _ in range(nO)]
    hi = 3.0 if rewards_above_one else 1.0
    def R(o):
        def r(h):
            if h not in table[o]:
                table[o][h] = float(rng.choice([0.0, 1.0, rng.uniform(0, hi)]))
            return table[o][h]
        return r
    return nO, K, T, P, [R(o) for o in range(nO)], table


for trial in range(1500):
    # ---- Lemma imit
    nO, K, T, P, R, _ = model()
    pis, pts = [random_policy(K) for _ in range(nO)], [random_policy(K) for _ in range(nO)]
    rho = lambda pol: sum(P[o] * expect_at(pol[o], T, R[o], ()) for o in range(nO))
    rhs = sum(sum(P[o] * expect_at(pts[o], t, lambda h, o=o: tv(pis[o](h), pts[o](h)), ()) for o in range(nO))
              for t in range(T))
    imit.check(rho(pts) - rho(pis) <= rhs + 1e-12, (rho(pts), rho(pis), rhs))
    if trial < 300:
        nO, K, T, P, Rb, _ = model(rewards_above_one=True)
        pis, pts = [random_policy(K) for _ in range(nO)], [random_policy(K) for _ in range(nO)]
        rho_b = lambda pol: sum(P[o] * expect_at(pol[o], T, Rb[o], ()) for o in range(nO))
        rhs = sum(sum(P[o] * expect_at(pts[o], t, lambda h, o=o: tv(pis[o](h), pts[o](h)), ()) for o in range(nO))
                  for t in range(T))
        teeth_imit.check(rho_b(pts) - rho_b(pis) <= rhs + 1e-12, "R>1")

    # ---- Theorem bound-formal / bound, Corollary regimes
    for drop_solved_hyp in (False, True):
        nO, K, T, P, R, table = model()
        logged = [tuple(int(x) for x in rng.integers(0, K, T)) for _ in range(nO)]
        solved = [bool(rng.random() < 0.4) for _ in range(nO)]
        late = [bool(rng.random() < 0.4) for _ in range(nO)]
        tl = [int(rng.integers(0, T + 2)) for _ in range(nO)]  # includes out-of-range t' (0, T+1)
        for o in range(nO):
            if solved[o]:
                table[o][logged[o]] = 0.0 if drop_solved_hyp else 1.0
        piT = [random_policy(K) for _ in range(nO)]
        prel = []
        for o in range(nO):
            def rel(h, o=o):
                if solved[o] or len(h) + 1 < tl[o]:
                    y = logged[o][len(h)] if len(h) < T else 0
                    return np.eye(K)[y]
                return piT[o](h)
            prel.append(rel)
        piWD = prel if drop_solved_hyp else [random_policy(K) for _ in range(nO)]
        rho = lambda pol: sum(P[o] * expect_at(pol[o], T, R[o], ()) for o in range(nO))
        rhoBC = sum(P[o] for o in range(nO) if solved[o])
        pf = sum(P[o] for o in range(nO) if not solved[o])
        eloc = (sum(P[o] for o in range(nO) if late[o] and not solved[o]) / pf) if pf > 0 else 0.0
        m = lambda o: max(tl[o] - 1, 0)
        f = [1 - expect_at(piT[o], max(T - m(o), 0), R[o], logged[o][:m(o)]) for o in range(nO)]
        pe = sum(P[o] for o in range(nO) if not solved[o] and not late[o])
        ecorr = (sum(P[o] * f[o] for o in range(nO) if not solved[o] and not late[o]) / pe) if pe > 0 else 0.0
        e = lambda o, t: expect_at(prel[o], t, lambda h, o=o: tv(piWD[o](h), prel[o](h)), ())
        E = [[e(o, t) for t in range(T)] for o in range(nO)]
        delta = sum(P[o] * E[o][t] for o in range(nO) for t in range(T))
        lhs1, rhs1 = 1 - rho(piWD), (1 - rhoBC) * (eloc + ecorr) + delta
        if drop_solved_hyp:
            if any(solved):
                teeth_bound.check(lhs1 <= rhs1 + 1e-12, "dropped")
            continue
        bform.check(lhs1 <= rhs1 + 1e-12, ("(1)", lhs1, rhs1))
        # part (2): make late prefixes dead, recompute
        for o in range(nO):
            if late[o] and not solved[o]:
                pre = logged[o][:m(o)]
                for c in itertools.product(range(K), repeat=max(T - m(o), 0)):
                    table[o][pre + c] = 0.0
        f = [1 - expect_at(piT[o], max(T - m(o), 0), R[o], logged[o][:m(o)]) for o in range(nO)]
        ecorr = (sum(P[o] * f[o] for o in range(nO) if not solved[o] and not late[o]) / pe) if pe > 0 else 0.0
        bform.check(abs((1 - rho(prel)) - (1 - rhoBC) * (eloc + (1 - eloc) * ecorr)) < 1e-10, "(2)")

        def cell_avg(cell):
            num = sum(P[o] * E[o][t] for o in range(nO) for t in range(T) if cell(o, t))
            den = sum(P[o] for o in range(nO) for t in range(T) if cell(o, t))
            return num / den if den != 0 else 0.0
        eret = cell_avg(lambda o, t: solved[o])
        efit = cell_avg(lambda o, t: not solved[o] and t < tl[o])
        erec = cell_avg(lambda o, t: not solved[o] and tl[o] <= t)
        regimes.check(delta <= T * (rhoBC * eret + (1 - rhoBC) * (efit + erec)) + 1e-12, "regimes")
        mass = lambda cell: sum(P[o] for o in range(nO) for t in range(T) if cell(o, t))
        M1 = mass(lambda o, t: solved[o]); M2 = mass(lambda o, t: not solved[o] and t < tl[o])
        M3 = mass(lambda o, t: not solved[o] and tl[o] <= t)
        regimes.check(abs(delta - (M1 * eret + M2 * efit + M3 * erec)) < 1e-10, "exact identity")
        regimes.check(abs(M1 - T * rhoBC) < 1e-12 and abs(M2 + M3 - T * (1 - rhoBC)) < 1e-12, "masses")
        ebar = delta / T
        bound.check(1 - rho(piWD) <= (1 - rhoBC) * (eloc + ecorr) + T * ebar + 1e-12, "thm_bound (1)")
        bound.check(ebar <= rhoBC * eret + (1 - rhoBC) * (efit + erec) + 1e-12, "thm_bound (2)")
finish([imit, bform, regimes, bound], [teeth_imit, teeth_bound])
