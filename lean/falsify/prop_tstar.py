"""Lemma `mono` and Proposition `tstar` on random small deterministic harnesses.

Recoverable(s, k): some action sequence of length <= k either ends the episode (done) or uses up
the whole budget (length k), and R holds at the state it reaches. Teeth: drop `Failed` (use a
successful trajectory) and the conclusion `nu = 0 at the end` fails."""
import itertools
import numpy as np
from _common import Counter, finish

rng = np.random.default_rng(1)
mono, tstar, teeth = Counter("lem_mono"), Counter("prop_tstar"), Counter("prop_tstar without Failed")


def make_harness():
    m, k = rng.integers(2, 7), rng.integers(1, 4)
    step = rng.integers(0, m, size=(m, k))
    done = rng.random(m) < 0.25
    for s in range(m):  # absorbing ended episodes (Assumption budget)
        if done[s]:
            step[s, :] = s
    R = rng.random(m) < 0.3
    return m, k, step, done, R


def run(step, s, seq):
    for a in seq:
        s = step[s, a]
    return s


def recoverable(step, done, R, k_actions, s, budget):
    for L in range(budget + 1):
        for seq in itertools.product(range(k_actions), repeat=L):
            e = run(step, s, seq)
            if (done[e] or L == budget) and R[e]:
                return True
    return False


for _ in range(400):
    m, k, step, done, R = make_harness()
    T = int(rng.integers(1, 5))
    s0 = int(rng.integers(0, m))
    acts = rng.integers(0, k, size=T)
    states = [s0]
    for t in range(T):
        states.append(step[states[-1], acts[t]])
    # episode length: first t with done, else T
    n = next((t for t in range(T + 1) if done[states[t]]), T)
    nu_all = [recoverable(step, done, R, k, states[t], T - t) for t in range(T + 1)]
    for t in range(T):  # Lemma mono for every t + 1 <= T
        mono.check(nu_all[t] or not nu_all[t + 1], (t, nu_all))
    nu = nu_all[:n + 1]
    failed = not R[states[n]]
    if not failed:  # teeth: a successful episode has nu = 1 at its end
        teeth.check(not nu[n], ("succeeded trajectory, nu at end", nu[n]))
        continue
    if not nu[0]:
        continue
    tstar.check(not nu[n], ("nu at end", nu))
    zeros = [t for t in range(n + 1) if not nu[t]]
    ts = zeros[0]
    tstar.check(1 <= ts <= n, ("range", ts, n))
    tstar.check(all(nu[t] for t in range(ts)) and not any(nu[t] for t in range(ts, n + 1)),
                ("monotone split", nu))
    prev = states[ts - 1]
    exists = any(recoverable(step, done, R, k, step[prev, a], T - ts) for a in range(k))
    tstar.check(exists, ("no preserving action", ts))
    tstar.check(not recoverable(step, done, R, k, step[prev, acts[ts - 1]], T - ts), ("executed preserves", ts))

finish([mono, tstar], [teeth])
