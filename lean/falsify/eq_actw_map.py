"""Eq. `sw` (uniform weights recover SFT), Remark `resample`, and Eq. `actw-map` on random steps,
with ties, single segments, and spreads inside and outside the dead zone (phi = 0.1,
tau_dz = 0.05 nats, the paper's constants, plus random ones)."""
import numpy as np
from _common import Counter, finish

rng = np.random.default_rng(3)
sw, res, amap = Counter("eq_sw"), Counter("rem_resample"), Counter("eq_actw_map")


def avg_rank(D):
    D = np.asarray(D)
    return np.array([(D < d).sum() + ((D == d).sum() + 1) / 2 for d in D])


def norm_rank(D):
    n = len(D)
    return (avg_rank(D) - 1) / (n - 1) if n > 1 else np.zeros(n)  # Lean: x/0 = 0


def rank_map(phi, tau, sig, r):
    return np.maximum(phi, np.minimum(1, phi + (1 - phi) * (0.5 + (r - 0.5) * min(1, sig / tau))))


for _ in range(5000):
    n = int(rng.integers(1, 30))
    CE = rng.exponential(1, n)
    c = float(rng.uniform(1e-3, 10))
    sw.check(abs((c * CE).sum() / (c * n) - CE.mean()) <= 1e-12 * (1 + CE.mean()), n)
    w = rng.uniform(0, 1, n) * (rng.random(n) < 0.9)
    if w.sum() > 0:
        q = w / w.sum()
        res.check((q >= 0).all() and abs(q.sum() - 1) < 1e-12 and abs((w * CE).sum() / w.sum() - (q * CE).sum()) < 1e-10, n)
    phi = 0.1 if rng.random() < 0.5 else float(rng.uniform(0, 1))
    tau = 0.05 if rng.random() < 0.5 else float(rng.uniform(1e-3, 2))
    m = int(rng.integers(1, 25))
    D = np.round(rng.normal(0, 0.1, m), int(rng.integers(1, 4)))  # rounding creates ties
    sig = D.max() - D.min()
    r = norm_rank(D)
    v = rank_map(phi, tau, sig, r)
    amap.check(((v >= phi - 1e-12) & (v <= 1 + 1e-12)).all(), "range")
    if m == 1 or sig == 0:
        amap.check(np.allclose(v, (1 + phi) / 2), ("dead zone", D))
    if m >= 2:
        amap.check(abs(v.mean() - (1 + phi) / 2) < 1e-12, ("mean", D, v))
        amap.check(all(v[i] <= v[j] + 1e-15 for i in range(m) for j in range(m) if D[i] < D[j]), ("monovary", D))
        if sig >= tau:
            i, j = int(np.argmax(D)), int(np.argmin(D))
            if (D == D[i]).sum() == 1:
                amap.check(abs(v[i] - 1) < 1e-12, ("top", D))
            if (D == D[j]).sum() == 1:
                amap.check(abs(v[j] - phi) < 1e-12, ("bottom", D))
finish([sw, res, amap])
