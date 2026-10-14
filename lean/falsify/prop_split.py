"""Lemma `sandwich` (with the logit-gradient identity) and Proposition `split` on random logits,
including near-one-hot and near-uniform distributions."""
import numpy as np
from _common import Counter, finish, softmax

rng = np.random.default_rng(4)
sand, split = Counter("lem_sandwich"), Counter("prop_split")
nll = lambda y, z: -np.log(softmax(z)[y])
gnorm = lambda y, z: np.linalg.norm(softmax(z) - np.eye(len(z))[y])
for _ in range(4000):
    V = int(rng.integers(2, 40)); z = rng.normal(0, float(rng.choice([0.1, 1, 5, 15])), V); y = int(rng.integers(V))
    ce = nll(y, z); g = gnorm(y, z)
    sand.check(1 - np.exp(-ce) <= g + 1e-12 and g <= np.sqrt(2) * (1 - np.exp(-ce)) + 1e-12, (ce, g))
    v = int(rng.integers(V)); h = 1e-6
    zp, zm = z.copy(), z.copy(); zp[v] += h; zm[v] -= h
    num = (nll(y, zp) - nll(y, zm)) / (2 * h)
    sand.check(abs(num - (softmax(z)[v] - (v == y))) < 1e-6, "grad")
    # Proposition split
    nr, na = int(rng.integers(1, 30)), int(rng.integers(1, 10))
    Zr = [rng.normal(0, 3, V) for _ in range(nr)]; yr = rng.integers(0, V, nr)
    Za = [rng.normal(0, 1, V) + 4 * np.eye(V)[k] for k in rng.integers(0, V, na)]; ya = rng.integers(0, V, na)
    CEr = np.array([nll(int(yy), zz) for zz, yy in zip(Zr, yr)]); CEa = np.array([nll(int(yy), zz) for zz, yy in zip(Za, ya)])
    share = (CEa.sum() / (nr + na)) / ((CEa.sum() + CEr.sum()) / (nr + na))
    ppl = (na * CEa.mean()) / (na * CEa.mean() + nr * CEr.mean())
    split.check(abs(share - ppl) < 1e-12, "loss share")
    Lam = CEr.max() * float(rng.choice([1.0, 1.5, 10.0]))
    Ga = sum(gnorm(int(yy), zz) for zz, yy in zip(Za, ya)); Gr = sum(gnorm(int(yy), zz) for zz, yy in zip(Zr, yr))
    X = np.sqrt(2) * CEa.sum(); Y = (1 - np.exp(-Lam)) / Lam * CEr.sum()
    split.check(Ga / (Ga + Gr) <= X / (X + Y) + 1e-12, ("grad share", Ga, Gr, X, Y))
finish([sand, split])
