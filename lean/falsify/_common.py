"""Shared helpers for the numeric falsification tests (Gate A, item 8).

Each test samples random instances that satisfy a statement's hypotheses, including edge cases,
checks the conclusion with the paper's exact constants, and exits non-zero on any violation.
These tests are independent of Lean."""
import sys
import numpy as np

TOL = 1e-9


class Counter:
    def __init__(self, name):
        self.name, self.n, self.bad = name, 0, []

    def check(self, cond, info):
        self.n += 1
        if not cond:
            self.bad.append(info)

    def report(self):
        status = "PASS" if not self.bad else "FAIL"
        print(f"{self.name}: {status}  ({self.n} checks, {len(self.bad)} violations)")
        for b in self.bad[:5]:
            print("   violation:", b)
        return not self.bad


def finish(counters, teeth=()):
    """Every counter must have no violations; every teeth counter (the conclusion checked with an
    essential hypothesis dropped) must have at least one."""
    ok = all(c.report() for c in counters)
    for t in teeth:
        print(f"{t.name}: {len(t.bad)} counterexamples in {t.n} instances with the hypothesis dropped "
              f"({'teeth confirmed' if t.bad else 'NO COUNTEREXAMPLE FOUND'})")
        ok = ok and bool(t.bad)
    sys.exit(0 if ok else 1)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()
