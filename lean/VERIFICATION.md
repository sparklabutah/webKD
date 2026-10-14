# Verification log: WebKD theory

Frozen (statements approved by the author, 2026-10-14):

```
4a129d08a3c4679667491a284e938799a8bfcf134b59aa2bb9f9c50cfd2f6270  WebkdLean/Defs.lean
388c30beb85bf8df0edb5e50dd8077314088ab6ba6c0b5a208b0f5c84f1303c7  WebkdLean/Statements.lean
```

Each result passes four gates: (A) the Lean statement says what the paper says, (B) the Lean
proof is the paper's proof, (C) a reader can follow it, (D) only the kernel and the three standard
axioms are trusted.

## Gate A: statement fidelity (all 16 statements)

- **Blind back-translation.** A fresh agent given only `Defs.lean` and `Statements.lean`
  translated every statement into LaTeX. Its findings were fixed before approval: `cor_budget`
  (4) relied on `log`/`div` junk values for `q ≤ 0` (now `0 < q`); `cor_beta` allowed `w ≤ 0`
  (now `0 < w`, the floor `φ > 0`); `eq_actw_map` never computed the spread from `Δ`, so the
  single-segment claim was not formalized (now `spread`, conjunct (2b)); `thm_bound`
  allowed `t' = 0` and only one direction of "solved" (now `1 ≤ t' ≤ T`,
  `solved ↔ R(logged) = 1`); `cor_regimes` adds the two failed cells where the exact
  decomposition is a convex combination (the exact identity and the cell masses are now
  conjuncts (2)–(4)).
- **Documented differences, approved.** Proposition `split` (i) is the token-pooled loss (the paper
  said `L_BC`; corrected in the paper). `ρ_BC` is the success rate of the logged seed rollouts
  (the paper treated the temperature-0.01 replay as unique; corrected). `late` may be any
  classification of the failed tasks; Proposition `tstar` is needed only for the exact comparator
  failure, part (2) (corrected in the paper's proof). `p ∈ (0, 1]` in Proposition `target` (free
  logits reach only `(0, 1)`; `p⋆ = 1` is the infimum). Unanchored tokens have `b = 0` (the paper's
  `γ = 10⁻⁶`). Normalized ranks are `(rank − 1)/(n − 1)`.
- **Definition fix.** `Harness.Recoverable` required a terminal action (`done`) for success; the
  paper's "ends with `R = 1`" also allows ending by exhausting the budget. It now accepts both, and
  Proposition `tstar` was re-proved.
- **Conventions checked.** ℕ subtraction (`T - t`, `tl - 1`, `T - (tl - 1)`), `x / 0 = 0`
  (`cond`, `cellAvg`, `ε_corr`, `normRank` at `n = 1`, `1/T` at `T = 0`), the Euclidean norm
  (`logitGradNorm` written out, not the sup norm on `ι → ℝ`), 0-based steps.
- **Non-vacuity.** `Sanity.lean` exhibits instances satisfying every statement's hypotheses, with
  non-degenerate conclusions (an interior cap; `t⋆ = 3 > 1`; a teacher failing with probability
  `1/2`; a nonzero disagreement).
- **Not trivially true.** `simp`, `aesop`, `positivity`, `linarith`, `unfold`+`simp`/`aesop`
  close none of the 16 statements (96 attempts).
- **Teeth.** Dropping an essential hypothesis yields counterexamples: Proposition `tstar` without
  `Failed` (108/108), Proposition `target` without `a > 0` (200/200), Lemma `imit` with rewards
  above `1` (20/300), Theorem `bound-formal` (1) without `solved → R(logged) = 1` (799/910).
- **Numeric falsification.** `falsify/*.py`: 0 violations (lem_mono 1045 checks, prop_tstar 350,
  prop_target 6000, eq_kl_grad 39230, cor_beta 6000, cor_budget 12739, eq_sw 5000,
  rem_resample 4975, eq_actw_map 19091, lem_sandwich 8000, prop_split 8000, thm_rank 231847,
  lem_imit 1500, thm_bound_formal 3000, cor_regimes 4500, thm_bound 3000).

## Gate B: the Lean proof is the paper's proof

- **Map.** `README.md` maps every paper result to its statement, proof file and the lemma for
  each paper step; `Audit.lean` prints the axioms of every declaration in the map.
- **Proof alignment.** Lemma `imit`: the paper's coupling proof was replaced by the one-step
  recursion that `value_gap_le` checks. Proposition `target`: the paper's proof keeps its
  derivative-sign argument and now also states the calculus-free route of `anchorLoss_pStar_le`.
  Theorem `bound-formal`: the paper's proof now says that the inequality does not use the lateness
  of the localization (Lean part (1) holds for any `late`) and that Proposition `tstar` makes the
  comparator's failure exact (part (2)). Corollary `regimes`: Lean proves the exact
  mass-times-average decomposition and then the paper's bound.
- **Side conditions moved to the paper** (blue): success rates conditional on the logged seed
  rollouts; Proposition `split` (i) is about the token-pooled loss; `ε̄ = (1/T) ∑_t δ_t`;
  `ε_ret` and `ε_fit` are disagreements with the relabeled process; the throttle's zero is a
  probability below `1` only when `b_i(1 - q_i) > a_i`; `c_i ≈ a_i` (not `=`) for `γ_i = 10⁻⁶`;
  `p⋆ = 1` is approached, not attained, with free logits; `t⋆ = 1` when `ν_0 = 0`; the rank map is
  first-order optimal only in the twin; "full weight" and "floor" need a unique top or bottom
  segment outside the dead zone; "standard SFT" is the per-token mean over the same tokens.
- **Independent audit.** A fresh agent given the paper and the Lean files listed paper claims not
  backed by Lean, Lean content the paper does not mention, and mismatched steps. Every item was
  fixed in the paper (the list above), documented in `README.md` (modeling choices and "Not
  formalized"), or is a deliberate generalization (e.g. `W`, `Γ` free positive reals; `prop_split`
  (ii) needs neither disjointness nor nonemptiness).
- **Remaining, documented.** Definition `premium` has no Lean object, so "in premium units"
  readings are interpretations. The rate hypothesis of Theorem `rank` is modeled, not derived.
  `late_localization_dead_reward` (harness) and `thm_bound_formal` (2) (response model) compose
  by instantiating the reward; they are not stated as one theorem. Padded post-termination steps
  enter `δ_t` (zero when both policies emit the same padding).
- **Claim scan.** The reproducibility statement ("every theorem can be machine-checked") and the
  LLM declaration ("verify every proof in Lean") overclaimed while Lemma `imit` was a hypothesis;
  both now describe what is formalized and point to this README.

## Gate C: comprehensibility

- **Blind explanation.** A fresh agent given only the proof files (not the paper, not
  `Statements.lean`) explained all thirteen main proofs. Each explanation matches the paper's proof
  except Proposition `target`: Lean minimizes with `log x ≤ x - 1` directly, while the paper argues
  from the sign of the derivative. The paper's proof now also states the calculus-free route.
- **Docstrings corrected** where the agent found them describing something other than the code:
  `Bound.lean` (part (2) assumes dead prefixes; the harness link is
  `late_localization_dead_reward`, added for this), `Imitation.lean` (the proof is an induction,
  not a coupling), `Anchor.lean` (the boundary case with an interior point above `1` uses
  `log p ≥ 1 - 1/p`), `Rank.lean` (the `κ`-monotonicity lemmas back the scope note, not the
  theorem), `FatalStep.lean` (why the continuation at `t⋆ - 1` is nonempty).
- **Automation commented** where it carries a mathematical step: the boundary case of
  `anchorLoss_pStar_le`, the cross-multiplication in `mu_strictMono`, `ratio_mono`, the inactive
  clip in `rankMap_eq_inner`, the cancellation in `rankMap_mean`, the derivative in
  `hasDerivAt_anchorLoss`.
- **Hypotheses the proofs do not use** are domain restrictions kept so that the statements match
  the paper's setting, not hypotheses the paper's proofs need: `q ≤ 1` (`prop_target`), `w > 0`
  (`cor_beta`), `0 ≤ b` in the shed identity and `0 < q` in the uncapped case (`cor_budget`),
  `0 ≤ φ` in `eq_actw_map` (4)–(6), `1 ≤ t'` and the reverse direction of
  `solved ↔ R(logged) = 1` (`thm_bound_formal`).

## Gate D: trust

- **Audit** (`lean_audit.py`, 2026-10-14): fresh compile of all 12 modules from source (1582 s),
  frozen statements unchanged, source scan clean (no forbidden or flagged construct), the 16
  statements bound to their proofs, axioms `propext`, `Classical.choice`, `Quot.sound` only (on
  every result and every line of `Audit.lean`), kernel replay of every package declaration: pass.
  **Verdict: PASS, 16/16 VERIFIED.** The final `--fresh` run replayed every constant the package
  sees, Mathlib included, into an empty environment (2765 s): pass, 16/16 VERIFIED.
- **Second kernel:** none installed (lean4lean, nanoda, Comparator). Residual risk: single-kernel
  trust.
- **Version hygiene:** the Lean 4.34.0 kernel soundness fixes (#14806, #14807, #14838) are in
  `v4.34.0-rc2`. Later fixes (4.35.0-rc: #15241, #15288, #15289) concern reference counts under
  inputs over 100 GB, multithreaded kernel use, or crafted inputs with 18 GB of free RAM; not
  applicable here, but the audit should be re-run on 4.35.0.
- **Redundancy outside Lean:** every proof also passed Gate B (paper proof aligned) and Gate C
  (blind explanation), and every statement passed the numeric falsification tests, so a kernel bug
  would have to coincide with an independent error.
- **Suspicion triggers checked:** no `decide` or `rfl` on large terms, no raised heartbeats or
  `set_option` other than `autoImplicit false`; short proofs are the thin statement wrappers and
  the algebraic identities (`leave_one_out_recovers`, `pmi_eq`, `order_only`), whose triviality
  mirrors the paper's claims. Hypotheses a proof does not use are listed under Gate C (domain
  restrictions, not hypotheses the paper's proofs need).
- **Lakefile:** `maxSynthPendingDepth = 3` and the linter option in `lakefile.toml` are unused by
  the audit, which compiles with bare `lean`; every module compiles without them.

Trust statement: see `README.md`.
