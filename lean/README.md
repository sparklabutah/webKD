# WebKD theory, machine-checked in Lean 4

Lean 4 + Mathlib formalization of the theory of *Teaching Agents in their Own Style: Blackbox
Teacher Distillation for Web Agents* (WebKD). Every proposition, lemma, corollary and theorem of
the paper's theory appendix is stated once, as a closed proposition in
`WebkdLean/Statements.lean`, and proved by a theorem whose type *is* that proposition, so the
kernel checks that what is proved is exactly what is stated. There is no `sorry`, no `axiom`, and
no native evaluation; every result depends only on `propext`, `Classical.choice` and `Quot.sound`.

## Layout

| File | Contents |
|---|---|
| `WebkdLean/Defs.lean` | every object the statements mention, each citing the paper equation it formalizes |
| `WebkdLean/Statements.lean` | one `def <label> : Prop` per paper result, named after its LaTeX label |
| `WebkdLean/Sanity.lean` | non-vacuity witnesses: a concrete instance satisfying each statement's hypotheses |
| `WebkdLean/Proofs/*.lean` | the proofs; each file opens with the strategy and ends with the `WebKD.<label>` theorems |
| `Audit.lean` | `#print axioms` for every declaration in the map below |
| `falsify/*.py` | randomized numeric tests of every statement with the paper's constants (independent of Lean) |
| `.statements.sha256` | hashes of `Defs.lean` and `Statements.lean`, frozen when the statements were approved |
| `VERIFICATION.md` | the verification log |

## Build and check

```
lake exe cache get        # fetch the Mathlib cache once
lake build                # builds every module
lake env lean Audit.lean  # axioms of every declaration in the map
cd falsify && for f in prop_tstar prop_target eq_actw_map prop_split thm_rank thm_bound; do python3 $f.py; done
```

Toolchain: Lean `v4.34.0-rc2` (`lean-toolchain`), Mathlib at tag `v4.34.0-rc2`
(commit `85e3a25e006c35636f0e53b0e9296caca2685bc0`, `lake-manifest.json`). On some macOS
setups `lake` crashes on exit (code 133); then compile each module with `lean -o` in dependency
order, with `LEAN_PATH` pointing at `.lake/packages/*/.lake/build/lib/lean`.

## Paper ↔ Lean map

Statements are `WebKD.Statements.<label>`; each is proved by `WebKD.<label>`.

| Paper | Statement | Proof file | Main lemmas |
|---|---|---|---|
| Definition `recover`, Assumptions `budget`, `det`, `reward` | (definitions) `Harness`, `Harness.Recoverable`, `Trajectory.nu`, `Trajectory.Failed` | `Defs.lean` | |
| Lemma `mono` | `lem_mono` | `FatalStep` | `Trajectory.nu_of_nu_succ` |
| Proposition `tstar` | `prop_tstar` | `FatalStep` | `first_fatal_step_of_min`, `first_fatal_step`, `first_fatal_step_unique`, `not_nu_end` |
| P3 asymmetry (late is lost, early is recoverable) | | `FatalStep` | `late_localization_lost`, `early_localization_recoverable` |
| Eq. `kl-grad` and its four regimes (incl. the sign change above `p⋆`) | `eq_kl_grad` | `Anchor` | `hasDerivAt_anchorLoss`, `hasDerivAt_anchored_coord`, `throttle_sign`, `offtarget_gap_rate` |
| Proposition `target` | `prop_target` | `Anchor` | `anchorLoss_pStar_le`, `pStar_pos_le_one` |
| Corollary `beta` and the binding condition after it | `cor_beta` | `Anchor` | `cap_exists_iff`, `pStar_lt_one_iff`, `pStar_lt_one_iff_beta` |
| Corollary `budget` | `cor_budget` | `Anchor` | `budget_bound`, `budget_eq_of_interior`, `budget_uncapped`, `budget_anti_in_beta`, `budget_mono_in_betaBar` |
| Eq. `sw`: uniform weights recover SFT | `eq_sw` | `Weighting` | `uniform_weights_recover_sft` |
| Remark `resample` | `rem_resample` | `Weighting` | `LSW_eq_expectation`, `qw_sum_eq_one` |
| Eq. `actw-map`: range, dead zone, mean weight, order, extremes | `eq_actw_map` | `Weighting` | `rankMap_zero_spread`, `spread_eq_zero_of_tie`, `rankMap_mean_of_ranks`, `avgRank_sum`, `avgRank_lt_of_lt`, `rankMap_monovary`, `rankMap_top`, `rankMap_bottom` |
| Remark `pmi` | (algebra only) | `Weighting` | `pmi_eq`: the log-ratio identity; calling it a PMI is an interpretation |
| Lemma `sandwich` | `lem_sandwich` | `Softmax`, `Split` | `hasDerivAt_nll_coord`, `grad_norm_bounds`, `sandwich` |
| Proposition `split` | `prop_split` | `Split` | `loss_share`, `grad_share`, `grad_share_logits`, `chord_le_one_sub_exp_neg` |
| Theorem `rank` | `thm_rank` | `Rank` | `deployment_prob_product`, `leave_one_out_recovers`, `hasDerivAt_log_deployment`, `mu_strictMono`, `mu_neg`, `rank_order_optimal`, `order_only` |
| Theorem `rank`, scope note (`μ_j` in `κ_j`) | | `Rank` | `mu_strictMonoOn_kappa`, `mu_strictAntiOn_kappa` |
| Lemma `imit` | `lem_imit` | `Imitation` | `sum_sub_mul_le_tv`, `value_gap_le` |
| Theorem `bound-formal` | `thm_bound_formal` | `Bound`, `Imitation` | `relabel_value_solved`, `relabel_value_failed`, `expectAt_replay`, `expectAt_eq_zero_of_dead`, `Population.prob_failed_notlate`, `Population.restricted_sum_eq` |
| Proof of Theorem `bound-formal`: a late localization is lost (uses Proposition `tstar`); gives the hypothesis of part (2) | | `Imitation` | `late_localization_dead`, `late_localization_dead_reward`, `dead_prefix_of_not_recoverable`, `state_eq_run_take` |
| Corollary `regimes` | `cor_regimes` | `Bound` | `cell_sum_eq`, `regime_decomposition_le` |
| Theorem `bound` (main text) | `thm_bound` | `Bound` | |

## What the statements assume (modeling choices)

- **Bound (Lemma `imit`, Theorem `bound-formal`, Corollary `regimes`, Theorem `bound`).** A finite
  task distribution and a finite set of structured responses (token strings of bounded length over
  a finite vocabulary). Under Assumption `det`, a history is the task plus the responses emitted so
  far, so policies are maps from response prefixes to probability vectors. The reward lies in
  `[0, 1]` and is read off the complete response sequence of length `T`; an episode that ends early
  is padded with responses that do not change its reward (Assumption `budget`). Padded steps enter
  `δ_t` and the cells of Corollary `regimes` like any other step; if both policies emit the same
  padding response after the end, they contribute nothing, and the cells count only real steps. All quantities are
  conditional on the logged seed rollouts: `ρ_BC` is the fraction of tasks whose logged rollout
  succeeded (`solved ω ↔ R(logged ω) = 1`), and the relabeled process `relabel` is built explicitly
  (replay the log, then the teacher from `h_{t'}`, `t' ∈ {1, …, T}`). The inequality holds for any
  classification of the failed tasks as late or not; part (2), the exact comparator failure
  probability, assumes that late prefixes are dead, which `late_localization_dead_reward` derives
  from Proposition `tstar` for the reward read off the harness.
- **Anchor (Eq. `kl-grad`, Proposition `target`, Corollaries `beta`, `budget`).** One supervised
  position with free logits (the paper's tabular idealization); `p` ranges over `(0, 1]` (the
  minimum at `p = 1` is approached as the target logit grows); unanchored tokens are modeled with
  `b = 0` (the paper uses `γ_i = 10⁻⁶`, so `b_i ≈ 0`); the clamp of `d_i` is not modeled.
- **Weighting (Eq. `actw-map`).** The normalized average rank is `(rank − 1)/(n − 1)` with ties
  sharing their mean rank; the spread is `max Δ − min Δ` over the step.
- **Proposition `split`.** Every position has its own logits; part (i) is the token-pooled uniform
  loss (every token weighted equally).
- **Theorem `rank`.** Assumptions `additive` and `emit` are encoded in the deployment probability;
  the rate hypothesis is the modeling step that makes `∑_j μ_j v_j` the first-order gain, and the
  theorem optimizes that sum.

## Not formalized

- Definition `premium` and Eq. `ppl` have no Lean object, so readings of results "in units of the
  premium" (Proposition `split`, Corollary `budget`, Theorem `rank`) are interpretations of the
  checked statements, which are about cross-entropies and log-probabilities.
- The reading of `Δ_s` as a pointwise mutual information (Remark `pmi`); only the log-ratio
  identity is checked. Likewise the sentence of Remark `resample` that gradient descent on
  `L_SW` is SFT on a resampled corpus, and the properties of the `k₃` estimator.
- In the proof of Proposition `target`, the description of the gradient flow beyond the
  coordinate form of the gradient (`hasDerivAt_anchored_coord`, `offtarget_gap_rate`).
- In Theorem `rank`, the step from a gradient step on `L_SW` to the first-order gain
  `η ∑_j μ_j v_j` (the rate hypothesis); the theorem optimizes that sum.
- The scope arguments that follow some proofs (the tabular idealization, the logit-space
  justification of the rate hypothesis, the effect of the log-ratio clamp) and the predictions
  of Remark `rank`.
- The composition of `late_localization_dead_reward` (harness) with part (2) of
  `thm_bound_formal` (response model) is by instantiating `R` with the harness reward; it is not
  stated as one theorem.

## Lean conventions that affect the statements

`x / 0 = 0` (conditional probabilities, cell averages and `ε_corr` of empty events are `0`);
natural-number subtraction truncates at `0` (all uses stay in range); `logitGradNorm` is the
Euclidean norm, written out; steps are `0`-based in Lean (`range T` is the paper's `t = 1, …, T`).

## Trust

Trusted: Lean kernel `leanprover/lean4:v4.34.0-rc2`, Mathlib `85e3a25e006c35636f0e53b0e9296caca2685bc0`,
axioms `propext`, `Classical.choice`, `Quot.sound`. Kernel replay: pass, `--fresh`: yes (every constant the
package sees, Mathlib included, replayed into an empty environment). Independent kernel: none, so this is single-kernel trust. Numeric
falsification: pass. Statements human-approved: yes (2026-10-14).

The audit (the `/theory` skill's `lean_audit.py`) recompiles the package from source into a
temporary directory, checks that `Defs.lean` and `Statements.lean` match their frozen hashes,
scans every source for forbidden constructs (`sorry`, `axiom`, `native_decide`, `implemented_by`,
`unsafe`, metaprograms, ...), checks that each `WebKD.<label>` proves
`WebKD.Statements.<label>`, prints the axioms of each result and of every line of `Audit.lean`,
and replays every declaration through the kernel (with `--fresh`, every imported constant too).
Verdict on 2026-10-14, with and without `--fresh`: PASS, all 16 statements `VERIFIED`.

Version hygiene: the kernel soundness fixes of Lean 4.34.0 (#14806, #14807, #14838) are in
`v4.34.0-rc2` (the release notes shipped with rc2 list them). Lean 4.35.0 release candidates fix
reference-count edge cases that need inputs over 100 GB, multithreaded kernel use, or a crafted
input on a machine with at least 18 GB of free RAM; they do not plausibly affect these small,
native-code-free proofs, but re-running the audit on 4.35.0 once Mathlib supports it is
recommended. A second kernel would remove the single-kernel trust: run
[Comparator](https://github.com/leanprover/comparator) with the nanoda checker (Linux), or
`lean4lean`, against `Statements.lean`.
