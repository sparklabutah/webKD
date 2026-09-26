# WebKD theory, machine-checked in Lean 4

Lean 4 + Mathlib formalization of the theoretical claims of the WebKD paper (`../main.tex`).
Every statement below is proved without `sorry`; `Audit.lean` prints the axioms each headline
result depends on (only `propext`, `Classical.choice`, `Quot.sound`).

## Build

```
lake exe cache get   # fetch the Mathlib cache once (a few minutes)
lake build           # builds all modules
lake env lean Audit.lean   # axiom audit of the headline theorems
```

Toolchain: `lean-toolchain` pins Lean `v4.34.0-rc2`, Mathlib at the matching tag.

## Paper ↔ Lean map

| Paper | Lean (namespace `WebKD`) |
|---|---|
| Definition `recover` | `Harness.Recoverable`, `Trajectory.nu` (FatalStep.lean) |
| Lemma `mono` (recoverability is absorbing) | `Trajectory.nu_of_nu_succ`, `not_nu_succ_of_not_nu` |
| Proposition `tstar` (unique first fatal step, existence of a recoverability-preserving action) | `Trajectory.first_fatal_step`, `first_fatal_step_unique` |
| P3 asymmetry (late localization is lost, early is still recoverable) | `late_localization_lost`, `early_localization_recoverable` |
| Proposition `scale` (uniform weights recover SFT) | `uniform_weights_recover_sft` (Weighting.lean) |
| Remark `resample` (`L_SW = E_{q_w}[CE]`) | `LSW_eq_expectation`, `qw_sum_eq_one` |
| Eq. `actw-map`: range, dead zone, mean weight | `rankMap_mem`, `rankMap_dead_zone`, `rankMap_mean` |
| Average ranks normalized to `[0,1]` have mean `1/2` (ties sharing their mean rank) | `avgRank_sum`, `normRank_mean`, `rankMap_mean_of_ranks` |
| Remark `pmi` (`Δ_s` is a conditional PMI) | `pmi_eq` |
| Correction: a dead-zone step is not "exactly no reweighting" | `dead_zone_step_not_sft` |
| Eq. `kl-grad` throttle (scalar form), inert at init, helps below the reference | `hasDerivAt_anchorLoss`, `throttle_at_init`, `throttle_gt_of_lt`, `throttle_strictAnti`, `throttle_eq_zero_iff` (Anchor.lean) |
| Eq. `kl-grad`, coordinate form `∇_z L = c_i (p - e_y)`; off-target logit gaps move at rate `-c_i (p_u - p_v)` | `hasDerivAt_anchored_coord`, `offtarget_gap_rate` |
| Proposition `target` (`p⋆` minimizes the per-token contribution on `(0,1]`) | `anchorLoss_pStar_le` |
| Corollary `roles` | `pStar_eq_one_of_le` |
| Corollary `beta` (cap iff `β > β̄`) | `cap_exists_iff`, `ratio_eq_betaBar_div` |
| Corollary `budget` (bound, uncapped case, monotone in `β` and `w`) | `budget_bound`, `budget_uncapped`, `budget_anti_in_beta`, `budget_mono_in_betaBar` |
| `∇_z CE = p - e_y` | `hasDerivAt_nll_coord` (Softmax.lean) |
| Lemma `sandwich` | `grad_sq_bounds`, `grad_norm_bounds`, `sandwich` |
| Proposition `split` (i), (ii); chord of `1 - e^{-x}` | `loss_share`, `grad_share`, `chord_le_one_sub_exp_neg` (Split.lean) |
| Theorem `rank`: product formula, leave-one-out exactness | `deployment_prob_product`, `leave_one_out_recovers` (Rank.lean) |
| Eq. `mu` as a derivative in `log κ`; sign and strict monotonicity | `hasDerivAt_log_factor`, `mu_strictMono`, `mu_zero`, `mu_neg`, `mu_pos` |
| Theorem `rank`, ordering claim (rearrangement) | `rank_order_optimal`, `order_only` |
| Theorem `bound-formal`: comparator failure probability | `comparator_failure_eq`, `comparator_failure_le` (Bound.lean) |
| Theorem `bound-formal`, assembled | `single_round_bound`, `single_round_bound_population` (comparator identity composed with the imitation hypothesis on one population) |
| Corollary `regimes` | `regime_decomposition`, `regime_decomposition_le` (cell masses bounded by `T` per episode, as the corollary states) |

Not formalized: Lemma `imit` (the maximal-coupling argument on the deterministic harness). It
enters `single_round_bound` and `single_round_bound_population` as the hypothesis `himit`, and it
is the only hypothesis of the latter besides the population itself. The comparator is modelled
on a finite population whose failure event is `¬solved ∧ (late ∨ tfail)`: that a late
localization loses the episode is proved on the harness (`late_localization_lost`) but enters
the population model through the definition of the event, not through a formal link between
the two models.

## Discrepancies found and corrected in `main.tex` (purple edits, explained in the text of Appendix J)

1. Eq. `actw-map` sends a dead-zone step (`σ = 0`) to the midpoint weight `(1+φ)/2` on every
   segment, while `⟨A⟩` and `⟨S⟩` tokens stay pinned at `1`. The sentence "degrades to uniform
   weights, which ... is exactly no reweighting" and the matching sentence in Remark `resample`
   were false as stated; `rankMap_mean` shows what is exact (the midpoint is the mean span weight
   of every ranked step, so the step is not down-weighted relative to other steps).
2. "A single-segment step takes `v_s = 1`" contradicts the map (`σ = 0` gives the midpoint). The
   text now states the map's value. If the implementation special-cases one-segment steps to `1`,
   that convention should be added to Eq. `actw-map`.
3. Corollary `budget`: "there is no budget" on `⟨A⟩`, `⟨S⟩` tokens reads as a zero budget; the
   proof gives no cap (`budget_uncapped`).
4. Proposition `tstar`: existence of `t⋆` needs `ν = 0` at the episode's end, not monotonicity
   alone; the statement now says so.
5. Appendix proof of Proposition `target`: the case `b ≤ a` needs `a > 0`, which the floor
   `φ > 0` supplies; now stated.

## Second pass (2026-09-13): gaps closed and discrepancies corrected (red edits in `main.tex`)

6. Appendix `proof-target` claimed that under gradient flow "off-target coordinates shed mass in
   proportion to their current probability, so the off-target distribution retains the
   reference's shape (renormalized)". The coordinate form of Eq. `kl-grad`
   (`hasDerivAt_anchored_coord`) gives `dz_v/dt = -c_i p_v` for `v ≠ y`, so the gap `z_u - z_v`
   moves at rate `-c_i (p_u - p_v)` (`offtarget_gap_rate`): the more probable alternatives fall
   faster and the off-target distribution flattens. The text now says so.
7. The same appendix said `c_i(p)` is strictly decreasing in `p`; this needs `b_i > 0`
   (`throttle_strictAnti`), and for `b_i = 0` the throttle is the constant `a_i`. Now stated.
8. Corollary `budget` printed `log(p⋆_i/q_i) = -log(1 - β̄_i/β)` as an equality. The proof (and
   `budget_bound`) give `≤`, with equality only when the interior cap `q_i b_i/(b_i - a_i)` is not
   clipped at `1`; when it is clipped the token sheds its full `CE_i`, which is below the
   right-hand side. Eq. `budget` and the proof sketch now read `≤`.
9. `rankMap_mean` took "average ranks normalized to `[0,1]` have mean `1/2`" as a hypothesis.
   `avgRank_sum` / `normRank_mean` prove it for every step with at least two segments and
   arbitrary ties, and `rankMap_mean_of_ranks` discharges the hypothesis.
10. Corollary `regimes` says each episode contributes *at most* `T` steps to its cells;
    `regime_decomposition` fixed the masses at exactly `T ρ_BC` and `T (1 - ρ_BC)`.
    `regime_decomposition_le` states the bounded version.
11. `single_round_bound` took the comparator bound as a hypothesis alongside Lemma `imit`;
    `single_round_bound_population` composes it with `comparator_failure_le`, so Lemma `imit` is
    the only hypothesis left.
