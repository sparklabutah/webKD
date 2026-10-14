import WebkdLean.Proofs.Imitation

set_option autoImplicit false
set_option relaxedAutoImplicit false

/-!
# The single-round relabeling bound (Theorem `bound-formal`, Corollary `regimes`, Theorem `bound`)

Strategy, following Appendix `bound`. First the relabeled process `π̃` (Imitation.lean): it
succeeds with probability `1` on solved tasks (it replays the successful log) and with the
teacher's success probability `1 - f(ω)` from `h_{t'}` on failed ones. So its failure probability
is `∑_{failed} P(ω) f(ω)`, which splits into the late tasks, bounded by their mass
`P(late ∧ ¬solved) = ε_loc (1 - ρ_BC)` because `f ≤ 1`, and the not-late tasks, whose sum is
`ε_corr P(¬solved ∧ ¬late) = ε_corr (1 - ε_loc)(1 - ρ_BC)`. This bound on late tasks needs no
property of the localization. Part (2) assumes that late prefixes are dead; for the reward read
off the harness this follows from Proposition `tstar` (`late_localization_dead_reward`), and then
`f = 1` there and the comparator's failure probability is exact.
Lemma `imit` then adds `∑_t δ_t`. For Corollary `regimes`, every (task, step) pair lies in exactly
one cell, a cell's total is its mass times its average, the solved cell has mass `T ρ_BC` and the
two failed cells together `T (1 - ρ_BC)`; `regime_decomposition_le` finishes. Theorem `bound` is
the two combined with `ε̄ = (1/T) ∑_t δ_t`.

The earlier predicate-based population model (`comparator_failure_eq`, `single_round_bound`,
`single_round_bound_population`, `regime_decomposition`) is kept for reference; the statements are
proved in the response model, and `regime_decomposition_le` is reused by `cor_regimes`.
-/

namespace WebKD

open Finset

variable {Ω : Type} [Fintype Ω] (pop : Population Ω)

lemma Population.prob_nonneg (E : Ω → Prop) [DecidablePred E] : 0 ≤ pop.prob E := by
  unfold Population.prob
  apply Finset.sum_nonneg
  intro ω _
  split_ifs
  · exact pop.nonneg ω
  · exact le_rfl

lemma Population.prob_mono (E F : Ω → Prop) [DecidablePred E] [DecidablePred F]
    (h : ∀ ω, E ω → F ω) : pop.prob E ≤ pop.prob F :=
  Finset.sum_le_sum fun ω _ => by
    by_cases hE : E ω
    · simp [hE, h ω hE]
    · simp only [hE, ite_false]
      split_ifs
      · exact pop.nonneg ω
      · exact le_rfl

lemma Population.prob_compl (E : Ω → Prop) [DecidablePred E] :
    pop.prob (fun ω => ¬ E ω) = 1 - pop.prob E := by
  unfold Population.prob
  rw [eq_sub_iff_add_eq, ← Finset.sum_add_distrib, ← pop.total]
  refine Finset.sum_congr rfl fun ω _ => ?_
  by_cases h : E ω <;> simp [h]

/-- `P(E ∧ F) = P(E | F) P(F)`, also when `P(F) = 0`. -/
lemma Population.prob_and_eq_cond_mul (E F : Ω → Prop) [DecidablePred E] [DecidablePred F] :
    pop.prob (fun ω => E ω ∧ F ω) = pop.cond E F * pop.prob F := by
  unfold Population.cond
  by_cases hF : pop.prob F = 0
  · have h0 : pop.prob (fun ω => E ω ∧ F ω) = 0 :=
      le_antisymm (hF ▸ pop.prob_mono _ _ fun ω h => h.2) (pop.prob_nonneg _)
    rw [h0, hF, mul_zero]
  · rw [div_mul_cancel₀ _ hF]

lemma Population.cond_nonneg (E F : Ω → Prop) [DecidablePred E] [DecidablePred F] :
    0 ≤ pop.cond E F :=
  div_nonneg (pop.prob_nonneg _) (pop.prob_nonneg _)

lemma Population.cond_le_one (E F : Ω → Prop) [DecidablePred E] [DecidablePred F] :
    pop.cond E F ≤ 1 := by
  unfold Population.cond
  by_cases hF : pop.prob F = 0
  · rw [hF, div_zero]; exact zero_le_one
  · rw [div_le_one (lt_of_le_of_ne (pop.prob_nonneg F) (Ne.symm hF))]
    exact pop.prob_mono _ _ fun ω h => h.2

/-- **The comparator (relabeled process) of Theorem `bound-formal`.** Let `solved ω` be the seed
rollout succeeding, `late ω` the localization being late (`t' > t⋆`), and `tfail ω` the
teacher-as-agent failing from `h_{t'}`. The relabeled process fails exactly on
`¬solved ∧ (late ∨ tfail)`, and with `ρ_BC = P(solved)`, `ε_loc = P(late | ¬solved)`,
`ε_corr = P(tfail | ¬solved ∧ ¬late)`, its failure probability is
`(1 - ρ_BC)(ε_loc + (1 - ε_loc) ε_corr)`. -/
theorem comparator_failure_eq (solved late tfail : Ω → Prop)
    [DecidablePred solved] [DecidablePred late] [DecidablePred tfail] :
    pop.prob (fun ω => ¬ solved ω ∧ (late ω ∨ tfail ω))
      = (1 - pop.prob solved)
        * (pop.cond late (fun ω => ¬ solved ω)
          + (1 - pop.cond late (fun ω => ¬ solved ω))
            * pop.cond tfail (fun ω => ¬ solved ω ∧ ¬ late ω)) := by
  -- split the failure event into the late part and the not-late-but-teacher-fails part
  have hsplit : pop.prob (fun ω => ¬ solved ω ∧ (late ω ∨ tfail ω))
      = pop.prob (fun ω => late ω ∧ ¬ solved ω)
        + pop.prob (fun ω => tfail ω ∧ (¬ solved ω ∧ ¬ late ω)) := by
    unfold Population.prob
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun ω _ => ?_
    by_cases hs : solved ω <;> by_cases hl : late ω <;> by_cases ht : tfail ω <;> simp [hs, hl, ht]
  -- the not-late mass among failures
  have hnotlate : pop.prob (fun ω => ¬ solved ω ∧ ¬ late ω)
      = (1 - pop.cond late (fun ω => ¬ solved ω)) * (1 - pop.prob solved) := by
    have h1 : pop.prob (fun ω => ¬ solved ω ∧ ¬ late ω) + pop.prob (fun ω => late ω ∧ ¬ solved ω)
        = pop.prob (fun ω => ¬ solved ω) := by
      unfold Population.prob
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun ω _ => ?_
      by_cases hs : solved ω <;> by_cases hl : late ω <;> simp [hs, hl]
    rw [pop.prob_and_eq_cond_mul late, pop.prob_compl] at h1
    linarith
  rw [hsplit, pop.prob_and_eq_cond_mul late, pop.prob_compl,
    pop.prob_and_eq_cond_mul tfail, hnotlate]
  ring

/-- The comparator failure probability is at most `(1 - ρ_BC)(ε_loc + ε_corr)`. -/
theorem comparator_failure_le (solved late tfail : Ω → Prop)
    [DecidablePred solved] [DecidablePred late] [DecidablePred tfail] :
    pop.prob (fun ω => ¬ solved ω ∧ (late ω ∨ tfail ω))
      ≤ (1 - pop.prob solved)
        * (pop.cond late (fun ω => ¬ solved ω)
          + pop.cond tfail (fun ω => ¬ solved ω ∧ ¬ late ω)) := by
  rw [comparator_failure_eq]
  have h1 : 0 ≤ 1 - pop.prob solved := by
    have := pop.prob_compl solved
    rw [← this]; exact pop.prob_nonneg _
  apply mul_le_mul_of_nonneg_left _ h1
  have hl0 := pop.cond_nonneg late (fun ω => ¬ solved ω)
  have hc0 := pop.cond_nonneg tfail (fun ω => ¬ solved ω ∧ ¬ late ω)
  nlinarith [mul_nonneg hl0 hc0]

/-- **Theorem `bound-formal`, assembled.** From the comparator bound and Lemma `imit`
(`ρ(π̃) - ρ(π_WD) ≤ ∑_t δ_t`, the maximal-coupling argument on the deterministic harness, taken
here as a hypothesis), `1 - ρ(π_WD) ≤ (1 - ρ_BC)(ε_loc + ε_corr) + ∑_t δ_t`. -/
theorem single_round_bound (ρWD ρtilde ρBC εloc εcorr : ℝ) (δ : ℕ → ℝ) (T : ℕ)
    (hρBC : ρBC ≤ 1) (hloc0 : 0 ≤ εloc) (hcorr : 0 ≤ εcorr)
    (hcomp : 1 - ρtilde ≤ (1 - ρBC) * (εloc + (1 - εloc) * εcorr))
    (himit : ρtilde - ρWD ≤ ∑ t ∈ Finset.range T, δ t) :
    1 - ρWD ≤ (1 - ρBC) * (εloc + εcorr) + ∑ t ∈ Finset.range T, δ t := by
  have : (1 - ρBC) * (εloc + (1 - εloc) * εcorr) ≤ (1 - ρBC) * (εloc + εcorr) := by
    apply mul_le_mul_of_nonneg_left _ (by linarith)
    nlinarith [mul_nonneg hloc0 hcorr]
  linarith

/-- **Corollary `regimes`.** If the total per-step disagreement mass decomposes over the three
cells as `T ρ_BC ε_ret + m₂ ε_fit + m₃ ε_rec` with `m₂ + m₃ = T (1 - ρ_BC)` (each failed episode
contributes `T` steps split between the supervised and the post-correction cell), then
`∑_t δ_t ≤ T (ρ_BC ε_ret + (1 - ρ_BC)(ε_fit + ε_rec))`. -/
theorem regime_decomposition (T ρBC εret εfit εrec m2 m3 : ℝ)
    (hfit : 0 ≤ εfit) (hrec : 0 ≤ εrec) (hm2 : 0 ≤ m2) (hm3 : 0 ≤ m3)
    (hmass : m2 + m3 = T * (1 - ρBC)) :
    T * ρBC * εret + m2 * εfit + m3 * εrec
      ≤ T * (ρBC * εret + (1 - ρBC) * (εfit + εrec)) := by
  have h2 : m2 * εfit ≤ T * (1 - ρBC) * εfit := by
    rw [← hmass]; nlinarith [mul_nonneg hm3 hfit]
  have h3 : m3 * εrec ≤ T * (1 - ρBC) * εrec := by
    rw [← hmass]; nlinarith [mul_nonneg hm2 hrec]
  nlinarith

/-- **Theorem `bound-formal` on a finite population.** With `ρ(π̃) = 1 - P(comparator fails)`
and Lemma `imit` as the only hypothesis (`ρ(π̃) - ρ(π_WD) ≤ ∑_t δ_t`),
`1 - ρ(π_WD) ≤ (1 - ρ_BC)(ε_loc + ε_corr) + ∑_t δ_t`. -/
theorem single_round_bound_population (solved late tfail : Ω → Prop)
    [DecidablePred solved] [DecidablePred late] [DecidablePred tfail]
    (ρWD : ℝ) (δ : ℕ → ℝ) (T : ℕ)
    (himit : (1 - pop.prob (fun ω => ¬ solved ω ∧ (late ω ∨ tfail ω))) - ρWD
        ≤ ∑ t ∈ Finset.range T, δ t) :
    1 - ρWD ≤ (1 - pop.prob solved)
        * (pop.cond late (fun ω => ¬ solved ω)
          + pop.cond tfail (fun ω => ¬ solved ω ∧ ¬ late ω))
      + ∑ t ∈ Finset.range T, δ t := by
  have hle := comparator_failure_le pop solved late tfail
  linarith

/-- **Corollary `regimes`, with bounded cell masses.** If the total per-step disagreement mass
decomposes over the three cells as `m₁ ε_ret + m₂ ε_fit + m₃ ε_rec` with `m₁ ≤ T ρ_BC` (solved
episodes contribute at most `T` steps each) and `m₂ + m₃ ≤ T (1 - ρ_BC)` (failed episodes
likewise), then `∑_t δ_t ≤ T (ρ_BC ε_ret + (1 - ρ_BC)(ε_fit + ε_rec))`. -/
theorem regime_decomposition_le (T ρBC εret εfit εrec m1 m2 m3 : ℝ)
    (hret : 0 ≤ εret) (hfit : 0 ≤ εfit) (hrec : 0 ≤ εrec) (hm2 : 0 ≤ m2) (hm3 : 0 ≤ m3)
    (hm1 : m1 ≤ T * ρBC) (hmass : m2 + m3 ≤ T * (1 - ρBC)) :
    m1 * εret + m2 * εfit + m3 * εrec
      ≤ T * (ρBC * εret + (1 - ρBC) * (εfit + εrec)) := by
  have h1 : m1 * εret ≤ T * ρBC * εret := mul_le_mul_of_nonneg_right hm1 hret
  have h2 : m2 * εfit ≤ T * (1 - ρBC) * εfit :=
    mul_le_mul_of_nonneg_right (by linarith) hfit
  have h3 : m3 * εrec ≤ T * (1 - ρBC) * εrec :=
    mul_le_mul_of_nonneg_right (by linarith) hrec
  nlinarith [h1, h2, h3]

/-! ### The comparator in the response model -/

/-- Mass of the failed, not-late tasks: `P(¬solved ∧ ¬late) = (1 - ε_loc)(1 - ρ_BC)`. -/
lemma Population.prob_failed_notlate (solved late : Ω → Prop) [DecidablePred solved]
    [DecidablePred late] :
    pop.prob (fun ω => ¬ solved ω ∧ ¬ late ω)
      = (1 - pop.cond late (fun ω => ¬ solved ω)) * (1 - pop.prob solved) := by
  have h1 : pop.prob (fun ω => ¬ solved ω ∧ ¬ late ω) + pop.prob (fun ω => late ω ∧ ¬ solved ω)
      = pop.prob (fun ω => ¬ solved ω) := by
    unfold Population.prob
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun ω _ => ?_
    by_cases hs : solved ω <;> by_cases hl : late ω <;> simp [hs, hl]
  rw [pop.prob_and_eq_cond_mul late, pop.prob_compl] at h1
  linarith

/-- A restricted sum is its conditional average times the conditioning mass, also when the mass
is `0` (then every task in the event has probability `0`). -/
lemma Population.restricted_sum_eq (E : Ω → Prop) [DecidablePred E] (g : Ω → ℝ) :
    (∑ ω, if E ω then pop.P ω * g ω else 0)
      = (∑ ω, if E ω then pop.P ω * g ω else 0) / pop.prob E * pop.prob E := by
  by_cases hz : pop.prob E = 0
  · have hP : ∀ ω, E ω → pop.P ω = 0 := by
      intro ω hω
      unfold Population.prob at hz
      have hterm := (Finset.sum_eq_zero_iff_of_nonneg (fun ω _ => by
        split_ifs
        · exact pop.nonneg ω
        · exact le_rfl)).1 hz ω (Finset.mem_univ ω)
      simpa [hω] using hterm
    have hsum : (∑ ω, if E ω then pop.P ω * g ω else 0) = 0 :=
      Finset.sum_eq_zero fun ω _ => by
        split_ifs with hω
        · rw [hP ω hω, zero_mul]
        · rfl
    rw [hsum, zero_div, zero_mul]
  · rw [div_mul_cancel₀ _ hz]

/-- **Theorem `bound-formal`** (`Statements.thm_bound_formal`). -/
theorem thm_bound_formal : Statements.thm_bound_formal := by
  intro Ω Y _ _ _ _ pop T R solved late _ _ logged tl πT πWD hR hπT hπWD hlen htl hsolved
    πrel ρ ρBC εloc f εcorr δ
  have hπrel : πrel = relabel solved logged tl πT := rfl
  have hρ : ∀ π, ρ π = ∑ ω, pop.P ω * expectAt (π ω) T (R ω) [] := fun _ => rfl
  have hρBC : ρBC = pop.prob solved := rfl
  have hεloc : εloc = pop.cond late (fun ω => ¬ solved ω) := rfl
  have hf : ∀ ω, f ω
      = 1 - expectAt (πT ω) (T - (tl ω - 1)) (R ω) ((logged ω).take (tl ω - 1)) := fun _ => rfl
  have hεcorr : εcorr = (∑ ω, if ¬ solved ω ∧ ¬ late ω then pop.P ω * f ω else 0)
      / pop.prob (fun ω => ¬ solved ω ∧ ¬ late ω) := rfl
  have hδ : ∀ t, δ t
      = ∑ ω, pop.P ω * expectAt (πrel ω) t (fun h => tv (πWD ω h) (πrel ω h)) [] := fun _ => rfl
  clear_value πrel ρ ρBC εloc f εcorr δ
  have hπrelP : ∀ ω, IsPolicy (πrel ω) := by
    rw [hπrel]; exact relabel_isPolicy solved logged tl πT hπT
  -- the teacher's success probability lies in `[0, 1]`, so does its failure probability `f`
  have hf01 : ∀ ω, 0 ≤ f ω ∧ f ω ≤ 1 := by
    intro ω
    have := expectAt_mem_unit (πT ω) (hπT ω) (R ω) (hR ω) (T - (tl ω - 1))
      ((logged ω).take (tl ω - 1))
    rw [hf ω]
    constructor <;> linarith [this.1, this.2]
  -- paper step: on solved tasks `π̃` replays the log and succeeds
  have hVs : ∀ ω, solved ω → expectAt (πrel ω) T (R ω) [] = 1 := by
    intro ω hs
    rw [hπrel, relabel_value_solved solved logged tl πT (R ω) T ω hs (hlen ω)]
    exact (hsolved ω).1 hs
  -- paper step: on failed tasks `π̃` succeeds as the teacher does from `h_{t'}`
  have hVf : ∀ ω, ¬ solved ω → expectAt (πrel ω) T (R ω) [] = 1 - f ω := by
    intro ω hs
    rw [hπrel, relabel_value_failed solved logged tl πT (R ω) T ω hs (hlen ω) (htl ω).2, hf ω]
    ring
  -- the comparator's failure probability is a sum over the failed tasks
  have hfail : 1 - ρ πrel = ∑ ω, if ¬ solved ω then pop.P ω * f ω else 0 := by
    rw [hρ]
    calc 1 - ∑ ω, pop.P ω * expectAt (πrel ω) T (R ω) []
        = ∑ ω, pop.P ω - ∑ ω, pop.P ω * expectAt (πrel ω) T (R ω) [] := by rw [pop.total]
      _ = ∑ ω, (pop.P ω - pop.P ω * expectAt (πrel ω) T (R ω) []) := by
          rw [Finset.sum_sub_distrib]
      _ = ∑ ω, if ¬ solved ω then pop.P ω * f ω else 0 := by
          refine Finset.sum_congr rfl fun ω _ => ?_
          by_cases hs : solved ω
          · rw [hVs ω hs]; simp [hs]
          · rw [hVf ω hs]; simp only [hs, not_false_eq_true, ↓reduceIte]; ring
  -- split the failed tasks into late and not late
  have hsplit : (∑ ω, if ¬ solved ω then pop.P ω * f ω else 0)
      = (∑ ω, if late ω ∧ ¬ solved ω then pop.P ω * f ω else 0)
        + ∑ ω, if ¬ solved ω ∧ ¬ late ω then pop.P ω * f ω else 0 := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun ω _ => ?_
    by_cases hs : solved ω <;> by_cases hl : late ω <;> simp [hs, hl]
  -- a late task fails with probability at most `1`, whatever the localization did
  have hlate_le : (∑ ω, if late ω ∧ ¬ solved ω then pop.P ω * f ω else 0)
      ≤ pop.prob (fun ω => late ω ∧ ¬ solved ω) := by
    unfold Population.prob
    refine Finset.sum_le_sum fun ω _ => ?_
    split_ifs
    · calc pop.P ω * f ω ≤ pop.P ω * 1 := mul_le_mul_of_nonneg_left (hf01 ω).2 (pop.nonneg ω)
        _ = pop.P ω := mul_one _
    · exact le_rfl
  -- the not-late tasks fail with conditional probability `ε_corr`
  have hnotlate : (∑ ω, if ¬ solved ω ∧ ¬ late ω then pop.P ω * f ω else 0)
      = εcorr * pop.prob (fun ω => ¬ solved ω ∧ ¬ late ω) := by
    rw [hεcorr]
    exact pop.restricted_sum_eq _ f
  have hmass_late : pop.prob (fun ω => late ω ∧ ¬ solved ω) = εloc * (1 - ρBC) := by
    rw [pop.prob_and_eq_cond_mul late, pop.prob_compl, hεloc, hρBC]
  have hmass_notlate : pop.prob (fun ω => ¬ solved ω ∧ ¬ late ω) = (1 - εloc) * (1 - ρBC) := by
    rw [pop.prob_failed_notlate, hεloc, hρBC]
  have hεcorr0 : 0 ≤ εcorr := by
    rw [hεcorr]
    refine div_nonneg (Finset.sum_nonneg fun ω _ => ?_) (pop.prob_nonneg _)
    split_ifs
    · exact mul_nonneg (pop.nonneg ω) (hf01 ω).1
    · exact le_rfl
  have hεloc01 : 0 ≤ εloc ∧ εloc ≤ 1 := by
    rw [hεloc]; exact ⟨pop.cond_nonneg _ _, pop.cond_le_one _ _⟩
  have hρBC1 : 0 ≤ 1 - ρBC := by
    rw [hρBC, ← pop.prob_compl]; exact pop.prob_nonneg _
  -- the comparator's failure probability, first display of the paper's proof
  have hcomp : 1 - ρ πrel ≤ (1 - ρBC) * (εloc + (1 - εloc) * εcorr) := by
    rw [hfail, hsplit, hnotlate, hmass_notlate]
    nlinarith [hlate_le, hmass_late]
  -- Lemma `imit` with `π = π_WD`, `π̃ = π̃`
  have himit : ρ πrel - ρ πWD ≤ ∑ t ∈ range T, δ t := by
    rw [hρ, hρ]
    simp only [hδ]
    exact lem_imit pop πWD πrel R T hπWD hπrelP hR
  refine ⟨?_, ?_⟩
  · -- (1): drop the factor `1 - ε_loc` and add the imitation term
    have : (1 - ρBC) * (εloc + (1 - εloc) * εcorr) ≤ (1 - ρBC) * (εloc + εcorr) := by
      apply mul_le_mul_of_nonneg_left _ hρBC1
      nlinarith [mul_nonneg hεloc01.1 hεcorr0]
    linarith
  · -- (2): with dead late prefixes, a late task fails with probability exactly `1`
    intro hdead
    have hlate_eq : (∑ ω, if late ω ∧ ¬ solved ω then pop.P ω * f ω else 0)
        = pop.prob (fun ω => late ω ∧ ¬ solved ω) := by
      unfold Population.prob
      refine Finset.sum_congr rfl fun ω _ => ?_
      split_ifs with h
      · have hz : expectAt (πT ω) (T - (tl ω - 1)) (R ω) ((logged ω).take (tl ω - 1)) = 0 :=
          expectAt_eq_zero_of_dead (πT ω) (R ω) _ _ (hdead ω h.2 h.1)
        rw [hf ω, hz, sub_zero, mul_one]
      · rfl
    rw [hfail, hsplit, hnotlate, hmass_notlate, hlate_eq, hmass_late]
    ring

/-! ### Corollary `regimes` -/

/-- A cell's total is its mass times its average, also when the mass is `0` (then every task in
the cell has probability `0`). -/
lemma cell_sum_eq (T : ℕ) (cell : Ω → ℕ → Prop) [∀ ω t, Decidable (cell ω t)] (e : Ω → ℕ → ℝ) :
    (∑ t ∈ range T, ∑ ω, if cell ω t then pop.P ω * e ω t else 0)
      = cellMass pop T cell * cellAvg pop T cell e := by
  unfold cellAvg cellMass
  by_cases hz : (∑ t ∈ range T, ∑ ω, if cell ω t then pop.P ω else 0) = 0
  · rw [hz, zero_mul]
    have hterm : ∀ t ∈ range T, ∀ ω, cell ω t → pop.P ω = 0 := by
      intro t ht ω hω
      have hin := (Finset.sum_eq_zero_iff_of_nonneg (fun t _ => Finset.sum_nonneg fun ω _ => by
        split_ifs
        · exact pop.nonneg ω
        · exact le_rfl)).1 hz t ht
      have := (Finset.sum_eq_zero_iff_of_nonneg (fun ω _ => by
        split_ifs
        · exact pop.nonneg ω
        · exact le_rfl)).1 hin ω (Finset.mem_univ ω)
      simpa [hω] using this
    refine Finset.sum_eq_zero fun t ht => Finset.sum_eq_zero fun ω _ => ?_
    split_ifs with hω
    · rw [hterm t ht ω hω, zero_mul]
    · rfl
  · field_simp

/-- Cell masses and averages are nonnegative. -/
lemma cellMass_nonneg (T : ℕ) (cell : Ω → ℕ → Prop) [∀ ω t, Decidable (cell ω t)] :
    0 ≤ cellMass pop T cell :=
  Finset.sum_nonneg fun t _ => Finset.sum_nonneg fun ω _ => by
    split_ifs
    · exact pop.nonneg ω
    · exact le_rfl

lemma cellAvg_nonneg (T : ℕ) (cell : Ω → ℕ → Prop) [∀ ω t, Decidable (cell ω t)]
    (e : Ω → ℕ → ℝ) (he : ∀ ω t, 0 ≤ e ω t) : 0 ≤ cellAvg pop T cell e := by
  refine div_nonneg (Finset.sum_nonneg fun t _ => Finset.sum_nonneg fun ω _ => ?_)
    (cellMass_nonneg pop T cell)
  split_ifs
  · exact mul_nonneg (pop.nonneg ω) (he ω t)
  · exact le_rfl

/-- **Corollary `regimes`** (`Statements.cor_regimes`). -/
theorem cor_regimes : Statements.cor_regimes := by
  intro Ω _ pop T solved _ tl e he
  -- every (task, step) pair lies in exactly one cell
  have htot : ∑ t ∈ range T, ∑ ω, pop.P ω * e ω t
      = (∑ t ∈ range T, ∑ ω, if solved ω then pop.P ω * e ω t else 0)
        + (∑ t ∈ range T, ∑ ω, if ¬ solved ω ∧ t < tl ω then pop.P ω * e ω t else 0)
        + (∑ t ∈ range T, ∑ ω, if ¬ solved ω ∧ tl ω ≤ t then pop.P ω * e ω t else 0) := by
    rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun t _ => ?_
    rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun ω _ => ?_
    by_cases hs : solved ω
    · simp [hs]
    · by_cases ht : t < tl ω
      · simp [hs, ht, not_le.2 ht]
      · simp [hs, ht, not_lt.1 ht]
  have h1 : (∑ t ∈ range T, ∑ ω, if solved ω then pop.P ω * e ω t else 0)
      = cellMass pop T (fun ω _ => solved ω) * cellAvg pop T (fun ω _ => solved ω) e :=
    cell_sum_eq pop T (fun ω _ => solved ω) e
  have h2 : (∑ t ∈ range T, ∑ ω, if ¬ solved ω ∧ t < tl ω then pop.P ω * e ω t else 0)
      = cellMass pop T (fun ω t => ¬ solved ω ∧ t < tl ω)
        * cellAvg pop T (fun ω t => ¬ solved ω ∧ t < tl ω) e :=
    cell_sum_eq pop T (fun ω t => ¬ solved ω ∧ t < tl ω) e
  have h3 : (∑ t ∈ range T, ∑ ω, if ¬ solved ω ∧ tl ω ≤ t then pop.P ω * e ω t else 0)
      = cellMass pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t)
        * cellAvg pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t) e :=
    cell_sum_eq pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t) e
  -- the solved cell has mass `T ρ_BC`
  have hm1 : cellMass pop T (fun ω _ => solved ω) = T * pop.prob solved := by
    simp only [cellMass, Population.prob]
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  -- the two failed cells together have mass `T (1 - ρ_BC)`
  have hm23 : cellMass pop T (fun ω t => ¬ solved ω ∧ t < tl ω)
      + cellMass pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t) = T * (1 - pop.prob solved) := by
    simp only [cellMass]
    rw [← Finset.sum_add_distrib]
    have hstep : ∀ t, ((∑ ω, if ¬ solved ω ∧ t < tl ω then pop.P ω else 0)
        + ∑ ω, if ¬ solved ω ∧ tl ω ≤ t then pop.P ω else 0) = pop.prob (fun ω => ¬ solved ω) := by
      intro t
      unfold Population.prob
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun ω _ => ?_
      by_cases hs : solved ω
      · simp [hs]
      · by_cases ht : t < tl ω
        · simp [hs, ht, not_le.2 ht]
        · simp [hs, ht, not_lt.1 ht]
    rw [Finset.sum_congr rfl fun t _ => hstep t, Finset.sum_const, Finset.card_range, nsmul_eq_mul,
      pop.prob_compl]
  have hexact : ∑ t ∈ range T, ∑ ω, pop.P ω * e ω t
      = cellMass pop T (fun ω _ => solved ω) * cellAvg pop T (fun ω _ => solved ω) e
        + cellMass pop T (fun ω t => ¬ solved ω ∧ t < tl ω)
          * cellAvg pop T (fun ω t => ¬ solved ω ∧ t < tl ω) e
        + cellMass pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t)
          * cellAvg pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t) e := by
    rw [htot, h1, h2, h3]
  refine ⟨?_, hexact, hm1, hm23⟩
  rw [hexact]
  exact regime_decomposition_le T (pop.prob solved) _ _ _ _ _ _
    (cellAvg_nonneg pop T _ e he) (cellAvg_nonneg pop T _ e he) (cellAvg_nonneg pop T _ e he)
    (cellMass_nonneg pop T _) (cellMass_nonneg pop T _) hm1.le hm23.le

/-! ### Theorem `bound` (main text) -/

/-- **Theorem `bound`** (`Statements.thm_bound`): Theorem `bound-formal` (1) with
`T ε̄ = ∑_t δ_t`, and Corollary `regimes` divided by `T`. -/
theorem thm_bound : Statements.thm_bound := by
  intro Ω Y _ _ _ _ pop T R solved late _ _ logged tl πT πWD hR hπT hπWD hlen htl hsolved
  have hb := (thm_bound_formal pop T R solved late logged tl πT πWD hR hπT hπWD hlen htl
    hsolved).1
  have hπrelP := relabel_isPolicy solved logged tl πT hπT
  have hreg := (cor_regimes pop T solved tl
    (fun ω t => expectAt (relabel solved logged tl πT ω) t
      (fun h => tv (πWD ω h) (relabel solved logged tl πT ω h)) [])
    (fun ω t => expectAt_nonneg _ (hπrelP ω) _ (fun h => tv_nonneg _ _) t [])).1
  dsimp only at hb hreg ⊢
  set S := ∑ t ∈ range T, ∑ ω, pop.P ω * expectAt (relabel solved logged tl πT ω) t
    (fun h => tv (πWD ω h) (relabel solved logged tl πT ω h)) [] with hS
  refine ⟨?_, ?_⟩
  · -- `T ε̄ = ∑_t δ_t`, also at `T = 0`, where both are `0`
    rcases Nat.eq_zero_or_pos T with hT | hT
    · subst hT
      simpa [hS] using hb
    · have hT' : (T : ℝ) ≠ 0 := by exact_mod_cast hT.ne'
      rw [show (T : ℝ) * (1 / (T : ℝ) * S) = S by field_simp]
      exact hb
  · rcases Nat.eq_zero_or_pos T with hT | hT
    · -- at `T = 0`, `ε̄ = 0` and the right-hand side is nonnegative
      subst hT
      have hρ0 := pop.prob_nonneg solved
      have hρ1 : 0 ≤ 1 - pop.prob solved := by rw [← pop.prob_compl]; exact pop.prob_nonneg _
      have hnn := fun cell [∀ ω t, Decidable (cell ω t)] =>
        cellAvg_nonneg pop 0 cell (fun ω t => expectAt (relabel solved logged tl πT ω) t
          (fun h => tv (πWD ω h) (relabel solved logged tl πT ω h)) [])
          (fun ω t => expectAt_nonneg _ (hπrelP ω) _ (fun h => tv_nonneg _ _) t [])
      simp only [Nat.cast_zero, div_zero, zero_mul]
      have := hnn (fun ω (_ : ℕ) => solved ω)
      have := hnn (fun ω t => ¬ solved ω ∧ t < tl ω)
      have := hnn (fun ω t => ¬ solved ω ∧ tl ω ≤ t)
      positivity
    · have hT' : (0 : ℝ) < T := by exact_mod_cast hT
      rw [one_div, inv_mul_le_iff₀ hT']
      exact hreg

end WebKD
