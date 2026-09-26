import Mathlib

/-!
# The single-round relabeling bound (Theorem `bound`)

The comparator step of the proof of Theorem `bound-formal` is formalized on a finite population
of episodes: the relabeled process fails exactly on seed failures that are either localized late
or that the teacher-as-agent cannot repair, and its failure probability is
`(1 - ρ_BC)(ε_loc + (1 - ε_loc) ε_corr)`. The imitation step (Lemma `imit`, a maximal-coupling
argument on the deterministic harness) is taken as a hypothesis, and the bound and Corollary
`regimes` are assembled from the two. `single_round_bound_population` composes the comparator
identity with the imitation hypothesis on one population, so Lemma `imit` is the only
hypothesis left, and `regime_decomposition_le` states Corollary `regimes` with each cell's step
mass bounded by `T` per episode ("at most `T` steps") rather than fixed.
-/

namespace WebKD

open Finset

/-- A finite population of episodes with a probability mass function. -/
structure Population (Ω : Type*) [Fintype Ω] where
  P : Ω → ℝ
  nonneg : ∀ ω, 0 ≤ P ω
  total : ∑ ω, P ω = 1

variable {Ω : Type*} [Fintype Ω] (pop : Population Ω)

/-- Probability of an event. -/
noncomputable def Population.prob (E : Ω → Prop) [DecidablePred E] : ℝ :=
  ∑ ω, if E ω then pop.P ω else 0

/-- Conditional probability `P(E | F)` (with the convention `x / 0 = 0`). -/
noncomputable def Population.cond (E F : Ω → Prop) [DecidablePred E] [DecidablePred F] : ℝ :=
  pop.prob (fun ω => E ω ∧ F ω) / pop.prob F

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

end WebKD
