import Mathlib

/-!
# Action-informative token weighting

Formalizes Proposition `scale` (uniform weights recover SFT exactly), Remark `resample`
(`L_SW` is an expectation under `q_w`), and the properties of the rank map of Eq. `actw-map`:
its range, the dead-zone value, and the fact that the dead-zone value is exactly the mean weight
of a ranked step (including the fact, previously a hypothesis, that within-step average ranks
normalized to `[0,1]` have mean `1/2`, ties sharing their mean rank). Remark `pmi` is the
identity `pmi_eq`. It also records, as a counterexample, that a dead-zone step is *not* "exactly
no reweighting" once the pinned action tokens (weight `1`) are counted, which corrects a sentence
of the paper.
-/

namespace WebKD

open Finset

variable {ι : Type*}

/-- Eq. `sw`: the weighted-mean imitation term `L_SW = ∑ w_i CE_i / ∑ w_i`. -/
noncomputable def LSW (T : Finset ι) (w CE : ι → ℝ) : ℝ :=
  (∑ i ∈ T, w i * CE i) / (∑ i ∈ T, w i)

/-- The per-token-mean SFT loss on the supervised tokens `T`. -/
noncomputable def LSFT (T : Finset ι) (CE : ι → ℝ) : ℝ :=
  (1 / (T.card : ℝ)) * ∑ i ∈ T, CE i

/-- **Proposition `scale`.** For every constant `c > 0`, `w_i ≡ c` gives `L_SW = L_SFT`
identically (hence equal gradients). -/
theorem uniform_weights_recover_sft (T : Finset ι) (CE : ι → ℝ) (c : ℝ) (hc : 0 < c) :
    LSW T (fun _ => c) CE = LSFT T CE := by
  unfold LSW LSFT
  rw [← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul]
  rcases T.eq_empty_or_nonempty with h | h
  · simp [h]
  · have hcard : (0 : ℝ) < T.card := by exact_mod_cast Finset.card_pos.2 h
    field_simp

/-- **Remark `resample`.** `L_SW = E_{i ∼ q_w}[CE_i]` with `q_w(i) = w_i / ∑_j w_j`. -/
theorem LSW_eq_expectation (T : Finset ι) (w CE : ι → ℝ) :
    LSW T w CE = ∑ i ∈ T, (w i / ∑ j ∈ T, w j) * CE i := by
  unfold LSW
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

/-- `q_w` is a probability distribution over the supervised positions. -/
theorem qw_sum_eq_one (T : Finset ι) (w : ι → ℝ) (h : ∑ j ∈ T, w j ≠ 0) :
    ∑ i ∈ T, w i / ∑ j ∈ T, w j = 1 := by
  rw [← Finset.sum_div, div_self h]

/-- Eq. `actw-map`: `v_s = clip_{[φ,1]}(φ + (1-φ)(1/2 + (r_s - 1/2) min(1, σ/τ)))`. -/
noncomputable def rankMap (φ τ σ r : ℝ) : ℝ :=
  max φ (min 1 (φ + (1 - φ) * (1 / 2 + (r - 1 / 2) * min 1 (σ / τ))))

/-- The span weight lies in `[φ, 1]`. -/
lemma rankMap_mem (φ τ σ r : ℝ) (hφ : φ ≤ 1) :
    φ ≤ rankMap φ τ σ r ∧ rankMap φ τ σ r ≤ 1 := by
  unfold rankMap
  exact ⟨le_max_left _ _, max_le hφ (min_le_left _ _)⟩

/-- For `r ∈ [0,1]`, `σ ≥ 0`, `τ > 0`, `0 ≤ φ ≤ 1` the clip never binds. -/
lemma rankMap_eq_inner (φ τ σ r : ℝ) (_hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) (hτ : 0 < τ) (hσ : 0 ≤ σ)
    (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    rankMap φ τ σ r = φ + (1 - φ) * (1 / 2 + (r - 1 / 2) * min 1 (σ / τ)) := by
  have hm0 : 0 ≤ min 1 (σ / τ) := le_min zero_le_one (div_nonneg hσ hτ.le)
  have hm1 : min 1 (σ / τ) ≤ 1 := min_le_left _ _
  have hinner0 : 0 ≤ 1 / 2 + (r - 1 / 2) * min 1 (σ / τ) := by
    nlinarith [mul_nonneg hr0 hm0]
  have hinner1 : 1 / 2 + (r - 1 / 2) * min 1 (σ / τ) ≤ 1 := by
    nlinarith [mul_nonneg (sub_nonneg.2 hr1) hm0]
  unfold rankMap
  rw [min_eq_right, max_eq_right] <;> nlinarith

/-- **Dead zone.** A step with within-step spread `σ = 0` (a single segment, or statistically
indistinguishable segments) receives the midpoint weight `(1 + φ)/2` on every segment, not `1`. -/
theorem rankMap_dead_zone (φ τ r : ℝ) (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) (hτ : 0 < τ)
    (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    rankMap φ τ 0 r = (1 + φ) / 2 := by
  rw [rankMap_eq_inner φ τ 0 r hφ0 hφ1 hτ le_rfl hr0 hr1]
  simp
  ring

/-- The mean span weight of a ranked step whose normalized average ranks have mean `1/2`
(which is what average ranks normalized to `[0,1]` satisfy) is exactly `(1 + φ)/2`, for every
spread `σ ≥ 0`. Hence a dead-zone step is not down-weighted relative to other steps. -/
theorem rankMap_mean (s : Finset ι) (hs : s.Nonempty) (r : ι → ℝ) (φ τ σ : ℝ)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) (hτ : 0 < τ) (hσ : 0 ≤ σ)
    (hr : ∀ i ∈ s, 0 ≤ r i ∧ r i ≤ 1)
    (hmean : (∑ i ∈ s, r i) / s.card = 1 / 2) :
    (∑ i ∈ s, rankMap φ τ σ (r i)) / s.card = (1 + φ) / 2 := by
  have hcard : (0 : ℝ) < s.card := by exact_mod_cast Finset.card_pos.2 hs
  have hsum : ∑ i ∈ s, r i = s.card / 2 := by
    rw [div_eq_iff hcard.ne'] at hmean
    rw [hmean]; ring
  set m := min 1 (σ / τ) with hm
  have hrw : ∀ i ∈ s, rankMap φ τ σ (r i)
      = (φ + (1 - φ) * (1 / 2 - m / 2)) + ((1 - φ) * m) * r i := by
    intro i hi
    rw [rankMap_eq_inner φ τ σ (r i) hφ0 hφ1 hτ hσ (hr i hi).1 (hr i hi).2, ← hm]
    ring
  rw [Finset.sum_congr rfl hrw, Finset.sum_add_distrib, Finset.sum_const, ← Finset.mul_sum,
    hsum, nsmul_eq_mul]
  field_simp
  ring

/-- **Correction to the paper.** A dead-zone step is *not* "exactly no reweighting" once the
pinned tokens are counted: with one rationale token at the midpoint weight `(1 + φ)/2` and one
action token pinned at `1`, `L_SW` differs from the SFT mean on that step (here `φ = 0.1`,
`CE = (1, 0)`). Only weights that are constant across *all* supervised tokens of the step
(Proposition `scale`) reproduce SFT. -/
theorem dead_zone_step_not_sft :
    LSW (Finset.univ : Finset (Fin 2)) ![(1 + 1 / 10) / 2, 1] ![1, 0]
      ≠ LSFT (Finset.univ : Finset (Fin 2)) ![1, 0] := by
  simp [LSW, LSFT, Fin.sum_univ_two]
  norm_num

/-- **Remark `pmi`.** `Δ_s = log P(A⋆ | s, c) - log P(A⋆ | c) = log (P(A⋆ | s, c) / P(A⋆ | c))`,
the pointwise mutual information between the segment and the decision given the rest. -/
theorem pmi_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    Real.log x - Real.log y = Real.log (x / y) := by
  rw [Real.log_div hx.ne' hy.ne']

/-! ### Average ranks normalized to `[0,1]` have mean `1/2` -/

/-- The within-step average rank of segment `i` among the segments `s` scored by `Δ`:
`1`-indexed, ties sharing their mean rank. It equals `#{j : Δ_j < Δ_i} + (#{j : Δ_j = Δ_i} + 1)/2`,
which is the mean of the positions a tie group occupies. -/
noncomputable def avgRank (s : Finset ι) (Δ : ι → ℝ) (i : ι) : ℝ :=
  ((s.filter fun j => Δ j < Δ i).card : ℝ) + (((s.filter fun j => Δ j = Δ i).card : ℝ) + 1) / 2

lemma card_filter_lt_eq_sum (s : Finset ι) (Δ : ι → ℝ) (i : ι) :
    ((s.filter fun j => Δ j < Δ i).card : ℝ) = ∑ j ∈ s, if Δ j < Δ i then (1 : ℝ) else 0 := by
  rw [Finset.card_filter]
  push_cast
  rfl

lemma card_filter_eq_eq_sum (s : Finset ι) (Δ : ι → ℝ) (i : ι) :
    ((s.filter fun j => Δ j = Δ i).card : ℝ) = ∑ j ∈ s, if Δ j = Δ i then (1 : ℝ) else 0 := by
  rw [Finset.card_filter]
  push_cast
  rfl

/-- Trichotomy as indicators: for every pair exactly one of `Δ_j < Δ_i`, `Δ_i < Δ_j`,
`Δ_j = Δ_i` holds. -/
lemma indicator_trichotomy (Δ : ι → ℝ) (i j : ι) :
    (if Δ j < Δ i then (1 : ℝ) else 0) + (if Δ i < Δ j then (1 : ℝ) else 0)
      + (if Δ j = Δ i then (1 : ℝ) else 0) = 1 := by
  rcases lt_trichotomy (Δ j) (Δ i) with h | h | h
  · simp [h, h.ne, not_lt.2 h.le]
  · simp [h]
  · simp [h, h.ne', not_lt.2 h.le]

/-- The average ranks of a step sum to `n (n + 1) / 2` whatever the ties. -/
theorem avgRank_sum (s : Finset ι) (Δ : ι → ℝ) :
    ∑ i ∈ s, avgRank s Δ i = (s.card : ℝ) * (s.card + 1) / 2 := by
  have hsym : ∑ i ∈ s, ∑ j ∈ s, (if Δ j < Δ i then (1 : ℝ) else 0)
      = ∑ i ∈ s, ∑ j ∈ s, (if Δ i < Δ j then (1 : ℝ) else 0) := Finset.sum_comm
  have htot : ∑ i ∈ s, ∑ j ∈ s, ((if Δ j < Δ i then (1 : ℝ) else 0)
      + (if Δ i < Δ j then (1 : ℝ) else 0) + (if Δ j = Δ i then (1 : ℝ) else 0))
      = (s.card : ℝ) * s.card := by
    simp_rw [indicator_trichotomy]
    simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
  simp_rw [Finset.sum_add_distrib] at htot
  rw [← hsym] at htot
  unfold avgRank
  simp_rw [card_filter_lt_eq_sum, card_filter_eq_eq_sum]
  rw [Finset.sum_add_distrib]
  have hhalf : ∑ i ∈ s, ((∑ j ∈ s, if Δ j = Δ i then (1 : ℝ) else 0) + 1) / 2
      = ((∑ i ∈ s, ∑ j ∈ s, if Δ j = Δ i then (1 : ℝ) else 0) + s.card) / 2 := by
    rw [← Finset.sum_div, Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, mul_one]
  rw [hhalf]
  linarith [htot]

/-- Every average rank of a step lies in `[1, n]`. -/
theorem avgRank_bounds (s : Finset ι) (Δ : ι → ℝ) (i : ι) (hi : i ∈ s) :
    1 ≤ avgRank s Δ i ∧ avgRank s Δ i ≤ s.card := by
  have hE1 : 1 ≤ ((s.filter fun j => Δ j = Δ i).card : ℝ) := by
    have : 0 < (s.filter fun j => Δ j = Δ i).card :=
      Finset.card_pos.2 ⟨i, Finset.mem_filter.2 ⟨hi, rfl⟩⟩
    exact_mod_cast this
  have hL0 : (0 : ℝ) ≤ (s.filter fun j => Δ j < Δ i).card := by positivity
  have hLE : ((s.filter fun j => Δ j < Δ i).card : ℝ)
      + ((s.filter fun j => Δ j = Δ i).card : ℝ) ≤ s.card := by
    rw [card_filter_lt_eq_sum, card_filter_eq_eq_sum, ← Finset.sum_add_distrib]
    calc ∑ j ∈ s, ((if Δ j < Δ i then (1 : ℝ) else 0) + if Δ j = Δ i then (1 : ℝ) else 0)
        ≤ ∑ j ∈ s, (1 : ℝ) := by
          apply Finset.sum_le_sum
          intro j _
          rcases lt_trichotomy (Δ j) (Δ i) with h | h | h
          · simp [h, h.ne]
          · simp [h]
          · simp [h.ne', not_lt.2 h.le]
      _ = s.card := by simp
  unfold avgRank
  constructor <;> linarith

/-- Average ranks normalized to `[0,1]`: `r_s = (rank - 1) / (n - 1)`. -/
noncomputable def normRank (s : Finset ι) (Δ : ι → ℝ) (i : ι) : ℝ :=
  (avgRank s Δ i - 1) / (s.card - 1)

/-- For a step with at least two segments, every normalized rank lies in `[0,1]`. -/
theorem normRank_mem (s : Finset ι) (Δ : ι → ℝ) (i : ι) (hi : i ∈ s) (h2 : 2 ≤ s.card) :
    0 ≤ normRank s Δ i ∧ normRank s Δ i ≤ 1 := by
  obtain ⟨h1, hn⟩ := avgRank_bounds s Δ i hi
  have hpos : (0 : ℝ) < s.card - 1 := by
    have : (2 : ℝ) ≤ s.card := by exact_mod_cast h2
    linarith
  unfold normRank
  constructor
  · exact div_nonneg (by linarith) hpos.le
  · rw [div_le_one hpos]
    linarith

/-- **Average ranks normalized to `[0,1]` have mean `1/2`** (the parenthetical claim of
Eq. `actw-map`'s dead-zone discussion), for every step with at least two segments and whatever
the ties. -/
theorem normRank_mean (s : Finset ι) (Δ : ι → ℝ) (h2 : 2 ≤ s.card) :
    (∑ i ∈ s, normRank s Δ i) / s.card = 1 / 2 := by
  have hpos : (0 : ℝ) < s.card - 1 := by
    have : (2 : ℝ) ≤ s.card := by exact_mod_cast h2
    linarith
  have hcard : (0 : ℝ) < s.card := by linarith
  have hne1 : (s.card : ℝ) - 1 ≠ 0 := hpos.ne'
  have hne : (s.card : ℝ) ≠ 0 := hcard.ne'
  unfold normRank
  rw [← Finset.sum_div, Finset.sum_sub_distrib, avgRank_sum, Finset.sum_const, nsmul_eq_mul,
    mul_one]
  field_simp
  ring

/-- **Mean span weight of a ranked step.** With the ranks of Eq. `actw-map` computed as
normalized average ranks, the mean span weight of every step with at least two segments is
`(1 + φ)/2`, for every spread `σ ≥ 0`. This discharges the mean-rank hypothesis of
`rankMap_mean`. -/
theorem rankMap_mean_of_ranks (s : Finset ι) (Δ : ι → ℝ) (h2 : 2 ≤ s.card) (φ τ σ : ℝ)
    (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) (hτ : 0 < τ) (hσ : 0 ≤ σ) :
    (∑ i ∈ s, rankMap φ τ σ (normRank s Δ i)) / s.card = (1 + φ) / 2 :=
  rankMap_mean s (Finset.card_pos.1 (by omega)) (normRank s Δ) φ τ σ hφ0 hφ1 hτ hσ
    (fun i hi => normRank_mem s Δ i hi h2) (normRank_mean s Δ h2)

end WebKD
