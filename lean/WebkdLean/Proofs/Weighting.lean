import WebkdLean.Statements

set_option autoImplicit false
set_option relaxedAutoImplicit false

/-!
# Action-informative token weighting (Eq. `sw`, Remark `resample`, Eq. `actw-map`)

Strategy. Uniform weights cancel against the weight-sum normalizer (`uniform_weights_recover_sft`),
and `L_SW` is a `q_w`-expectation because dividing by the weight sum distributes over the sum
(`LSW_eq_expectation`). For the rank map: the clip of Eq. `actw-map` never binds for ranks in
`[0, 1]`, so on a ranked step the map is affine in the rank with slope `(1-φ) min(1, σ/τ_dz) ≥ 0`.
Average ranks of a step sum to `n(n+1)/2` whatever the ties (`avgRank_sum`, by pairing each
strict comparison with its reverse), so normalized ranks average `1/2` and span weights average
`(1+φ)/2` (`rankMap_mean_of_ranks`). A strictly larger score has a strictly larger average rank
(`avgRank_lt_of_lt`), so the map assigns weights in the order of `Δ` (`rankMap_monovary`); a
unique top (bottom) segment has normalized rank `1` (`0`) and, outside the dead zone, weight `1`
(`φ`). A tied step has spread `0` and gets the midpoint. Remark `pmi` is only the algebraic
identity `pmi_eq`; that `Δ_s` deserves the name PMI is an interpretation, not a theorem.
-/

namespace WebKD

open Finset

variable {ι : Type}

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
  -- the clip is inactive: `φ ≤ φ + (1 - φ) x ≤ 1` for `x ∈ [0, 1]` and `φ ≤ 1`
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
  -- the `m`-terms cancel because the ranks average `1/2`
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

/-! ### Order, ties and extremes of the rank map (Eq. `actw-map`) -/

/-- With spread `σ = 0` the map returns the midpoint `(1 + φ)/2` whatever the rank. -/
theorem rankMap_zero_spread (φ τ r : ℝ) (hφ1 : φ ≤ 1) : rankMap φ τ 0 r = (1 + φ) / 2 := by
  unfold rankMap
  rw [zero_div, min_eq_right zero_le_one, mul_zero, add_zero]
  rw [min_eq_right (by linarith), max_eq_right (by linarith)]
  ring

/-- A step whose segments all tie (in particular a single-segment step) has spread `0`. -/
theorem spread_eq_zero_of_tie (s : Finset ι) (Δ : ι → ℝ) (i : ι) (hi : i ∈ s)
    (htie : ∀ j ∈ s, Δ j = Δ i) : spread s Δ = 0 := by
  have hne : s.Nonempty := ⟨i, hi⟩
  have hsup : s.sup' hne Δ = Δ i :=
    le_antisymm (Finset.sup'_le hne Δ fun j hj => (htie j hj).le) (Finset.le_sup' Δ hi)
  have hinf : s.inf' hne Δ = Δ i :=
    le_antisymm (Finset.inf'_le Δ hi) (Finset.le_inf' hne Δ fun j hj => (htie j hj).ge)
  simp only [spread, hne, ↓reduceDIte, hsup, hinf, sub_self]

/-- The rank map is monotone in the normalized rank (its slope `(1-φ) min(1, σ/τ)` is
nonnegative, and the clip is monotone). -/
lemma rankMap_mono_rank (φ τ σ : ℝ) (hφ1 : φ ≤ 1) (hm : 0 ≤ min 1 (σ / τ)) {r₁ r₂ : ℝ}
    (h : r₁ ≤ r₂) : rankMap φ τ σ r₁ ≤ rankMap φ τ σ r₂ := by
  unfold rankMap
  apply max_le_max le_rfl
  apply min_le_min le_rfl
  have hslope : (r₁ - 1 / 2) * min 1 (σ / τ) ≤ (r₂ - 1 / 2) * min 1 (σ / τ) :=
    mul_le_mul_of_nonneg_right (by linarith) hm
  have := mul_le_mul_of_nonneg_left hslope (sub_nonneg.2 hφ1)
  linarith

/-- **A strictly larger score has a strictly larger average rank.** Every segment scoring at most
`Δ i` scores strictly below `Δ j`, and `j` ties with itself, so
`rank_j ≥ #{< Δ i} + #{= Δ i} + 1 > #{< Δ i} + (#{= Δ i} + 1)/2 = rank_i`. -/
lemma avgRank_lt_of_lt (s : Finset ι) (Δ : ι → ℝ) {i j : ι} (hj : j ∈ s) (h : Δ i < Δ j) :
    avgRank s Δ i < avgRank s Δ j := by
  classical
  have hsub : (s.filter fun k => Δ k < Δ i) ∪ (s.filter fun k => Δ k = Δ i)
      ⊆ s.filter fun k => Δ k < Δ j := by
    intro k hk
    simp only [Finset.mem_union, Finset.mem_filter] at hk ⊢
    rcases hk with ⟨hks, hk⟩ | ⟨hks, hk⟩
    · exact ⟨hks, lt_trans hk h⟩
    · exact ⟨hks, hk ▸ h⟩
  have hdisj : Disjoint (s.filter fun k => Δ k < Δ i) (s.filter fun k => Δ k = Δ i) := by
    rw [Finset.disjoint_filter]
    intro k _ h1 h2
    exact (ne_of_lt h1) h2
  have hcard : ((s.filter fun k => Δ k < Δ i).card : ℝ) + (s.filter fun k => Δ k = Δ i).card
      ≤ (s.filter fun k => Δ k < Δ j).card := by
    have := Finset.card_le_card hsub
    rw [Finset.card_union_of_disjoint hdisj] at this
    exact_mod_cast this
  have hEj : (1 : ℝ) ≤ (s.filter fun k => Δ k = Δ j).card := by
    have : 0 < (s.filter fun k => Δ k = Δ j).card :=
      Finset.card_pos.2 ⟨j, Finset.mem_filter.2 ⟨hj, rfl⟩⟩
    exact_mod_cast this
  have hEi : (0 : ℝ) ≤ (s.filter fun k => Δ k = Δ i).card := by positivity
  unfold avgRank
  linarith

/-- **Eq. `actw-map` assigns weights in the order of `Δ`.** -/
theorem rankMap_monovary [Fintype ι] (Δ : ι → ℝ) (φ τ σ : ℝ) (h2 : 2 ≤ Fintype.card ι)
    (hφ1 : φ ≤ 1) (hτ : 0 < τ) (hσ : 0 ≤ σ) :
    Monovary (fun i => rankMap φ τ σ (normRank Finset.univ Δ i)) Δ := by
  intro i j hij
  have hpos : (0 : ℝ) < (Finset.univ : Finset ι).card - 1 := by
    have : (2 : ℝ) ≤ (Finset.univ : Finset ι).card := by
      rw [Finset.card_univ]; exact_mod_cast h2
    linarith
  have hr : normRank Finset.univ Δ i ≤ normRank Finset.univ Δ j := by
    unfold normRank
    apply div_le_div_of_nonneg_right _ hpos.le
    linarith [avgRank_lt_of_lt Finset.univ Δ (Finset.mem_univ j) hij]
  exact rankMap_mono_rank φ τ σ hφ1 (le_min zero_le_one (div_nonneg hσ hτ.le)) hr

/-- A unique top-scoring segment has average rank `n`, hence normalized rank `1`. -/
lemma normRank_top [Fintype ι] (Δ : ι → ℝ) (i : ι) (h2 : 2 ≤ Fintype.card ι)
    (htop : ∀ j, j ≠ i → Δ j < Δ i) : normRank Finset.univ Δ i = 1 := by
  classical
  have hlt : (Finset.univ.filter fun k => Δ k < Δ i) = Finset.univ.erase i := by
    ext k
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_erase, ne_eq, and_true]
    constructor
    · intro hk hki; rw [hki] at hk; exact lt_irrefl _ hk
    · intro hki; exact htop k hki
  have heq : (Finset.univ.filter fun k => Δ k = Δ i) = {i} := by
    ext k
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
    constructor
    · intro hk; by_contra hki; exact (ne_of_lt (htop k hki)) hk
    · intro hki; rw [hki]
  have hc : 1 ≤ (Finset.univ : Finset ι).card := by rw [Finset.card_univ]; omega
  have hpos : (0 : ℝ) < (Finset.univ : Finset ι).card - 1 := by
    have : (2 : ℝ) ≤ (Finset.univ : Finset ι).card := by
      rw [Finset.card_univ]; exact_mod_cast h2
    linarith
  unfold normRank avgRank
  rw [hlt, heq, Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_singleton,
    Nat.cast_sub hc]
  field_simp
  ring

/-- A unique bottom-scoring segment has average rank `1`, hence normalized rank `0`. -/
lemma normRank_bottom [Fintype ι] (Δ : ι → ℝ) (i : ι)
    (hbot : ∀ j, j ≠ i → Δ i < Δ j) : normRank Finset.univ Δ i = 0 := by
  classical
  have hlt : (Finset.univ.filter fun k => Δ k < Δ i) = ∅ := by
    ext k
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty, iff_false]
    intro hk
    by_cases hki : k = i
    · rw [hki] at hk; exact lt_irrefl _ hk
    · exact lt_asymm hk (hbot k hki)
  have heq : (Finset.univ.filter fun k => Δ k = Δ i) = {i} := by
    ext k
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
    constructor
    · intro hk; by_contra hki; exact (ne_of_lt (hbot k hki)).symm hk
    · intro hki; rw [hki]
  unfold normRank avgRank
  rw [hlt, heq, Finset.card_empty, Finset.card_singleton]
  norm_num

/-- Outside the dead zone (`σ ≥ τ_dz`) a unique top segment keeps full weight `1`. -/
theorem rankMap_top [Fintype ι] (Δ : ι → ℝ) (φ τ σ : ℝ) (i : ι) (h2 : 2 ≤ Fintype.card ι)
    (hφ1 : φ ≤ 1) (hτ : 0 < τ) (hστ : τ ≤ σ) (htop : ∀ j, j ≠ i → Δ j < Δ i) :
    rankMap φ τ σ (normRank Finset.univ Δ i) = 1 := by
  rw [normRank_top Δ i h2 htop]
  unfold rankMap
  rw [min_eq_left ((one_le_div hτ).2 hστ)]
  rw [show φ + (1 - φ) * (1 / 2 + (1 - 1 / 2) * 1) = 1 by ring, min_self, max_eq_right hφ1]

/-- Outside the dead zone (`σ ≥ τ_dz`) a unique bottom segment drops to the floor `φ`. -/
theorem rankMap_bottom [Fintype ι] (Δ : ι → ℝ) (φ τ σ : ℝ) (i : ι)
    (hφ1 : φ ≤ 1) (hτ : 0 < τ) (hστ : τ ≤ σ) (hbot : ∀ j, j ≠ i → Δ i < Δ j) :
    rankMap φ τ σ (normRank Finset.univ Δ i) = φ := by
  rw [normRank_bottom Δ i hbot]
  unfold rankMap
  rw [min_eq_left ((one_le_div hτ).2 hστ)]
  rw [show φ + (1 - φ) * (1 / 2 + (0 - 1 / 2) * 1) = φ by ring, min_eq_right hφ1, max_self]

/-! ### The statements -/

/-- **Uniform weights recover SFT** (`Statements.eq_sw`). -/
theorem eq_sw : Statements.eq_sw := by
  intro ι T CE c hc
  exact uniform_weights_recover_sft T CE c hc

/-- **Remark `resample`** (`Statements.rem_resample`). -/
theorem rem_resample : Statements.rem_resample := by
  intro ι T w CE hw hW
  exact ⟨fun i hi => div_nonneg (hw i hi) hW.le, qw_sum_eq_one T w hW.ne',
    LSW_eq_expectation T w CE⟩

/-- **Eq. `actw-map`** (`Statements.eq_actw_map`). -/
theorem eq_actw_map : Statements.eq_actw_map := by
  refine ⟨fun φ τ σ r hφ => rankMap_mem φ τ σ r hφ, fun φ τ r hφ => rankMap_zero_spread φ τ r hφ,
    ?_, ?_, ?_, ?_, ?_⟩
  · intro ι s Δ φ τ i hi htie hφ1
    rw [spread_eq_zero_of_tie s Δ i hi htie]
    exact rankMap_zero_spread φ τ _ hφ1
  · intro ι s Δ φ τ σ h2 hφ0 hφ1 hτ hσ
    exact rankMap_mean_of_ranks s Δ h2 φ τ σ hφ0 hφ1 hτ hσ
  · intro ι _ Δ φ τ σ h2 _ hφ1 hτ hσ
    exact rankMap_monovary Δ φ τ σ h2 hφ1 hτ hσ
  · intro ι _ Δ φ τ σ i h2 _ hφ1 hτ hστ htop
    exact rankMap_top Δ φ τ σ i h2 hφ1 hτ hστ htop
  · intro ι _ Δ φ τ σ i _ _ hφ1 hτ hστ hbot
    exact rankMap_bottom Δ φ τ σ i hφ1 hτ hστ hbot

end WebKD
