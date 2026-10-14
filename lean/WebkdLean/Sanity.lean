import WebkdLean.Statements

set_option autoImplicit false
set_option relaxedAutoImplicit false

/-!
# Non-vacuity witnesses (Gate A)

For every statement in `Statements.lean`, a concrete instance that satisfies all of its
hypotheses, chosen to avoid the degenerate case of the conclusion where there is one (an interior
cap, a fatal step after the first step, a non-zero disagreement, a teacher that fails with
probability strictly between `0` and `1`). Contradictory hypotheses would make every conclusion
provable; these instances rule that out.
-/

namespace WebKD.Sanity

open WebKD Finset

/-! ## Lemma `mono`, Proposition `tstar` -/

/-- A line harness on positions `ℕ`: action `true` advances, `false` stays, the reward is `1` at
position `2`, and no action is terminal. -/
def lineH : Harness ℕ Bool where
  step s a := if a then s + 1 else s
  done _ := False
  R s := s = 2
  absorbing _ _ h := h.elim

/-- Advance once, then stay twice (positions `0, 1, 1, 1`); budget `T = 3`. -/
def advanceOnce : Trajectory ℕ Bool where
  s0 := 0
  act | 0 => true | _ => false
  n := 3

/-- Stay three times (positions `0, 0, 0, 0`); budget `T = 3`. -/
def stayPut : Trajectory ℕ Bool where
  s0 := 0
  act _ := false
  n := 3

lemma advanceOnce_failed : advanceOnce.Failed lineH 3 :=
  ⟨le_rfl, Or.inr rfl, by norm_num [Trajectory.state, lineH, advanceOnce]⟩

/-- `ν_0 = 1`: `[true, true, false]` uses up the budget and ends at position `2`. -/
lemma advanceOnce_nu0 : advanceOnce.nu lineH 3 0 :=
  ⟨[true, true, false], by simp, Or.inr rfl, by norm_num [Harness.run, Trajectory.state, lineH, advanceOnce]⟩

/-- Non-degenerate: the fatal step is not the first one (`ν_2 = 1`), so `t⋆ = 3`. -/
lemma advanceOnce_nu2 : advanceOnce.nu lineH 3 2 :=
  ⟨[true], by simp, Or.inr rfl, by norm_num [Harness.run, Trajectory.state, lineH, advanceOnce]⟩

/-- Proposition `tstar`: all hypotheses hold for `advanceOnce`. -/
example : advanceOnce.Failed lineH 3 ∧ advanceOnce.nu lineH 3 0 :=
  ⟨advanceOnce_failed, advanceOnce_nu0⟩

/-- Lemma `mono`: its hypotheses hold at `t = 2` for `stayPut` (`2 + 1 ≤ 3`, `ν_2 = 0`). -/
example : 2 + 1 ≤ 3 ∧ ¬ stayPut.nu lineH 3 2 := by
  refine ⟨le_rfl, ?_⟩
  rintro ⟨seq, hlen, hend, hR⟩
  -- from position `0` with one step left, position `2` is out of reach
  have hstate : stayPut.state lineH 2 = 0 := by norm_num [Trajectory.state, lineH, stayPut]
  simp only [hstate] at *
  rcases seq with _ | ⟨a, _ | ⟨b, rest⟩⟩
  · norm_num [Harness.run, lineH] at hR
  · cases a <;> norm_num [Harness.run, lineH] at hR
  · simp at hlen

/-! ## The anchored objective -/

/-- Proposition `target`: `a = 1/2`, `b = 1`, `q = 1/4` satisfy the hypotheses and the target is
interior (`p⋆ = 1/2`). -/
example : (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 ∧
    pStar (1 / 2) 1 (1 / 4) = 1 / 2 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  norm_num [pStar]

/-- Eq. `kl-grad`, regime (7): `0 < a < b`, `p, q > 0` hold for `a = 1/2`, `b = 1`, `q = 1/4`,
`p = 1/8`, below the stationary point `q b/(b - a) = 1/2`. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 8 ∧ (0 : ℝ) < 1 / 4 ∧
    (1 / 8 : ℝ) < 1 / 4 * 1 / (1 - 1 / 2) := by norm_num

/-- Corollary `beta`: `W = γ = Γ = 1`, `w = 1`, `β = 2`, `q = 1/4` satisfy the hypotheses and the
cap binds (`β̄/(1 - q) = 4/3 < 2`). -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) < 1 ∧
    betaBar 1 1 1 1 / (1 - 1 / 4) < 2 := by
  norm_num [betaBar]

/-- Corollary `budget`, capped case with equality: `a = 1/2 < b = 1`, `q = 1/4`, interior cap
`q b/(b - a) = 1/2 ≤ 1`. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 4 ∧
    (1 / 4 : ℝ) * 1 / (1 - 1 / 2) ≤ 1 := by norm_num

/-! ## The weighted objective -/

/-- Remark `resample`: weights `(1, 3)` on two positions are nonnegative with positive sum. -/
example : (∀ i ∈ (univ : Finset (Fin 2)), (0 : ℝ) ≤ ![1, 3] i) ∧
    (0 : ℝ) < ∑ i ∈ (univ : Finset (Fin 2)), ![(1 : ℝ), 3] i := by
  refine ⟨fun i _ => ?_, ?_⟩
  · fin_cases i <;> norm_num
  · norm_num [Fin.sum_univ_two]

/-- Eq. `actw-map` (2b): a single-segment step satisfies the tie hypothesis. -/
example : (0 : Fin 1) ∈ (univ : Finset (Fin 1)) ∧
    (∀ j ∈ (univ : Finset (Fin 1)), ![(0.3 : ℝ)] j = ![(0.3 : ℝ)] 0) ∧ (1 / 10 : ℝ) ≤ 1 := by
  refine ⟨mem_univ _, fun j _ => ?_, by norm_num⟩
  fin_cases j; rfl

/-- Eq. `actw-map`: two segments with distinct scores, outside the dead zone (`σ = 1 ≥ τ = 1/20`),
`φ = 1/10`; segment `1` is the unique top and segment `0` the unique bottom. -/
example : 2 ≤ Fintype.card (Fin 2) ∧ (0 : ℝ) ≤ 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 20 ∧
    (1 / 20 : ℝ) ≤ 1 ∧ (∀ j : Fin 2, j ≠ 1 → ![(0 : ℝ), 1] j < ![(0 : ℝ), 1] 1) ∧
    (∀ j : Fin 2, j ≠ 0 → ![(0 : ℝ), 1] 0 < ![(0 : ℝ), 1] j) := by
  refine ⟨by simp, by norm_num, by norm_num, by norm_num, by norm_num, ?_, ?_⟩
  · intro j hj; fin_cases j <;> simp_all
  · intro j hj; fin_cases j <;> simp_all

/-! ## Premium split and rank weighting -/

/-- Proposition `split`: `I_rat = {0}`, `I_act = {1}` are disjoint and nonempty; with zero logits
over a two-word vocabulary every `CE_i = log 2 ≤ Λ = 1`. -/
example : Disjoint ({0} : Finset (Fin 2)) {1} ∧ ({0} : Finset (Fin 2)).Nonempty ∧
    ({1} : Finset (Fin 2)).Nonempty ∧
    ∀ i ∈ ({0} : Finset (Fin 2)), nll (0 : Fin 2) (fun _ : Fin 2 => (0 : ℝ)) ≤ 1 := by
  refine ⟨by simp, by simp, by simp, fun i _ => ?_⟩
  have h2 : Real.log 2 < 1 := by
    have := Real.log_two_lt_d9
    linarith
  simp only [nll, softmax, Real.exp_zero, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  rw [show ((2 : ℕ) : ℝ) * 1 = 2 by norm_num, one_div, Real.log_inv, neg_neg]
  exact h2.le

/-- Theorem `rank`: `κ = 1/2 ∈ (0,1)` and the identity assignment `v = Δ` respects the order. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧ Monovary ![(0 : ℝ), 1] ![(0 : ℝ), 1] :=
  ⟨by norm_num, by norm_num, monovary_self _⟩

/-! ## The single-round bound

Two tasks with probability `1/2` each, binary responses, budget `T = 1`, reward `1` exactly for the
response `true`. Task `true`'s logged rollout `[true]` succeeded; task `false`'s logged rollout
`[false]` failed and is localized at `t' = 1`. The teacher and the student both answer uniformly,
so the teacher fails from `h_{t'}` with probability `1/2` and the student disagrees with the
replay with total variation `1/2`. -/

/-- The uniform task distribution on two tasks. -/
noncomputable def pop2 : Population Bool where
  P _ := 1 / 2
  nonneg _ := by norm_num
  total := by norm_num [Fintype.sum_bool]

/-- Reward `1` exactly for the complete response sequence `[true]`. -/
noncomputable def R2 (_ : Bool) (h : List Bool) : ℝ := if h = [true] then 1 else 0

/-- The uniform policy over binary responses. -/
noncomputable def unif : List Bool → Bool → ℝ := fun _ _ => 1 / 2

lemma unif_isPolicy : IsPolicy unif := fun _ => ⟨fun _ => by norm_num [unif], by
  norm_num [unif, Fintype.sum_bool]⟩

lemma R2_bounds : ∀ ω h, 0 ≤ R2 ω h ∧ R2 ω h ≤ 1 := by
  intro ω h
  unfold R2
  split_ifs <;> norm_num

/-- All hypotheses of Lemma `imit`, Theorem `bound-formal` and Theorem `bound` hold, with
`T = 1`, `tl ≡ 1`, `solved ω ↔ ω = true`. -/
example : (∀ ω h, 0 ≤ R2 ω h ∧ R2 ω h ≤ 1) ∧ (∀ _ : Bool, IsPolicy unif) ∧
    (∀ ω : Bool, [ω].length = 1) ∧ (∀ _ : Bool, 1 ≤ 1 ∧ 1 ≤ 1) ∧
    (∀ ω : Bool, ω = true ↔ R2 ω [ω] = 1) := by
  refine ⟨R2_bounds, fun _ => unif_isPolicy, fun _ => rfl, fun _ => ⟨le_rfl, le_rfl⟩, ?_⟩
  intro ω
  cases ω <;> norm_num [R2]

/-- Non-degenerate: the teacher fails from `h_{t'}` with probability `1/2` on the failed task. -/
example : 1 - expectAt unif (1 - (1 - 1)) (R2 false) (([false] : List Bool).take (1 - 1)) = 1 / 2 := by
  norm_num [expectAt, unif, R2, Fintype.sum_bool]

/-- Corollary `regimes`: the constant disagreement `1` is nonnegative. -/
example : ∀ (_ : Bool) (_ : ℕ), (0 : ℝ) ≤ 1 := fun _ _ => zero_le_one

end WebKD.Sanity
