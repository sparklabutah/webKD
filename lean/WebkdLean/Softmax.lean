import Mathlib

/-!
# Softmax cross-entropy and Lemma `sandwich`

Formalizes the logit-gradient identity `∇_z CE = p - e_y` (coordinatewise, as a derivative
along each logit) and Lemma `sandwich`: `1 - e^{-CE} ≤ ‖∇_z CE‖₂ ≤ √2 (1 - e^{-CE})`.
-/

namespace WebKD

open Finset

variable {ι : Type*} [Fintype ι]

/-- Softmax of the logits `z`. -/
noncomputable def softmax (z : ι → ℝ) (v : ι) : ℝ := Real.exp (z v) / ∑ u, Real.exp (z u)

lemma sum_exp_pos [Nonempty ι] (z : ι → ℝ) : 0 < ∑ u, Real.exp (z u) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

lemma softmax_pos [Nonempty ι] (z : ι → ℝ) (v : ι) : 0 < softmax z v :=
  div_pos (Real.exp_pos _) (sum_exp_pos z)

lemma softmax_nonneg [Nonempty ι] (z : ι → ℝ) (v : ι) : 0 ≤ softmax z v :=
  (softmax_pos z v).le

lemma softmax_sum [Nonempty ι] (z : ι → ℝ) : ∑ v, softmax z v = 1 := by
  unfold softmax
  rw [← Finset.sum_div, div_self (sum_exp_pos z).ne']

/-- Per-token cross-entropy `CE = -log softmax(z)_y`. -/
noncomputable def nll (y : ι) (z : ι → ℝ) : ℝ := -Real.log (softmax z y)

lemma nll_eq [Nonempty ι] (y : ι) (z : ι → ℝ) :
    nll y z = -z y + Real.log (∑ u, Real.exp (z u)) := by
  unfold nll softmax
  rw [Real.log_div (Real.exp_pos _).ne' (sum_exp_pos z).ne', Real.log_exp]
  ring

/-- `1 - e^{-CE} = 1 - p_y`. -/
lemma one_sub_exp_neg_nll [Nonempty ι] (y : ι) (z : ι → ℝ) :
    1 - Real.exp (-nll y z) = 1 - softmax z y := by
  unfold nll
  rw [neg_neg, Real.exp_log (softmax_pos z y)]

variable [DecidableEq ι]

/-- Derivative of the partition function along logit `v`. -/
lemma hasDerivAt_sum_exp_update (v : ι) (z : ι → ℝ) :
    HasDerivAt (fun t => ∑ u, Real.exp (Function.update z v t u)) (Real.exp (z v)) (z v) := by
  have h : ∀ u ∈ (Finset.univ : Finset ι),
      HasDerivAt (fun t => Real.exp (Function.update z v t u))
        (if u = v then Real.exp (z v) else 0) (z v) := by
    intro u _
    by_cases huv : u = v
    · subst huv
      simp only [Function.update_self, ite_true]
      exact Real.hasDerivAt_exp (z u)
    · simp only [Function.update_of_ne huv, huv, ite_false]
      exact hasDerivAt_const _ _
  have hfun : (fun t => ∑ u, Real.exp (Function.update z v t u))
      = ∑ u, (fun t => Real.exp (Function.update z v t u)) := by
    funext t
    simp only [Finset.sum_apply]
  have hder : (∑ u, if u = v then Real.exp (z v) else 0) = Real.exp (z v) := by simp
  rw [hfun, ← hder]
  exact HasDerivAt.sum h

/-- **Logit gradient of the cross-entropy.** `∂ CE / ∂ z_v = softmax(z)_v - 𝟙[v = y]`, i.e.
`∇_z CE = p - e_y`. -/
theorem hasDerivAt_nll_coord [Nonempty ι] (y v : ι) (z : ι → ℝ) :
    HasDerivAt (fun t => nll y (Function.update z v t))
      (softmax z v - if v = y then 1 else 0) (z v) := by
  have hf : (fun t => nll y (Function.update z v t))
      = (fun t => -(Function.update z v t y))
        + fun t => Real.log (∑ u, Real.exp (Function.update z v t u)) := by
    funext t
    simp only [Pi.add_apply]
    exact nll_eq y _
  rw [hf]
  have hlog : HasDerivAt (fun t => Real.log (∑ u, Real.exp (Function.update z v t u)))
      (Real.exp (z v) / ∑ u, Real.exp (z u)) (z v) := by
    have := (hasDerivAt_sum_exp_update v z).log (by
      rw [Function.update_eq_self]; exact (sum_exp_pos z).ne')
    rwa [Function.update_eq_self] at this
  by_cases hvy : v = y
  · subst hvy
    have h1 : HasDerivAt (fun t => -(Function.update z v t v)) (-1) (z v) := by
      simp only [Function.update_self]
      exact (hasDerivAt_id (z v)).neg
    refine (h1.add hlog).congr_deriv ?_
    rw [show (if v = v then (1 : ℝ) else 0) = 1 by simp]
    unfold softmax
    ring
  · have h1 : HasDerivAt (fun t => -(Function.update z v t y)) (-0) (z v) := by
      have hne : y ≠ v := fun h => hvy h.symm
      simp only [Function.update_of_ne hne]
      exact (hasDerivAt_const (z v) (z y)).neg
    refine (h1.add hlog).congr_deriv ?_
    rw [show (if v = y then (1 : ℝ) else 0) = 0 by simp [hvy]]
    unfold softmax
    ring

/-- **Lemma `sandwich`, squared form.** For a probability vector `p` and target `y`,
`(1 - p_y)² ≤ ‖p - e_y‖² ≤ 2 (1 - p_y)²`. -/
theorem grad_sq_bounds (p : ι → ℝ) (hp : ∀ v, 0 ≤ p v) (hsum : ∑ v, p v = 1) (y : ι) :
    (1 - p y) ^ 2 ≤ ∑ v, (p v - if v = y then 1 else 0) ^ 2 ∧
    ∑ v, (p v - if v = y then 1 else 0) ^ 2 ≤ 2 * (1 - p y) ^ 2 := by
  have hsplit : ∑ v, (p v - if v = y then 1 else 0) ^ 2
      = (1 - p y) ^ 2 + ∑ v ∈ Finset.univ.erase y, (p v) ^ 2 := by
    rw [← Finset.add_sum_erase Finset.univ _ (Finset.mem_univ y)]
    simp only [ite_true]
    congr 1
    · ring
    · apply Finset.sum_congr rfl
      intro v hv
      rw [Finset.mem_erase] at hv
      simp [hv.1]
  have hrest_sum : ∑ v ∈ Finset.univ.erase y, p v = 1 - p y := by
    rw [Finset.sum_erase_eq_sub (Finset.mem_univ y), hsum]
  have hrest_nonneg : 0 ≤ ∑ v ∈ Finset.univ.erase y, (p v) ^ 2 :=
    Finset.sum_nonneg fun v _ => sq_nonneg _
  have hrest_le : ∑ v ∈ Finset.univ.erase y, (p v) ^ 2 ≤ (1 - p y) ^ 2 := by
    rw [← hrest_sum]
    exact Finset.sum_sq_le_sq_sum_of_nonneg fun v _ => hp v
  rw [hsplit]
  constructor <;> linarith

/-- **Lemma `sandwich`.** `1 - p_y ≤ ‖p - e_y‖₂ ≤ √2 (1 - p_y)`; with `p = softmax z` and
`CE = -log p_y` this is `1 - e^{-CE} ≤ ‖∇_z CE‖₂ ≤ √2 (1 - e^{-CE})`. -/
theorem grad_norm_bounds (p : ι → ℝ) (hp : ∀ v, 0 ≤ p v) (hsum : ∑ v, p v = 1) (y : ι) :
    1 - p y ≤ Real.sqrt (∑ v, (p v - if v = y then 1 else 0) ^ 2) ∧
    Real.sqrt (∑ v, (p v - if v = y then 1 else 0) ^ 2) ≤ Real.sqrt 2 * (1 - p y) := by
  have hpy : p y ≤ 1 := by
    have := Finset.single_le_sum (fun v _ => hp v) (Finset.mem_univ y)
    rwa [hsum] at this
  have h0 : 0 ≤ 1 - p y := by linarith
  obtain ⟨h1, h2⟩ := grad_sq_bounds p hp hsum y
  constructor
  · rw [Real.le_sqrt h0 (Finset.sum_nonneg fun v _ => sq_nonneg _)]
    exact h1
  · rw [← Real.sqrt_sq h0, ← Real.sqrt_mul (by norm_num)]
    exact Real.sqrt_le_sqrt h2

/-- Lemma `sandwich` stated for the softmax cross-entropy at logits `z`. -/
theorem sandwich [Nonempty ι] (y : ι) (z : ι → ℝ) :
    1 - Real.exp (-nll y z)
        ≤ Real.sqrt (∑ v, (softmax z v - if v = y then 1 else 0) ^ 2) ∧
    Real.sqrt (∑ v, (softmax z v - if v = y then 1 else 0) ^ 2)
        ≤ Real.sqrt 2 * (1 - Real.exp (-nll y z)) := by
  rw [one_sub_exp_neg_nll]
  exact grad_norm_bounds (softmax z) (softmax_nonneg z) (softmax_sum z) y

end WebKD
