import Mathlib

/-!
# Where the uniform update goes (Proposition `split`)

Formalizes both parts of Proposition `split`: (i) the action's share of the uniform loss
restricted to `I_rat ∪ I_act` written with pooled perplexities, and (ii) the bound on the
action's share of the logit-gradient mass under Assumption `cap`, using Lemma `sandwich` as
the hypothesis on the per-token gradient norms.
-/

namespace WebKD

open Finset

/-- `1 - e^{-x} ≤ x`. -/
lemma one_sub_exp_neg_le (x : ℝ) : 1 - Real.exp (-x) ≤ x := by
  have := Real.add_one_le_exp (-x)
  linarith

/-- `0 ≤ 1 - e^{-x}` for `x ≥ 0`. -/
lemma one_sub_exp_neg_nonneg (x : ℝ) (hx : 0 ≤ x) : 0 ≤ 1 - Real.exp (-x) := by
  have : Real.exp (-x) ≤ 1 := Real.exp_le_one_iff.2 (by linarith)
  linarith

/-- Chord of the concave map `x ↦ 1 - e^{-x}` on `[0, Λ]`:
`(1 - e^{-Λ}) x / Λ ≤ 1 - e^{-x}`. -/
lemma chord_le_one_sub_exp_neg (Λ x : ℝ) (hΛ : 0 < Λ) (hx0 : 0 ≤ x) (hxΛ : x ≤ Λ) :
    (1 - Real.exp (-Λ)) * x / Λ ≤ 1 - Real.exp (-x) := by
  have ht0 : 0 ≤ x / Λ := div_nonneg hx0 hΛ.le
  have ht1 : x / Λ ≤ 1 := by rw [div_le_one hΛ]; exact hxΛ
  have hconv := convexOn_exp.2 (Set.mem_univ (0 : ℝ)) (Set.mem_univ (-Λ))
    (sub_nonneg.2 ht1) ht0 (by ring)
  simp only [smul_eq_mul, mul_zero, zero_add, Real.exp_zero, mul_one] at hconv
  have hx : x / Λ * -Λ = -x := by field_simp
  rw [hx] at hconv
  have : (1 - Real.exp (-Λ)) * x / Λ = x / Λ - x / Λ * Real.exp (-Λ) := by ring
  rw [this]
  linarith

variable {ι : Type*}

/-- **Proposition `split` (i).** Restricted to `I_rat ∪ I_act`, the action's share of the
uniform loss is `n_act log PPL_act / (n_act log PPL_act + n_rat log PPL_rat)`, where
`log PPL_C = (1/n_C) ∑_{i ∈ I_C} CE_i` is the pooled per-token perplexity. -/
theorem loss_share [DecidableEq ι] (Irat Iact : Finset ι) (hdisj : Disjoint Irat Iact) (CE : ι → ℝ)
    (hr : Irat.Nonempty) (ha : Iact.Nonempty) :
    ((1 / ((Irat.card : ℝ) + Iact.card)) * ∑ i ∈ Iact, CE i)
        / ((1 / ((Irat.card : ℝ) + Iact.card)) * ∑ i ∈ Irat ∪ Iact, CE i)
      = (Iact.card : ℝ) * ((∑ i ∈ Iact, CE i) / Iact.card)
        / ((Iact.card : ℝ) * ((∑ i ∈ Iact, CE i) / Iact.card)
            + (Irat.card : ℝ) * ((∑ i ∈ Irat, CE i) / Irat.card)) := by
  have hrc : (0 : ℝ) < Irat.card := by exact_mod_cast Finset.card_pos.2 hr
  have hac : (0 : ℝ) < Iact.card := by exact_mod_cast Finset.card_pos.2 ha
  have h1 : (Iact.card : ℝ) * ((∑ i ∈ Iact, CE i) / Iact.card) = ∑ i ∈ Iact, CE i := by
    field_simp
  have h2 : (Irat.card : ℝ) * ((∑ i ∈ Irat, CE i) / Irat.card) = ∑ i ∈ Irat, CE i := by
    field_simp
  rw [h1, h2, Finset.sum_union hdisj,
    mul_div_mul_left _ _ (by positivity : (1 / ((Irat.card : ℝ) + Iact.card)) ≠ 0), add_comm]

/-- Monotonicity of `t ↦ t/(t+s)` in `t` and antitonicity in `s`. -/
lemma ratio_mono (t1 t2 s1 s2 : ℝ) (ht1 : 0 ≤ t1) (ht12 : t1 ≤ t2) (hs2 : 0 ≤ s2)
    (hs21 : s2 ≤ s1) : t1 / (t1 + s1) ≤ t2 / (t2 + s2) := by
  rcases eq_or_lt_of_le ht1 with h | h
  · rw [← h, zero_div]
    exact div_nonneg (by linarith) (by linarith)
  · have hd1 : 0 < t1 + s1 := by linarith
    have hd2 : 0 < t2 + s2 := by linarith
    rw [div_le_div_iff₀ hd1 hd2]
    nlinarith [mul_le_mul_of_nonneg_left hs21 ht1, mul_le_mul_of_nonneg_right ht12 (by linarith : 0 ≤ s1)]

/-- **Proposition `split` (ii).** Under Assumption `cap` (`CE_i ≤ Λ` on the rationale) and
Lemma `sandwich` (`1 - e^{-CE_i} ≤ g_i ≤ √2 (1 - e^{-CE_i})` for the per-token logit-gradient
norms `g_i`), the action's share of the logit-gradient mass satisfies
`G_act/(G_act + G_rat) ≤ X/(X + Y)` with `X = √2 ∑_act CE_i` and
`Y = ((1 - e^{-Λ})/Λ) ∑_rat CE_i`. -/
theorem grad_share (Irat Iact : Finset ι) (CE g : ι → ℝ) (Λ : ℝ) (hΛ : 0 < Λ)
    (hCE0 : ∀ i, 0 ≤ CE i) (hcap : ∀ i ∈ Irat, CE i ≤ Λ)
    (hlow : ∀ i, 1 - Real.exp (-CE i) ≤ g i)
    (hup : ∀ i, g i ≤ Real.sqrt 2 * (1 - Real.exp (-CE i))) :
    (∑ i ∈ Iact, g i) / ((∑ i ∈ Iact, g i) + ∑ i ∈ Irat, g i)
      ≤ (Real.sqrt 2 * ∑ i ∈ Iact, CE i)
        / ((Real.sqrt 2 * ∑ i ∈ Iact, CE i)
            + (1 - Real.exp (-Λ)) / Λ * ∑ i ∈ Irat, CE i) := by
  have hg0 : ∀ i, 0 ≤ g i := fun i => le_trans (one_sub_exp_neg_nonneg _ (hCE0 i)) (hlow i)
  have hGact : ∑ i ∈ Iact, g i ≤ Real.sqrt 2 * ∑ i ∈ Iact, CE i := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    calc g i ≤ Real.sqrt 2 * (1 - Real.exp (-CE i)) := hup i
      _ ≤ Real.sqrt 2 * CE i :=
          mul_le_mul_of_nonneg_left (one_sub_exp_neg_le _) (Real.sqrt_nonneg _)
  have hY : (1 - Real.exp (-Λ)) / Λ * ∑ i ∈ Irat, CE i ≤ ∑ i ∈ Irat, g i := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    calc (1 - Real.exp (-Λ)) / Λ * CE i = (1 - Real.exp (-Λ)) * CE i / Λ := by ring
      _ ≤ 1 - Real.exp (-CE i) := chord_le_one_sub_exp_neg Λ (CE i) hΛ (hCE0 i) (hcap i hi)
      _ ≤ g i := hlow i
  have hY0 : 0 ≤ (1 - Real.exp (-Λ)) / Λ * ∑ i ∈ Irat, CE i :=
    mul_nonneg (div_nonneg (one_sub_exp_neg_nonneg Λ hΛ.le) hΛ.le)
      (Finset.sum_nonneg fun i _ => hCE0 i)
  exact ratio_mono _ _ _ _ (Finset.sum_nonneg fun i _ => hg0 i) hGact hY0 hY

end WebKD
