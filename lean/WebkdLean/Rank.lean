import Mathlib

/-!
# Rank weighting is first-order optimal credit assignment (Theorem `rank`)

Formalizes the product formula for the deployment probability under Assumptions `additive` and
`emit`, the marginal gain `μ_j` of Eq. `mu` as a derivative in `log κ_j`, its sign and strict
monotonicity in `Δ_j`, and the ordering claim via the rearrangement inequality.
-/

namespace WebKD

open Finset

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Product formula of Theorem `rank`.** Summing over inclusion patterns `U ⊆ {1..n}` with
`Pr[U] = ∏_{j∈U} κ_j ∏_{j∉U} (1-κ_j)` (Assumption `emit`) and
`P(A⋆ | h, U) = exp(α₀ + ∑_{j∈U} Δ_j)` (Assumption `additive`) gives
`e^{α₀} ∏_j (1 - κ_j + κ_j e^{Δ_j})`. -/
theorem deployment_prob_product (α0 : ℝ) (κ Δ : ι → ℝ) :
    ∑ U : Finset ι, ((∏ j ∈ U, κ j) * ∏ j ∈ Finset.univ \ U, (1 - κ j))
        * Real.exp (α0 + ∑ j ∈ U, Δ j)
      = Real.exp α0 * ∏ j, (1 - κ j + κ j * Real.exp (Δ j)) := by
  have h : ∀ j, 1 - κ j + κ j * Real.exp (Δ j) = κ j * Real.exp (Δ j) + (1 - κ j) :=
    fun j => by ring
  simp_rw [h]
  rw [Finset.prod_add, Finset.powerset_univ, Finset.mul_sum]
  refine Finset.sum_congr rfl fun U _ => ?_
  rw [Real.exp_add, Real.exp_sum, Finset.prod_mul_distrib]
  ring

/-- Under Assumption `additive`, the leave-one-out score of Eq. `actw` recovers `Δ_j` exactly. -/
theorem leave_one_out_recovers (α0 : ℝ) (Δ : ι → ℝ) (j : ι) :
    (α0 + ∑ i, Δ i) - (α0 + ∑ i ∈ Finset.univ.erase j, Δ i) = Δ j := by
  rw [Finset.sum_erase_eq_sub (Finset.mem_univ j)]
  ring

/-- `μ_j` of Eq. `mu`: `κ (e^Δ - 1) / (1 - κ + κ e^Δ)`. -/
noncomputable def mu (κ Δ : ℝ) : ℝ := κ * (Real.exp Δ - 1) / (1 - κ + κ * Real.exp Δ)

lemma factor_pos (κ Δ : ℝ) (h0 : 0 < κ) (h1 : κ < 1) : 0 < 1 - κ + κ * Real.exp Δ := by
  have := Real.exp_pos Δ
  nlinarith

/-- **Eq. `mu`.** `μ_j` is the derivative of `log(1 - κ_j + κ_j e^{Δ_j})` in `u = log κ_j`,
i.e. the gain in `log P(A⋆ | h)` per nat of premium absorbed on segment `j`. -/
theorem hasDerivAt_log_factor (Δ u : ℝ) (hu : u < 0) :
    HasDerivAt (fun u => Real.log (1 - Real.exp u + Real.exp u * Real.exp Δ))
      (mu (Real.exp u) Δ) u := by
  have hκ0 : 0 < Real.exp u := Real.exp_pos u
  have hκ1 : Real.exp u < 1 := Real.exp_lt_one_iff.2 hu
  have h : HasDerivAt (fun u => 1 - Real.exp u + Real.exp u * Real.exp Δ)
      (0 - Real.exp u + Real.exp u * Real.exp Δ) u :=
    ((hasDerivAt_const u (1 : ℝ)).sub (Real.hasDerivAt_exp u)).add
      ((Real.hasDerivAt_exp u).mul_const _)
  refine (h.log (factor_pos _ Δ hκ0 hκ1).ne').congr_deriv ?_
  unfold mu
  ring

/-- `μ_j` is strictly increasing in `Δ_j` (for `κ_j ∈ (0,1)`). -/
theorem mu_strictMono (κ : ℝ) (h0 : 0 < κ) (h1 : κ < 1) : StrictMono (mu κ) := by
  intro Δ1 Δ2 h
  have ha : Real.exp Δ1 < Real.exp Δ2 := Real.exp_lt_exp.2 h
  unfold mu
  rw [div_lt_div_iff₀ (factor_pos κ Δ1 h0 h1) (factor_pos κ Δ2 h0 h1)]
  nlinarith [mul_pos h0 (sub_pos.2 ha)]

/-- `μ_j = 0` at `Δ_j = 0`. -/
theorem mu_zero (κ : ℝ) : mu κ 0 = 0 := by
  simp [mu]

/-- `μ_j < 0` for `Δ_j < 0`: learning a distractor lowers the deployment probability. -/
theorem mu_neg (κ Δ : ℝ) (h0 : 0 < κ) (h1 : κ < 1) (hΔ : Δ < 0) : mu κ Δ < 0 := by
  unfold mu
  apply div_neg_of_neg_of_pos _ (factor_pos κ Δ h0 h1)
  have : Real.exp Δ < 1 := Real.exp_lt_one_iff.2 hΔ
  nlinarith

/-- `μ_j > 0` for `Δ_j > 0`. -/
theorem mu_pos (κ Δ : ℝ) (h0 : 0 < κ) (h1 : κ < 1) (hΔ : 0 < Δ) : 0 < mu κ Δ := by
  unfold mu
  apply div_pos _ (factor_pos κ Δ h0 h1)
  have : 1 < Real.exp Δ := Real.one_lt_exp_iff.2 hΔ
  nlinarith

omit [DecidableEq ι] in
/-- **Ordering claim of Theorem `rank`.** With equal `κ`, if the weights `v` are assigned in the
order of `Δ` (`Δ i < Δ j → v i ≤ v j`), then no permutation `σ` of the same multiset of weights
gives a larger first-order gain `∑_j μ_j v_j`. -/
theorem rank_order_optimal (κ : ℝ) (h0 : 0 < κ) (h1 : κ < 1) (Δ v : ι → ℝ)
    (hv : Monovary v Δ) (σ : Equiv.Perm ι) :
    ∑ j, mu κ (Δ j) * v (σ j) ≤ ∑ j, mu κ (Δ j) * v j := by
  apply Monovary.sum_mul_comp_perm_le_sum_mul
  intro i j hij
  apply (mu_strictMono κ h0 h1).monotone
  by_contra h
  exact absurd hij (not_lt.2 (hv (not_le.1 h)))

omit [Fintype ι] [DecidableEq ι] in
/-- The maximizing assignment depends on `Δ` only through its order: two score vectors inducing
the same strict order admit the same order-respecting weight assignments. -/
theorem order_only (Δ Δ' v : ι → ℝ) (hord : ∀ i j, Δ i < Δ j ↔ Δ' i < Δ' j) :
    Monovary v Δ ↔ Monovary v Δ' := by
  constructor
  · intro h i j hij; exact h ((hord i j).2 hij)
  · intro h i j hij; exact h ((hord i j).1 hij)

end WebKD
