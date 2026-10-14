import WebkdLean.Statements

set_option autoImplicit false
set_option relaxedAutoImplicit false

/-!
# Rank weighting is first-order optimal credit assignment (Theorem `rank`)

Strategy, following Appendix `theory-update`. Under Assumptions `additive` and `emit`, summing
`Pr[U] e^{α₀ + ∑_U Δ}` over the subsets `U` of segments factorizes coordinate by coordinate into
`e^{α₀} ∏_j (1 - κ_j + κ_j e^{Δ_j})` (`deployment_prob_product`, by `Finset.prod_add`). Only the
`j`-th factor depends on `κ_j`, so the derivative of the log deployment probability in
`u = log κ_j` is the logarithmic derivative of that factor, `μ_j` (`hasDerivAt_log_deployment`).
`μ_j` has the sign of `Δ_j` and is strictly increasing in it (`mu_strictMono`, by cross-
multiplying); in `κ_j` it is increasing when `Δ_j > 0` and decreasing when `Δ_j < 0`
(`mu_strictMonoOn_kappa`, `mu_strictAntiOn_kappa`; these back the scope note after the paper's
proof and are not used by the theorem). With a common `κ`, the `μ_j` are in the order
of the `Δ_j`, so the rearrangement inequality (`Monovary.sum_mul_comp_perm_le_sum_mul`) makes the
order-respecting assignment of any fixed multiset of weights optimal (`rank_order_optimal`).
-/

namespace WebKD

open Finset

variable {ι : Type} [Fintype ι] [DecidableEq ι]

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
  -- after cross-multiplying, the difference of the two sides is `κ (e^{Δ₂} - e^{Δ₁}) > 0`
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

/-! ### Eq. `mu` in full, `μ` in `κ`, and the statement -/

/-- **Eq. `mu`, full form.** The log deployment probability
`log(e^{α₀} ∏_i (1 - κ_i + κ_i e^{Δ_i}))`, as a function of `u = log κ_j` with the other `κ_i`
fixed, has derivative `μ_j` at `u = log κ_j`. -/
theorem hasDerivAt_log_deployment (α₀ : ℝ) (κ Δ : ι → ℝ) (j : ι) (hκ : ∀ i, 0 < κ i ∧ κ i < 1) :
    HasDerivAt
      (fun u => Real.log (Real.exp α₀ * ∏ i, (1 - Function.update κ j (Real.exp u) i
        + Function.update κ j (Real.exp u) i * Real.exp (Δ i))))
      (mu (κ j) (Δ j)) (Real.log (κ j)) := by
  set C := Real.exp α₀ * ∏ i ∈ Finset.univ.erase j, (1 - κ i + κ i * Real.exp (Δ i)) with hC
  have hCpos : 0 < C := mul_pos (Real.exp_pos _)
    (Finset.prod_pos fun i _ => factor_pos (κ i) (Δ i) (hκ i).1 (hκ i).2)
  -- only the `j`-th factor depends on `u`
  have hfun : (fun u => Real.log (Real.exp α₀ * ∏ i, (1 - Function.update κ j (Real.exp u) i
        + Function.update κ j (Real.exp u) i * Real.exp (Δ i))))
      = fun u => Real.log (C * (1 - Real.exp u + Real.exp u * Real.exp (Δ j))) := by
    funext u
    congr 1
    rw [← Finset.mul_prod_erase Finset.univ
      (fun i => 1 - Function.update κ j (Real.exp u) i
        + Function.update κ j (Real.exp u) i * Real.exp (Δ i)) (Finset.mem_univ j)]
    simp only [Function.update_self]
    rw [Finset.prod_congr rfl
      (fun i hi => by rw [Function.update_of_ne (Finset.ne_of_mem_erase hi)])]
    rw [hC]
    ring
  rw [hfun]
  have hκj : Real.exp (Real.log (κ j)) = κ j := Real.exp_log (hκ j).1
  have hg : HasDerivAt (fun u => C * (1 - Real.exp u + Real.exp u * Real.exp (Δ j)))
      (C * (0 - Real.exp (Real.log (κ j)) + Real.exp (Real.log (κ j)) * Real.exp (Δ j)))
      (Real.log (κ j)) :=
    (((hasDerivAt_const _ (1 : ℝ)).sub (Real.hasDerivAt_exp _)).add
      ((Real.hasDerivAt_exp _).mul_const _)).const_mul C
  have hne : C * (1 - Real.exp (Real.log (κ j)) + Real.exp (Real.log (κ j)) * Real.exp (Δ j))
      ≠ 0 := by
    rw [hκj]; exact (mul_pos hCpos (factor_pos _ _ (hκ j).1 (hκ j).2)).ne'
  refine (hg.log hne).congr_deriv ?_
  -- `(log (C g))' = g'/g`, and `g'/g = μ_j` at `e^u = κ_j`
  rw [hκj]
  have hf := (factor_pos (κ j) (Δ j) (hκ j).1 (hκ j).2).ne'
  unfold mu
  field_simp
  ring

/-- **`μ` in `κ`, positive score.** For `Δ > 0`, `μ` is strictly increasing in `κ ∈ (0, 1)`:
the difference of cross-products is `(e^Δ - 1)(κ₂ - κ₁) > 0`. -/
theorem mu_strictMonoOn_kappa (Δ : ℝ) (hΔ : 0 < Δ) :
    StrictMonoOn (fun κ => mu κ Δ) (Set.Ioo 0 1) := by
  intro κ₁ h₁ κ₂ h₂ h
  have hd : 0 < Real.exp Δ - 1 := by have := Real.one_lt_exp_iff.2 hΔ; linarith
  simp only [mu]
  rw [div_lt_div_iff₀ (factor_pos κ₁ Δ h₁.1 h₁.2) (factor_pos κ₂ Δ h₂.1 h₂.2)]
  nlinarith [mul_pos hd (sub_pos.2 h)]

/-- **`μ` in `κ`, negative score.** For `Δ < 0`, `μ` is strictly decreasing in `κ ∈ (0, 1)`:
learning a distracting segment more often hurts more. -/
theorem mu_strictAntiOn_kappa (Δ : ℝ) (hΔ : Δ < 0) :
    StrictAntiOn (fun κ => mu κ Δ) (Set.Ioo 0 1) := by
  intro κ₁ h₁ κ₂ h₂ h
  have hd : Real.exp Δ - 1 < 0 := by have := Real.exp_lt_one_iff.2 hΔ; linarith
  simp only [mu]
  rw [div_lt_div_iff₀ (factor_pos κ₂ Δ h₂.1 h₂.2) (factor_pos κ₁ Δ h₁.1 h₁.2)]
  nlinarith [mul_pos (neg_pos.2 hd) (sub_pos.2 h)]

/-- **Theorem `rank`** (`Statements.thm_rank`). -/
theorem thm_rank : Statements.thm_rank := by
  refine ⟨fun α₀ κ Δ => deployment_prob_product α₀ κ Δ,
    fun α₀ Δ j => leave_one_out_recovers α₀ Δ j,
    fun α₀ κ Δ j hκ => hasDerivAt_log_deployment α₀ κ Δ j hκ,
    fun κ h0 h1 => mu_strictMono κ h0 h1, fun κ => mu_zero κ,
    fun κ Δ h0 h1 hΔ => mu_neg κ Δ h0 h1 hΔ, ?_, ?_⟩
  · intro ι _ κ Δ v h0 h1 hv σ
    exact rank_order_optimal κ h0 h1 Δ v hv σ
  · intro ι Δ Δ' v hord
    exact order_only Δ Δ' v hord

end WebKD
