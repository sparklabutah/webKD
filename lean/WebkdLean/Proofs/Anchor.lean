import WebkdLean.Proofs.Softmax

set_option autoImplicit false
set_option relaxedAutoImplicit false

/-!
# Role-asymmetric anchoring (Eq. `kl-grad`, Proposition `target`, Corollaries `beta`, `budget`)

Strategy, following Appendix `proof-target`. Position `i` contributes
`L_i(p) = -a log p + b (q/p + log(p/q) - 1)` with `dL_i/dp = -c_i(p)/p` for the throttle
`c_i(p) = a - b(1 - q/p)` (`hasDerivAt_anchorLoss`). For `b > 0` the throttle is strictly decreasing
(`throttle_strictAnti`) and vanishes exactly at `q b/(b - a)` when `0 < a < b`
(`throttle_eq_zero_iff`), so it is positive below that point and negative above it
(`throttle_sign`). The minimization is proved directly, without calculus, from `log x ≤ x - 1`:
`L(p) - L(p̂) = (q b/p̂)((p̂/p - 1) - log(p̂/p)) ≥ 0` at the interior point; at the boundary `p⋆ = 1`
it is `L(p) - L(1) = (b - a) log p + b q (1/p - 1) ≥ 0`, by `log p ≤ 0` when `b ≤ a` and by
`log p ≥ 1 - 1/p` with `b - a < q b` when the interior point exceeds `1` (`anchorLoss_pStar_le`).
The paper's proof reaches the same minimizer through the sign of the derivative. The cap binds (`p⋆ < 1`) iff `b(1 - q) > a`
(`pStar_lt_one_iff`), which in anchor strength reads `β > β̄/(1 - q)`. The budget is
`log(p⋆/q) ≤ log(b/(b - a)) = -log(1 - a/b)`, with equality when the interior point is at most `1`.
The chain rule through the softmax gives the coordinate form `∇_z L_i = c_i (p - e_y)`
(`hasDerivAt_anchored_coord`).
-/

namespace WebKD

open Real

/-- `dL_i/dp = -c_i(p)/p`: the anchor rescales the SFT force through the throttle. -/
theorem hasDerivAt_anchorLoss (a b q p : ℝ) (hp : 0 < p) (hq : 0 < q) :
    HasDerivAt (anchorLoss a b q) (-(throttle a b q p) / p) p := by
  have h1 : HasDerivAt (fun x : ℝ => -a * Real.log x) (-a * p⁻¹) p :=
    (Real.hasDerivAt_log hp.ne').const_mul (-a)
  have h2 : HasDerivAt (fun x : ℝ => q / x) ((0 * p - q * 1) / p ^ 2) p :=
    (hasDerivAt_const p q).div (hasDerivAt_id p) hp.ne'
  have h3 : HasDerivAt (fun x : ℝ => Real.log (x / q)) ((1 / q) / (p / q)) p :=
    ((hasDerivAt_id p).div_const q).log (div_pos hp hq).ne'
  have h4 := ((h2.add h3).sub_const 1).const_mul b
  have h5 := h1.add h4
  have hfun : anchorLoss a b q = (fun x : ℝ => -a * Real.log x)
      + fun x : ℝ => b * (q / x + Real.log (x / q) - 1) := by
    funext x; simp [anchorLoss]
  rw [hfun]
  refine h5.congr_deriv ?_
  -- collect `-a/p + b(-q/p² + 1/p)` into `-c(p)/p`
  unfold throttle
  field_simp
  ring

/-- At initialization (`p = q`) the anchor is inert: `c_i = a_i`. -/
theorem throttle_at_init (a b q : ℝ) (hq : 0 < q) : throttle a b q q = a := by
  unfold throttle
  rw [div_self hq.ne']
  ring

/-- Below the reference (`p < q`) the anchor adds force toward the corpus token: `c_i > a_i`. -/
theorem throttle_gt_of_lt (a b q p : ℝ) (hb : 0 < b) (hp : 0 < p) (hpq : p < q) :
    a < throttle a b q p := by
  unfold throttle
  have : 1 < q / p := by rw [lt_div_iff₀ hp]; linarith
  nlinarith

/-- The throttle is strictly decreasing in `p` (for `b > 0`). -/
theorem throttle_strictAnti (a b q : ℝ) (hb : 0 < b) (hq : 0 < q) :
    StrictAntiOn (throttle a b q) (Set.Ioi 0) := by
  intro p1 hp1 p2 hp2 h
  unfold throttle
  have h1 : (0 : ℝ) < p1 := hp1
  have h2 : (0 : ℝ) < p2 := hp2
  have : q / p2 < q / p1 := div_lt_div_of_pos_left hq h1 h
  nlinarith

/-- Above `p⋆` the throttle is negative (the update stops); the stationary point is `q b/(b-a)`. -/
theorem throttle_eq_zero_iff (a b q p : ℝ) (ha : 0 < a) (hab : a < b) (hp : 0 < p)
    (hq : 0 < q) :
    throttle a b q p = 0 ↔ p = q * b / (b - a) := by
  unfold throttle
  have hba : 0 < b - a := sub_pos.2 hab
  have hb : 0 < b := by linarith
  have hrw : b * (1 - q / p) = b * (p - q) / p := by field_simp
  rw [hrw]
  constructor
  · intro h
    have h' : a * p - b * (p - q) = 0 := by
      field_simp at h
      linarith
    rw [eq_div_iff hba.ne']
    linarith
  · intro h
    rw [h]
    field_simp
    ring

/-- Key identity: `L(p) - L(1) = (b - a) log p + b q (1/p - 1)`. -/
lemma anchorLoss_sub_one (a b q p : ℝ) (hp : 0 < p) (hq : 0 < q) :
    anchorLoss a b q p - anchorLoss a b q 1
      = (b - a) * Real.log p + b * q * (1 / p - 1) := by
  unfold anchorLoss
  rw [Real.log_div hp.ne' hq.ne', Real.log_div one_ne_zero hq.ne', Real.log_one]
  ring

/-- Key identity: `L(p) - L(r) = (b - a)(log p - log r) + b q (1/p - 1/r)`. -/
lemma anchorLoss_sub (a b q p r : ℝ) (hp : 0 < p) (hr : 0 < r) (hq : 0 < q) :
    anchorLoss a b q p - anchorLoss a b q r
      = (b - a) * (Real.log p - Real.log r) + b * q * (1 / p - 1 / r) := by
  unfold anchorLoss
  rw [Real.log_div hp.ne' hq.ne', Real.log_div hr.ne' hq.ne']
  ring

/-- **Proposition `target`.** With `a > 0`, `b ≥ 0`, `q > 0`, the per-position contribution
is minimized over `p ∈ (0, 1]` at `p⋆ = min{1, q b/(b-a)}` if `b > a`, and at `p⋆ = 1` if
`b ≤ a`. -/
theorem anchorLoss_pStar_le (a b q : ℝ) (ha : 0 < a) (hb : 0 ≤ b) (hq0 : 0 < q)
    (p : ℝ) (hp0 : 0 < p) (hp1 : p ≤ 1) :
    anchorLoss a b q (pStar a b q) ≤ anchorLoss a b q p := by
  have hlogp : 1 - 1 / p ≤ Real.log p := by
    have := Real.log_le_sub_one_of_pos (inv_pos.2 hp0)
    rw [Real.log_inv] at this
    rw [one_div]; linarith
  have hlogp' : Real.log p ≤ 0 := Real.log_nonpos hp0.le hp1
  have hinvp : 0 ≤ 1 / p - 1 := by
    rw [sub_nonneg, le_div_iff₀ hp0]; linarith
  unfold pStar
  split_ifs with hab
  · have hba : 0 < b - a := sub_pos.2 hab
    have hbpos : 0 < b := by linarith
    set phat := q * b / (b - a) with hphat
    have hphat_pos : 0 < phat := div_pos (mul_pos hq0 hbpos) hba
    have hkey : ∀ x, 0 < x → anchorLoss a b q phat ≤ anchorLoss a b q x := by
      intro x hx
      rw [← sub_nonneg, anchorLoss_sub a b q x phat hx hphat_pos hq0]
      have hlog : Real.log (phat / x) ≤ phat / x - 1 :=
        Real.log_le_sub_one_of_pos (div_pos hphat_pos hx)
      rw [Real.log_div hphat_pos.ne' hx.ne'] at hlog
      have hba' : b - a = q * b / phat := by
        rw [hphat]; field_simp
      have hid : (b - a) * (Real.log x - Real.log phat) + b * q * (1 / x - 1 / phat)
          = (q * b / phat) * ((phat / x - 1) - (Real.log phat - Real.log x)) := by
        rw [hba']; field_simp; ring
      rw [hid]
      apply mul_nonneg (div_nonneg (mul_nonneg hq0.le hb) hphat_pos.le)
      linarith
    rcases le_or_gt phat 1 with h | h
    · rw [min_eq_right h]; exact hkey p hp0
    · rw [min_eq_left h.le]
      rw [← sub_nonneg, anchorLoss_sub_one a b q p hp0 hq0]
      have hqb : b - a ≤ q * b := by
        rw [hphat, lt_div_iff₀ hba] at h; linarith
      have h1 : (b - a) * (1 - 1 / p) ≤ (b - a) * Real.log p :=
        mul_le_mul_of_nonneg_left hlogp hba.le
      -- `(b - a) log p + b q (1/p - 1) ≥ (b - a)(1 - 1/p) + b q (1/p - 1)`
      -- `= (1/p - 1)(q b - (b - a)) ≥ 0`
      nlinarith [mul_nonneg hinvp (sub_nonneg.2 hqb)]
  · have hab' : b ≤ a := not_lt.1 hab
    rw [← sub_nonneg, anchorLoss_sub_one a b q p hp0 hq0]
    have h1 : 0 ≤ (b - a) * Real.log p := mul_nonneg_of_nonpos_of_nonpos (by linarith) hlogp'
    have h2 : 0 ≤ b * q * (1 / p - 1) := mul_nonneg (mul_nonneg hb hq0.le) hinvp
    linarith

/-- Corollary `roles`: on tokens with `b ≤ a` (in particular `γ_i ≈ 0`, so `b_i ≈ 0 < a_i`)
the target is the one-hot SFT target. -/
theorem pStar_eq_one_of_le (a b q : ℝ) (hab : b ≤ a) : pStar a b q = 1 := by
  unfold pStar
  split_ifs with h
  · exact absurd h (not_lt.2 hab)
  · rfl

/-- **Corollary `beta`.** With `a = w_i/∑w` and `b = β γ_i/∑γ`, the cap exists (`a < b`)
iff `β > β̄_i`. -/
theorem cap_exists_iff (w W γ Γ β : ℝ) (hW : 0 < W) (hγ : 0 < γ) (hΓ : 0 < Γ) :
    w / W < β * γ / Γ ↔ betaBar w W γ Γ < β := by
  unfold betaBar
  have e1 : w / W < β * γ / Γ ↔ w * Γ < β * γ * W := by
    rw [div_lt_iff₀ hW, div_mul_eq_mul_div, lt_div_iff₀ hΓ]
  have e2 : (w / γ) * (Γ / W) < β ↔ w * Γ < β * γ * W := by
    rw [div_mul_div_comm, div_lt_iff₀ (mul_pos hγ hW)]
    constructor <;> intro h <;> linarith
  rw [e1, e2]

/-- `a_i / b_i = β̄_i / β`. -/
theorem ratio_eq_betaBar_div (w W γ Γ β : ℝ) (hW : 0 < W) (hγ : 0 < γ) (hΓ : 0 < Γ)
    (hβ : 0 < β) :
    (w / W) / (β * γ / Γ) = betaBar w W γ Γ / β := by
  unfold betaBar
  field_simp

/-- **Corollary `budget`, capped case.** When `a < b`, the premium shed on the token between
initialization (`p = q`) and the per-token optimum, `log(p⋆/q)`, is at most
`log(b/(b-a)) = -log(1 - a/b)`. -/
theorem budget_bound (a b q : ℝ) (ha : 0 < a) (hab : a < b) (hq0 : 0 < q) :
    Real.log (pStar a b q / q) ≤ -Real.log (1 - a / b) := by
  have hb : 0 < b := by linarith
  have hba : 0 < b - a := sub_pos.2 hab
  unfold pStar
  split_ifs
  have h1 : 1 - a / b = (b - a) / b := by field_simp
  rw [h1, ← Real.log_inv, inv_div]
  apply Real.log_le_log (div_pos (lt_min one_pos (div_pos (mul_pos hq0 hb) hba)) hq0)
  calc min 1 (q * b / (b - a)) / q ≤ (q * b / (b - a)) / q := by
        gcongr
        exact min_le_right _ _
    _ = b / (b - a) := by field_simp

/-- **Corollary `budget`, uncapped case.** When `b ≤ a` the token may be fit to `p⋆ = 1`, and the
shed surprisal is the full `CE_i = -log q` at initialization. -/
theorem budget_uncapped (a b q : ℝ) (hab : b ≤ a) :
    Real.log (pStar a b q / q) = -Real.log q := by
  rw [pStar_eq_one_of_le a b q hab, one_div, Real.log_inv]

/-- The budget `-log(1 - β̄/β)` decreases in `β` (for `β > β̄ ≥ 0`). -/
theorem budget_anti_in_beta (bb β1 β2 : ℝ) (hbb : 0 ≤ bb) (h1 : bb < β1) (h12 : β1 ≤ β2) :
    -Real.log (1 - bb / β2) ≤ -Real.log (1 - bb / β1) := by
  have hβ1 : 0 < β1 := by linarith
  have hβ2 : 0 < β2 := by linarith
  have hpos : 0 < 1 - bb / β1 := by
    rw [sub_pos, div_lt_one hβ1]; exact h1
  have hle : bb / β2 ≤ bb / β1 := div_le_div_of_nonneg_left hbb hβ1 h12
  have : Real.log (1 - bb / β1) ≤ Real.log (1 - bb / β2) :=
    Real.log_le_log hpos (by linarith)
  linarith

/-- The budget increases in `β̄` (hence in `w_i`) at fixed `β` (for `0 ≤ β̄₁ ≤ β̄₂ < β`). -/
theorem budget_mono_in_betaBar (bb1 bb2 β : ℝ) (h0 : 0 ≤ bb1) (h12 : bb1 ≤ bb2) (h2 : bb2 < β) :
    -Real.log (1 - bb1 / β) ≤ -Real.log (1 - bb2 / β) := by
  have hβ : 0 < β := by linarith
  have hpos : 0 < 1 - bb2 / β := by
    rw [sub_pos, div_lt_one hβ]; exact h2
  have hle : bb1 / β ≤ bb2 / β := div_le_div_of_nonneg_right h12 hβ.le
  have : Real.log (1 - bb2 / β) ≤ Real.log (1 - bb1 / β) :=
    Real.log_le_log hpos (by linarith)
  linarith

/-- **Eq. `kl-grad`, coordinate form.** Position `i`'s contribution to the anchored objective,
seen as a function of its logits `z` through `p = softmax(z)_y`, has partial derivative
`c_i · (softmax(z)_v - 𝟙[v = y])` along every logit `z_v`. This is `∇_z L = c_i (p - e_y)`: the
anchor rescales the SFT direction by the throttle and never rotates it. In particular each
off-target logit (`v ≠ y`) moves at rate `-c_i p_v`, proportional to its current probability, so
under gradient flow the gap between two off-target logits `z_u - z_v` moves at rate
`-c_i (p_u - p_v)` and the off-target distribution flattens rather than keeping its shape. -/
theorem hasDerivAt_anchored_coord {ι : Type} [Fintype ι] [DecidableEq ι] [Nonempty ι]
    (a b q : ℝ) (hq : 0 < q) (y v : ι) (z : ι → ℝ) :
    HasDerivAt (fun t => anchorLoss a b q (softmax (Function.update z v t) y))
      (throttle a b q (softmax z y) * (softmax z v - if v = y then 1 else 0)) (z v) := by
  have hp : 0 < softmax z y := softmax_pos z y
  have hnll := hasDerivAt_nll_coord y v z
  -- `p(t) = exp (-(nll y (update z v t)))`
  have hpfun : (fun t => softmax (Function.update z v t) y)
      = fun t => Real.exp (-(nll y (Function.update z v t))) := by
    funext t
    unfold nll
    rw [neg_neg, Real.exp_log (softmax_pos _ y)]
  have hpder : HasDerivAt (fun t => softmax (Function.update z v t) y)
      (softmax z y * -(softmax z v - if v = y then 1 else 0)) (z v) := by
    have h1 : HasDerivAt (fun t => Real.exp (-(nll y (Function.update z v t))))
        (Real.exp (-(nll y (Function.update z v (z v))))
          * -(softmax z v - if v = y then 1 else 0)) (z v) := (hnll.neg).exp
    have h2 : Real.exp (-(nll y (Function.update z v (z v)))) = softmax z y := by
      rw [Function.update_eq_self]
      unfold nll
      rw [neg_neg, Real.exp_log hp]
    rw [h2] at h1
    rw [hpfun]
    exact h1
  have hx : (fun t => softmax (Function.update z v t) y) (z v) = softmax z y := by
    simp only [Function.update_eq_self]
  have hL : HasDerivAt (anchorLoss a b q) (-(throttle a b q (softmax z y)) / softmax z y)
      ((fun t => softmax (Function.update z v t) y) (z v)) := by
    rw [hx]
    exact hasDerivAt_anchorLoss a b q (softmax z y) hp hq
  have hcomp := hL.comp (z v) hpder
  have hderiv : -(throttle a b q (softmax z y)) / softmax z y
      * (softmax z y * -(softmax z v - if v = y then 1 else 0))
      = throttle a b q (softmax z y) * (softmax z v - if v = y then 1 else 0) := by
    have hpne : softmax z y ≠ 0 := hp.ne'
    field_simp
  rw [hderiv] at hcomp
  exact hcomp

/-- The gap between two off-target logits moves at rate `-c_i (p_u - p_v)`: the difference of
the two coordinate derivatives of `hasDerivAt_anchored_coord`. -/
theorem offtarget_gap_rate {ι : Type} [Fintype ι] [DecidableEq ι] [Nonempty ι]
    (a b q : ℝ) (y u v : ι) (z : ι → ℝ) (hu : u ≠ y) (hv : v ≠ y) :
    throttle a b q (softmax z y) * (softmax z u - if u = y then 1 else 0)
      - throttle a b q (softmax z y) * (softmax z v - if v = y then 1 else 0)
      = throttle a b q (softmax z y) * (softmax z u - softmax z v) := by
  simp only [hu, hv, ite_false]
  ring

/-! ### Additions: range of `p⋆`, sign of the throttle, binding, equality case -/

/-- The per-token target lies in `(0, 1]`. -/
lemma pStar_pos_le_one (a b q : ℝ) (ha : 0 < a) (hq : 0 < q) :
    0 < pStar a b q ∧ pStar a b q ≤ 1 := by
  unfold pStar
  split_ifs with hab
  · have hba : 0 < b - a := sub_pos.2 hab
    have hbpos : 0 < b := lt_trans ha hab
    exact ⟨lt_min one_pos (div_pos (mul_pos hq hbpos) hba), min_le_left _ _⟩
  · exact ⟨one_pos, le_rfl⟩

/-- **Sign of the throttle.** For `0 < a < b` it is positive below `q b/(b - a)` (the update moves
toward the corpus token) and negative above it (the update pushes `p` back down). -/
theorem throttle_sign (a b q p : ℝ) (ha : 0 < a) (hab : a < b) (hp : 0 < p) (hq : 0 < q) :
    (p < q * b / (b - a) → 0 < throttle a b q p) ∧
      (q * b / (b - a) < p → throttle a b q p < 0) := by
  have hb : 0 < b := lt_trans ha hab
  have hstar : 0 < q * b / (b - a) := div_pos (mul_pos hq hb) (sub_pos.2 hab)
  have hzero : throttle a b q (q * b / (b - a)) = 0 :=
    (throttle_eq_zero_iff a b q _ ha hab hstar hq).2 rfl
  have hanti := throttle_strictAnti a b q hb hq
  constructor
  · intro h
    have := hanti (Set.mem_Ioi.2 hp) (Set.mem_Ioi.2 hstar) h
    linarith
  · intro h
    have := hanti (Set.mem_Ioi.2 hstar) (Set.mem_Ioi.2 hp) h
    linarith

/-- **The cap binds iff `b (1 - q) > a`.** The interior optimum `q b/(b - a)` lies below `1`
exactly when `b(1 - q) > a`; otherwise the target is the one-hot token. -/
theorem pStar_lt_one_iff (a b q : ℝ) (hb : 0 ≤ b) (hq0 : 0 < q) :
    pStar a b q < 1 ↔ a < b * (1 - q) := by
  unfold pStar
  split_ifs with hab
  · have hba : 0 < b - a := sub_pos.2 hab
    rw [min_lt_iff, div_lt_one hba]
    constructor
    · rintro (h | h)
      · exact absurd h (lt_irrefl 1)
      · linarith
    · intro h; right; linarith
  · constructor
    · intro h; exact absurd h (lt_irrefl 1)
    · intro h
      exfalso
      -- `b (1 - q) ≤ b ≤ a` when the cap does not exist
      have : b * (1 - q) ≤ b := by nlinarith
      exact hab (lt_of_lt_of_le h this)

/-- **Binding threshold in anchor strength.** With `a = w/W`, `b = β γ/Γ`, the cap binds iff
`β > β̄/(1 - q)`. -/
theorem pStar_lt_one_iff_beta (w W γ Γ β q : ℝ) (hW : 0 < W) (hγ : 0 < γ) (hΓ : 0 < Γ)
    (hβ : 0 ≤ β) (hq0 : 0 < q) (hq1 : q < 1) :
    pStar (w / W) (β * γ / Γ) q < 1 ↔ betaBar w W γ Γ / (1 - q) < β := by
  have hb : 0 ≤ β * γ / Γ := div_nonneg (mul_nonneg hβ hγ.le) hΓ.le
  have h1q : 0 < 1 - q := sub_pos.2 hq1
  rw [pStar_lt_one_iff _ _ q hb hq0, betaBar]
  -- both sides are `w Γ < β γ (1 - q) W` after clearing the positive denominators
  rw [div_lt_iff₀ h1q, div_mul_div_comm, div_lt_iff₀ (mul_pos hγ hW)]
  rw [div_lt_iff₀ hW]
  constructor
  · intro h
    have h' : w * Γ < β * γ / Γ * (1 - q) * W * Γ := by nlinarith
    have hrw : β * γ / Γ * (1 - q) * W * Γ = β * (1 - q) * (γ * W) := by field_simp
    linarith
  · intro h
    have hrw : β * γ / Γ * (1 - q) * W * Γ = β * (1 - q) * (γ * W) := by field_simp
    have h' : w * Γ < β * γ / Γ * (1 - q) * W * Γ := by linarith
    nlinarith

/-- The premium shed between initialization and the optimum is `log(p⋆/q)`. -/
lemma shed_eq_log_ratio (a b q : ℝ) (ha : 0 < a) (hq : 0 < q) :
    (-Real.log q) - (-Real.log (pStar a b q)) = Real.log (pStar a b q / q) := by
  rw [Real.log_div (pStar_pos_le_one a b q ha hq).1.ne' hq.ne']
  ring

/-- **Corollary `budget`, equality case.** When the interior cap is at most `1`, the budget bound
holds with equality. -/
theorem budget_eq_of_interior (a b q : ℝ) (ha : 0 < a) (hab : a < b) (hq0 : 0 < q)
    (hin : q * b / (b - a) ≤ 1) :
    Real.log (pStar a b q / q) = -Real.log (1 - a / b) := by
  have hb : 0 < b := lt_trans ha hab
  have hba : 0 < b - a := sub_pos.2 hab
  have hp : pStar a b q = q * b / (b - a) := by
    simp only [pStar, hab, ↓reduceIte, min_eq_right hin]
  have h1 : 1 - a / b = (b - a) / b := by field_simp
  rw [hp, h1, ← Real.log_inv, inv_div]
  congr 1
  field_simp

/-! ### The statements -/

/-- **Proposition `target`** (`Statements.prop_target`). -/
theorem prop_target : Statements.prop_target := by
  intro a b q ha hb hq _
  exact ⟨pStar_pos_le_one a b q ha hq, fun p hp0 hp1 => anchorLoss_pStar_le a b q ha hb hq p hp0 hp1⟩

/-- **Eq. `kl-grad` and its regimes** (`Statements.eq_kl_grad`). -/
theorem eq_kl_grad : Statements.eq_kl_grad := by
  refine ⟨fun a b q p hp hq => hasDerivAt_anchorLoss a b q p hp hq, ?_,
    fun a b q hq => throttle_at_init a b q hq,
    fun a b q p hb hp hpq => throttle_gt_of_lt a b q p hb hp hpq,
    fun a b q hb hq => throttle_strictAnti a b q hb hq,
    fun a q p => by simp [throttle],
    fun a b q p ha hab hp hq => ⟨throttle_eq_zero_iff a b q p ha hab hp hq,
      (throttle_sign a b q p ha hab hp hq).1, (throttle_sign a b q p ha hab hp hq).2⟩⟩
  intro ι _ _ _ a b q y v z hq
  exact hasDerivAt_anchored_coord a b q hq y v z

/-- **Corollary `beta`** (`Statements.cor_beta`). -/
theorem cor_beta : Statements.cor_beta := by
  refine ⟨fun w W γ Γ β _ hW hγ hΓ => cap_exists_iff w W γ Γ β hW hγ hΓ, ?_⟩
  intro w W γ Γ β q _ hW hγ hΓ hβ hq0 hq1
  exact pStar_lt_one_iff_beta w W γ Γ β q hW hγ hΓ hβ hq0 hq1

/-- **Corollary `budget`** (`Statements.cor_budget`). -/
theorem cor_budget : Statements.cor_budget := by
  refine ⟨fun w W γ Γ β hW hγ hΓ hβ => ratio_eq_betaBar_div w W γ Γ β hW hγ hΓ hβ,
    fun a b q ha _ hq => shed_eq_log_ratio a b q ha hq,
    fun a b q ha hab hq => ⟨budget_bound a b q ha hab hq, budget_eq_of_interior a b q ha hab hq⟩,
    fun a b q hab _ => budget_uncapped a b q hab,
    fun bb β₁ β₂ h0 h1 h12 => budget_anti_in_beta bb β₁ β₂ h0 h1 h12,
    fun bb₁ bb₂ β h0 h12 h2 => budget_mono_in_betaBar bb₁ bb₂ β h0 h12 h2⟩

end WebKD
