import WebkdLean.Defs

set_option autoImplicit false
set_option relaxedAutoImplicit false

/-!
# Statements of the WebKD theory

One closed `def <label> : Prop` per result of the paper, named after its LaTeX label
(`prop:tstar` ↦ `prop_tstar`). Every hypothesis is written inside the proposition. The theorem
`WebKD.<label>` proves `WebKD.Statements.<label>`, so the kernel checks that what is proved is
exactly what is stated here. This file and `Defs.lean` are frozen after statement approval.
-/

namespace WebKD.Statements

open WebKD Finset

/-! ## Appendix `tstar`: the first fatal step -/

/-- **Lemma `mono`** (recoverability is absorbing): along a trajectory, within the budget,
`ν_t = 0` implies `ν_{t+1} = 0`. -/
def lem_mono : Prop :=
  ∀ {S A : Type} (H : Harness S A) (τ : Trajectory S A) (T t : ℕ),
    t + 1 ≤ T → ¬ τ.nu H T t → ¬ τ.nu H T (t + 1)

/-- **Proposition `tstar`.** Along a failed trajectory of a task solvable from its initial state
(`ν_0 = 1`), `ν = 0` at the episode's end, and `t⋆ = min {t : ν_t = 0}` exists, satisfies
`1 ≤ t⋆ ≤ n`, every step before it leaves the task solvable, every step from it on (up to the end)
leaves it unsolvable, at step `t⋆` a recoverability-preserving action existed but the executed one
was not, and `t⋆` is the only index with these first two properties. -/
def prop_tstar : Prop :=
  ∀ {S A : Type} (H : Harness S A) (τ : Trajectory S A) (T : ℕ),
    τ.Failed H T → τ.nu H T 0 →
    ¬ τ.nu H T τ.n ∧
    ∃ tstar : ℕ, 1 ≤ tstar ∧ tstar ≤ τ.n ∧ ¬ τ.nu H T tstar ∧
      (∀ t, t < tstar → τ.nu H T t) ∧
      (∀ t, tstar ≤ t → t ≤ τ.n → ¬ τ.nu H T t) ∧
      (∃ a : A, H.Recoverable (H.step (τ.state H (tstar - 1)) a) (T - tstar)) ∧
      ¬ H.Recoverable (H.step (τ.state H (tstar - 1)) (τ.act (tstar - 1))) (T - tstar) ∧
      (∀ t', ¬ τ.nu H T t' → (∀ t, t < t' → τ.nu H T t) → t' = tstar)

/-! ## Section `kl`: the anchored objective -/

/-- **Proposition `target`.** With `a > 0` (the floor `φ > 0`), `b ≥ 0`, `0 < q ≤ 1`, the per-token
target `p⋆` of Eq. `pstar` lies in `(0, 1]` and minimizes the position's contribution
`L_i(p)` over `p ∈ (0, 1]`. -/
def prop_target : Prop :=
  ∀ a b q : ℝ, 0 < a → 0 ≤ b → 0 < q → q ≤ 1 →
    (0 < pStar a b q ∧ pStar a b q ≤ 1) ∧
    ∀ p : ℝ, 0 < p → p ≤ 1 → anchorLoss a b q (pStar a b q) ≤ anchorLoss a b q p

/-- **Eq. `kl-grad`** and the four regimes listed after it. (1) `dL_i/dp = -c_i(p)/p`.
(2) Coordinate form: `∂L_i/∂z_v = c_i (softmax(z)_v - 𝟙[v = y])`, i.e. `∇_z L_i = c_i (p - e_y)`.
(3) At initialization (`p = q`) the throttle is `a`. (4) Below the reference (`p < q`, `b > 0`) it
exceeds `a`. (5) For `b > 0` it is strictly decreasing in `p`. (6) With `b = 0` (unanchored
tokens) it is the constant `a`. (7) For `0 < a < b` it vanishes exactly at `p = q b/(b - a)`, is
positive below that point and negative above it. -/
def eq_kl_grad : Prop :=
  (∀ a b q p : ℝ, 0 < p → 0 < q →
      HasDerivAt (anchorLoss a b q) (-(throttle a b q p) / p) p) ∧
  (∀ {ι : Type} [Fintype ι] [DecidableEq ι] [Nonempty ι] (a b q : ℝ) (y v : ι) (z : ι → ℝ),
      0 < q →
      HasDerivAt (fun t => anchorLoss a b q (softmax (Function.update z v t) y))
        (throttle a b q (softmax z y) * (softmax z v - if v = y then 1 else 0)) (z v)) ∧
  (∀ a b q : ℝ, 0 < q → throttle a b q q = a) ∧
  (∀ a b q p : ℝ, 0 < b → 0 < p → p < q → a < throttle a b q p) ∧
  (∀ a b q : ℝ, 0 < b → 0 < q → StrictAntiOn (throttle a b q) (Set.Ioi 0)) ∧
  (∀ a q p : ℝ, throttle a 0 q p = a) ∧
  (∀ a b q p : ℝ, 0 < a → a < b → 0 < p → 0 < q →
      (throttle a b q p = 0 ↔ p = q * b / (b - a)) ∧
      (p < q * b / (b - a) → 0 < throttle a b q p) ∧
      (q * b / (b - a) < p → throttle a b q p < 0))

/-- **Corollary `beta`.** With `a = w/W` and `b = β γ/Γ` (`w > 0` by the floor `φ`, `W = ∑ w`,
`Γ = ∑ γ`): the interior
optimum exists (`a < b`) iff `β > β̄`; and it lies below `1`, so the cap actually binds
(`p⋆ < 1`), iff `β > β̄/(1 - q)`, i.e. `b (1 - q) > a`. -/
def cor_beta : Prop :=
  (∀ w W γ Γ β : ℝ, 0 < w → 0 < W → 0 < γ → 0 < Γ →
      (w / W < β * γ / Γ ↔ betaBar w W γ Γ < β)) ∧
  (∀ w W γ Γ β q : ℝ, 0 < w → 0 < W → 0 < γ → 0 < Γ → 0 ≤ β → 0 < q → q < 1 →
      (pStar (w / W) (β * γ / Γ) q < 1 ↔ betaBar w W γ Γ / (1 - q) < β))

/-- **Corollary `budget`.** (1) `a/b = β̄/β`. (2) The premium shed between initialization and the
per-token optimum is `CE_i(q) - CE_i(p⋆) = log(p⋆/q)`. (3) Capped case (`a < b`, i.e. `β > β̄`):
`log(p⋆/q) ≤ -log(1 - a/b)`, with equality when the interior cap `q b/(b - a)` is at most `1`.
(4) Uncapped case (`b ≤ a`): `log(p⋆/q) = -log q`, the full `CE_i` at initialization. (5) The
budget `-log(1 - β̄/β)` decreases in `β` and (6) increases in `β̄`. -/
def cor_budget : Prop :=
  (∀ w W γ Γ β : ℝ, 0 < W → 0 < γ → 0 < Γ → 0 < β →
      (w / W) / (β * γ / Γ) = betaBar w W γ Γ / β) ∧
  (∀ a b q : ℝ, 0 < a → 0 ≤ b → 0 < q →
      (-Real.log q) - (-Real.log (pStar a b q)) = Real.log (pStar a b q / q)) ∧
  (∀ a b q : ℝ, 0 < a → a < b → 0 < q →
      Real.log (pStar a b q / q) ≤ -Real.log (1 - a / b) ∧
      (q * b / (b - a) ≤ 1 → Real.log (pStar a b q / q) = -Real.log (1 - a / b))) ∧
  (∀ a b q : ℝ, b ≤ a → 0 < q → Real.log (pStar a b q / q) = -Real.log q) ∧
  (∀ bb β₁ β₂ : ℝ, 0 ≤ bb → bb < β₁ → β₁ ≤ β₂ →
      -Real.log (1 - bb / β₂) ≤ -Real.log (1 - bb / β₁)) ∧
  (∀ bb₁ bb₂ β : ℝ, 0 ≤ bb₁ → bb₁ ≤ bb₂ → bb₂ < β →
      -Real.log (1 - bb₁ / β) ≤ -Real.log (1 - bb₂ / β))

/-! ## Section `actw` and Appendix `WebKDLossDetails`: the weighted objective -/

/-- **Uniform weights recover SFT** (the sentence after Eq. `sw`): for every constant `c > 0`,
`w_i ≡ c` gives `L_SW = L_SFT` identically. -/
def eq_sw : Prop :=
  ∀ {ι : Type} (T : Finset ι) (CE : ι → ℝ) (c : ℝ), 0 < c → LSW T (fun _ => c) CE = LSFT T CE

/-- **Remark `resample`.** For nonnegative weights with positive sum, `q_w(i) = w_i/∑_j w_j` is a
probability distribution over the supervised positions and `L_SW = 𝔼_{i ∼ q_w}[CE_i]`. -/
def rem_resample : Prop :=
  ∀ {ι : Type} (T : Finset ι) (w CE : ι → ℝ), (∀ i ∈ T, 0 ≤ w i) → 0 < ∑ i ∈ T, w i →
    (∀ i ∈ T, 0 ≤ w i / ∑ j ∈ T, w j) ∧ (∑ i ∈ T, w i / ∑ j ∈ T, w j) = 1 ∧
    LSW T w CE = ∑ i ∈ T, (w i / ∑ j ∈ T, w j) * CE i

/-- **Eq. `actw-map`.** (1) Every span weight lies in `[φ, 1]`. (2) Spread `σ = 0` gives the
midpoint `(1 + φ)/2`, whatever the rank; (2b) in particular a step whose segments all tie, such as
a single-segment step, has spread `0` and gets the midpoint on every segment. (3) With
normalized average ranks, the mean span weight of every step with at least two segments is
`(1 + φ)/2`, for every spread. (4) The map assigns weights in the order of `Δ` (for every spread `σ ≥ 0`). (5) Outside the
dead zone (`σ ≥ τ_dz`), a unique top-scoring segment gets weight `1` and (6) a unique
bottom-scoring segment gets the floor `φ`. -/
def eq_actw_map : Prop :=
  (∀ φ τ σ r : ℝ, φ ≤ 1 → φ ≤ rankMap φ τ σ r ∧ rankMap φ τ σ r ≤ 1) ∧
  (∀ φ τ r : ℝ, φ ≤ 1 → rankMap φ τ 0 r = (1 + φ) / 2) ∧
  (∀ {ι : Type} (s : Finset ι) (Δ : ι → ℝ) (φ τ : ℝ) (i : ι), i ∈ s → (∀ j ∈ s, Δ j = Δ i) →
      φ ≤ 1 → rankMap φ τ (spread s Δ) (normRank s Δ i) = (1 + φ) / 2) ∧
  (∀ {ι : Type} (s : Finset ι) (Δ : ι → ℝ) (φ τ σ : ℝ), 2 ≤ s.card → 0 ≤ φ → φ ≤ 1 → 0 < τ →
      0 ≤ σ → (∑ i ∈ s, rankMap φ τ σ (normRank s Δ i)) / s.card = (1 + φ) / 2) ∧
  (∀ {ι : Type} [Fintype ι] (Δ : ι → ℝ) (φ τ σ : ℝ), 2 ≤ Fintype.card ι → 0 ≤ φ → φ ≤ 1 →
      0 < τ → 0 ≤ σ → Monovary (fun i => rankMap φ τ σ (normRank Finset.univ Δ i)) Δ) ∧
  (∀ {ι : Type} [Fintype ι] (Δ : ι → ℝ) (φ τ σ : ℝ) (i : ι), 2 ≤ Fintype.card ι → 0 ≤ φ →
      φ ≤ 1 → 0 < τ → τ ≤ σ → (∀ j, j ≠ i → Δ j < Δ i) →
      rankMap φ τ σ (normRank Finset.univ Δ i) = 1) ∧
  (∀ {ι : Type} [Fintype ι] (Δ : ι → ℝ) (φ τ σ : ℝ) (i : ι), 2 ≤ Fintype.card ι → 0 ≤ φ →
      φ ≤ 1 → 0 < τ → τ ≤ σ → (∀ j, j ≠ i → Δ i < Δ j) →
      rankMap φ τ σ (normRank Finset.univ Δ i) = φ)

/-! ## Appendix `theory-update`: premium split and rank weighting -/

/-- **Lemma `sandwich`.** `∇_z CE = p - e_y` coordinatewise, and
`1 - e^{-CE} ≤ ‖p - e_y‖₂ ≤ √2 (1 - e^{-CE})`. -/
def lem_sandwich : Prop :=
  ∀ {ι : Type} [Fintype ι] [DecidableEq ι] [Nonempty ι] (y : ι) (z : ι → ℝ),
    (∀ v, HasDerivAt (fun t => nll y (Function.update z v t))
        (softmax z v - if v = y then 1 else 0) (z v)) ∧
    1 - Real.exp (-nll y z) ≤ logitGradNorm y z ∧
    logitGradNorm y z ≤ Real.sqrt 2 * (1 - Real.exp (-nll y z))

/-- **Proposition `split`.** (i) For disjoint nonempty position sets, the action tokens' share of
the token-pooled uniform loss restricted to `I_rat ∪ I_act` is
`n_act log PPL_act / (n_act log PPL_act + n_rat log PPL_rat)`, with
`log PPL_C = (1/n_C) ∑_{i ∈ I_C} CE_i`. (ii) If every rationale position has `CE_i ≤ Λ`
(Assumption `cap`), the action's share of the logit-gradient mass `G_C = ∑_{i ∈ I_C} ‖∇_z CE_i‖₂`
is at most `X/(X + Y)`, `X = √2 ∑_act CE_i`, `Y = ((1 - e^{-Λ})/Λ) ∑_rat CE_i`. Each position has
its own logits `z i` over the vocabulary `V`. -/
def prop_split : Prop :=
  (∀ {ι : Type} [DecidableEq ι] (Irat Iact : Finset ι) (CE : ι → ℝ),
      Disjoint Irat Iact → Irat.Nonempty → Iact.Nonempty →
      ((1 / ((Irat.card : ℝ) + Iact.card)) * ∑ i ∈ Iact, CE i)
          / ((1 / ((Irat.card : ℝ) + Iact.card)) * ∑ i ∈ Irat ∪ Iact, CE i)
        = (Iact.card : ℝ) * ((∑ i ∈ Iact, CE i) / Iact.card)
          / ((Iact.card : ℝ) * ((∑ i ∈ Iact, CE i) / Iact.card)
              + (Irat.card : ℝ) * ((∑ i ∈ Irat, CE i) / Irat.card))) ∧
  (∀ {ι V : Type} [Fintype V] [DecidableEq V] [Nonempty V] (Irat Iact : Finset ι)
      (z : ι → V → ℝ) (y : ι → V) (Λ : ℝ), 0 < Λ → (∀ i ∈ Irat, nll (y i) (z i) ≤ Λ) →
      (∑ i ∈ Iact, logitGradNorm (y i) (z i))
          / ((∑ i ∈ Iact, logitGradNorm (y i) (z i)) + ∑ i ∈ Irat, logitGradNorm (y i) (z i))
        ≤ (Real.sqrt 2 * ∑ i ∈ Iact, nll (y i) (z i))
          / ((Real.sqrt 2 * ∑ i ∈ Iact, nll (y i) (z i))
              + (1 - Real.exp (-Λ)) / Λ * ∑ i ∈ Irat, nll (y i) (z i)))

/-- **Theorem `rank`.** (1) Under Assumptions `additive` and `emit`, the deployment probability of
the corrected action is `e^{α₀} ∏_j (1 - κ_j + κ_j e^{Δ_j})`. (2) Under Assumption `additive` the
leave-one-out score recovers `Δ_j`. (3) Eq. `mu`: the derivative of the log deployment probability
in `log κ_j` (the other `κ_i` fixed) is `μ_j`. (4) `μ_j` is strictly increasing in `Δ_j`,
(5) zero at `Δ_j = 0`, (6) negative for `Δ_j < 0`. (7) With a common `κ ∈ (0,1)`, assigning the
weights in the order of `Δ` maximizes `∑_j μ_j v_j` over all permutations of the same weights, and
(8) the order-respecting assignments depend on `Δ` only through its order. -/
def thm_rank : Prop :=
  (∀ {ι : Type} [Fintype ι] [DecidableEq ι] (α₀ : ℝ) (κ Δ : ι → ℝ),
      ∑ U : Finset ι, ((∏ j ∈ U, κ j) * ∏ j ∈ Finset.univ \ U, (1 - κ j))
          * Real.exp (α₀ + ∑ j ∈ U, Δ j)
        = Real.exp α₀ * ∏ j, (1 - κ j + κ j * Real.exp (Δ j))) ∧
  (∀ {ι : Type} [Fintype ι] [DecidableEq ι] (α₀ : ℝ) (Δ : ι → ℝ) (j : ι),
      (α₀ + ∑ i, Δ i) - (α₀ + ∑ i ∈ Finset.univ.erase j, Δ i) = Δ j) ∧
  (∀ {ι : Type} [Fintype ι] [DecidableEq ι] (α₀ : ℝ) (κ Δ : ι → ℝ) (j : ι),
      (∀ i, 0 < κ i ∧ κ i < 1) →
      HasDerivAt
        (fun u => Real.log (Real.exp α₀ * ∏ i, (1 - Function.update κ j (Real.exp u) i
          + Function.update κ j (Real.exp u) i * Real.exp (Δ i))))
        (mu (κ j) (Δ j)) (Real.log (κ j))) ∧
  (∀ κ : ℝ, 0 < κ → κ < 1 → StrictMono (mu κ)) ∧
  (∀ κ : ℝ, mu κ 0 = 0) ∧
  (∀ κ Δ : ℝ, 0 < κ → κ < 1 → Δ < 0 → mu κ Δ < 0) ∧
  (∀ {ι : Type} [Fintype ι] (κ : ℝ) (Δ v : ι → ℝ), 0 < κ → κ < 1 → Monovary v Δ →
      ∀ σ : Equiv.Perm ι, ∑ j, mu κ (Δ j) * v (σ j) ≤ ∑ j, mu κ (Δ j) * v j) ∧
  (∀ {ι : Type} (Δ Δ' v : ι → ℝ), (∀ i j, Δ i < Δ j ↔ Δ' i < Δ' j) →
      (Monovary v Δ ↔ Monovary v Δ'))

/-! ## Appendix `bound`: the single-round relabeling bound

The finite model: a task distribution `pop` on a finite set `Ω` of tasks, a finite set `Y` of
structured responses, histories given by response prefixes (Assumption `det`), a step budget
`T`, and a reward `R ω` in `[0,1]` read off the complete response sequence (Assumptions `budget`,
`reward`). The success rate of a policy `π` is `ρ(π) = ∑_ω P(ω) · expectAt (π ω) T (R ω) []`. -/

/-- **Lemma `imit`.** For any two policies `π` and `π̃`,
`ρ(π̃) - ρ(π) ≤ ∑_{t=1}^{T} 𝔼_{h ∼ d^t_{π̃}}[TV(π(·|h), π̃(·|h))]`, the history distribution
including the draw of the task. -/
def lem_imit : Prop :=
  ∀ {Ω Y : Type} [Fintype Ω] [Fintype Y] (pop : Population Ω)
    (π π' : Ω → List Y → Y → ℝ) (R : Ω → List Y → ℝ) (T : ℕ),
    (∀ ω, IsPolicy (π ω)) → (∀ ω, IsPolicy (π' ω)) → (∀ ω h, 0 ≤ R ω h ∧ R ω h ≤ 1) →
    (∑ ω, pop.P ω * expectAt (π' ω) T (R ω) []) - (∑ ω, pop.P ω * expectAt (π ω) T (R ω) [])
      ≤ ∑ t ∈ range T, ∑ ω, pop.P ω * expectAt (π' ω) t (fun h => tv (π ω h) (π' ω h)) []

/-- **Theorem `bound-formal`.** Let `logged ω` be the seed's logged rollout on task `ω` (length
`T`; an episode that ended early is padded with responses that do not change its reward), `solved ω`
mean that it succeeded, `tl ω ∈ {1, …, T}` be the localization `t'`, `late` any classification
of the failed tasks (in the paper, `t' > t⋆`), `πT` the teacher-as-agent and `πWD` the trained
student, and let `π̃ = relabel …` be the relabeled process. With `ρ_BC = P(solved)`,
`ε_loc = P(late | ¬solved)`, `ε_corr` the teacher's mean failure probability from `h_{t'}` over
failed, not-late tasks, and `δ_t` the expected TV between `πWD` and `π̃` under `π̃`'s step-`t`
history distribution:
(1) `1 - ρ(πWD) ≤ (1 - ρ_BC)(ε_loc + ε_corr) + ∑_t δ_t`;
(2) if every late localization leaves a dead prefix (no completion succeeds, which Proposition
`tstar` gives for `t' > t⋆`), the relabeled process fails with probability exactly
`(1 - ρ_BC)(ε_loc + (1 - ε_loc) ε_corr)`. -/
def thm_bound_formal : Prop :=
  ∀ {Ω Y : Type} [Fintype Ω] [Fintype Y] [DecidableEq Y] [Inhabited Y]
    (pop : Population Ω) (T : ℕ) (R : Ω → List Y → ℝ)
    (solved late : Ω → Prop) [DecidablePred solved] [DecidablePred late]
    (logged : Ω → List Y) (tl : Ω → ℕ) (πT πWD : Ω → List Y → Y → ℝ),
    (∀ ω h, 0 ≤ R ω h ∧ R ω h ≤ 1) →
    (∀ ω, IsPolicy (πT ω)) → (∀ ω, IsPolicy (πWD ω)) →
    (∀ ω, (logged ω).length = T) → (∀ ω, 1 ≤ tl ω ∧ tl ω ≤ T) →
    (∀ ω, solved ω ↔ R ω (logged ω) = 1) →
    let πrel := relabel solved logged tl πT
    let ρ := fun π : Ω → List Y → Y → ℝ => ∑ ω, pop.P ω * expectAt (π ω) T (R ω) []
    let ρBC := pop.prob solved
    let εloc := pop.cond late (fun ω => ¬ solved ω)
    let f := fun ω => 1 - expectAt (πT ω) (T - (tl ω - 1)) (R ω) ((logged ω).take (tl ω - 1))
    let εcorr := (∑ ω, if ¬ solved ω ∧ ¬ late ω then pop.P ω * f ω else 0)
      / pop.prob (fun ω => ¬ solved ω ∧ ¬ late ω)
    let δ := fun t => ∑ ω, pop.P ω * expectAt (πrel ω) t (fun h => tv (πWD ω h) (πrel ω h)) []
    (1 - ρ πWD ≤ (1 - ρBC) * (εloc + εcorr) + ∑ t ∈ range T, δ t) ∧
    ((∀ ω, ¬ solved ω → late ω → ∀ c : List Y, c.length = T - (tl ω - 1) →
        R ω ((logged ω).take (tl ω - 1) ++ c) = 0) →
      1 - ρ πrel = (1 - ρBC) * (εloc + (1 - εloc) * εcorr))

/-- **Corollary `regimes`.** For any nonnegative per-task, per-step disagreement `e ω t`, split the
(task, step) pairs into solved tasks, failed tasks up to and including `t'`, and failed tasks after
`t'`; with `ε_ret`, `ε_fit`, `ε_rec` the per-step averages over the three cells,
(1) `∑_t ∑_ω P(ω) e ω t ≤ T (ρ_BC ε_ret + (1 - ρ_BC)(ε_fit + ε_rec))`; (2) exactly, the total is
the mass-weighted sum of the cell averages, (3) the solved cell has mass `T ρ_BC` and (4) the two
failed cells together have mass `T (1 - ρ_BC)`, so (1) replaces their convex combination by a sum. Step `t` is the paper's step
`t + 1`, so "up to and including `t'`" is `t < t'`. -/
def cor_regimes : Prop :=
  ∀ {Ω : Type} [Fintype Ω] (pop : Population Ω) (T : ℕ) (solved : Ω → Prop)
    [DecidablePred solved] (tl : Ω → ℕ) (e : Ω → ℕ → ℝ), (∀ ω t, 0 ≤ e ω t) →
    (∑ t ∈ range T, ∑ ω, pop.P ω * e ω t
      ≤ T * (pop.prob solved * cellAvg pop T (fun ω _ => solved ω) e
          + (1 - pop.prob solved)
            * (cellAvg pop T (fun ω t => ¬ solved ω ∧ t < tl ω) e
              + cellAvg pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t) e))) ∧
    (∑ t ∈ range T, ∑ ω, pop.P ω * e ω t
      = cellMass pop T (fun ω _ => solved ω) * cellAvg pop T (fun ω _ => solved ω) e
        + cellMass pop T (fun ω t => ¬ solved ω ∧ t < tl ω)
          * cellAvg pop T (fun ω t => ¬ solved ω ∧ t < tl ω) e
        + cellMass pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t)
          * cellAvg pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t) e) ∧
    cellMass pop T (fun ω _ => solved ω) = T * pop.prob solved ∧
    cellMass pop T (fun ω t => ¬ solved ω ∧ t < tl ω)
      + cellMass pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t) = T * (1 - pop.prob solved)

/-- **Theorem `bound`** (main text). With `ε̄ = (1/T) ∑_t δ_t` the per-step disagreement:
`1 - ρ(πWD) ≤ (1 - ρ_BC)(ε_loc + ε_corr) + T ε̄` and
`ε̄ ≤ ρ_BC ε_ret + (1 - ρ_BC)(ε_fit + ε_rec)`, the cells of Corollary `regimes` evaluated on the
disagreement between `πWD` and the relabeled process. -/
def thm_bound : Prop :=
  ∀ {Ω Y : Type} [Fintype Ω] [Fintype Y] [DecidableEq Y] [Inhabited Y]
    (pop : Population Ω) (T : ℕ) (R : Ω → List Y → ℝ)
    (solved late : Ω → Prop) [DecidablePred solved] [DecidablePred late]
    (logged : Ω → List Y) (tl : Ω → ℕ) (πT πWD : Ω → List Y → Y → ℝ),
    (∀ ω h, 0 ≤ R ω h ∧ R ω h ≤ 1) →
    (∀ ω, IsPolicy (πT ω)) → (∀ ω, IsPolicy (πWD ω)) →
    (∀ ω, (logged ω).length = T) → (∀ ω, 1 ≤ tl ω ∧ tl ω ≤ T) →
    (∀ ω, solved ω ↔ R ω (logged ω) = 1) →
    let πrel := relabel solved logged tl πT
    let ρWD := ∑ ω, pop.P ω * expectAt (πWD ω) T (R ω) []
    let ρBC := pop.prob solved
    let εloc := pop.cond late (fun ω => ¬ solved ω)
    let f := fun ω => 1 - expectAt (πT ω) (T - (tl ω - 1)) (R ω) ((logged ω).take (tl ω - 1))
    let εcorr := (∑ ω, if ¬ solved ω ∧ ¬ late ω then pop.P ω * f ω else 0)
      / pop.prob (fun ω => ¬ solved ω ∧ ¬ late ω)
    let e := fun ω t => expectAt (πrel ω) t (fun h => tv (πWD ω h) (πrel ω h)) []
    let εbar := (1 / (T : ℝ)) * ∑ t ∈ range T, ∑ ω, pop.P ω * e ω t
    let εret := cellAvg pop T (fun ω _ => solved ω) e
    let εfit := cellAvg pop T (fun ω t => ¬ solved ω ∧ t < tl ω) e
    let εrec := cellAvg pop T (fun ω t => ¬ solved ω ∧ tl ω ≤ t) e
    (1 - ρWD ≤ (1 - ρBC) * (εloc + εcorr) + T * εbar) ∧
    εbar ≤ ρBC * εret + (1 - ρBC) * (εfit + εrec)

end WebKD.Statements
