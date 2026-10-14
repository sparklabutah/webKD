import Mathlib

set_option autoImplicit false
set_option relaxedAutoImplicit false

/-!
# Definitions of the WebKD formalization

Every object that appears in `Statements.lean` is defined here, with the paper equation or label
it formalizes. This file and `Statements.lean` are frozen after statement approval (Gate A); helper
definitions used only inside proofs live next to those proofs.

Paper ↔ objects:
* harness and trajectories (Definition `recover`, Assumptions `budget`, `det`, `reward`):
  `Harness`, `Harness.run`, `Harness.Recoverable`, `Trajectory`, `Trajectory.state`,
  `Trajectory.nu`, `Trajectory.Failed`;
* weighted objective and rank map (Eqs. `sw`, `actw-map`): `LSW`, `LSFT`, `rankMap`,
  `avgRank`, `normRank`, `spread`;
* softmax cross-entropy (Lemma `sandwich`): `softmax`, `nll`, `logitGradNorm`;
* anchored objective (Eqs. `kl`, `kl-grad`, `pstar`, `budget`): `anchorLoss`, `throttle`, `pStar`,
  `betaBar`;
* Theorem `rank`: `mu`;
* the single-round bound (Theorem `bound-formal`, Lemma `imit`, Corollary `regimes`):
  `Population`, `Population.prob`, `Population.cond`, `IsPolicy`, `tv`, `expectAt`, `pointMass`,
  `relabel`, `cellMass`, `cellAvg`.
-/

namespace WebKD

open Finset

/-! ## The harness (Definition `recover`, Assumptions `budget`, `det`, `reward`) -/

/-- A deterministic web harness (Assumption `det`): `step` is the deterministic transition `𝕋`
composed with the deterministic parser `f_parse`, `done` marks that a terminal action was issued,
and `R` is the terminal binary reward read off the final state (Assumption `reward`). An ended
episode is absorbing, with its reward fixed (Assumption `budget`). -/
structure Harness (S A : Type) where
  /-- deterministic transition `𝕋` composed with the deterministic parser `f_parse` -/
  step : S → A → S
  /-- the episode has ended (a terminal action was issued) -/
  done : S → Prop
  /-- the terminal binary reward, read off the final state -/
  R : S → Prop
  /-- an ended episode is absorbing, with its reward fixed (Assumption `budget`) -/
  absorbing : ∀ s a, done s → step s a = s

/-- Run an action sequence from a state. -/
def Harness.run {S A : Type} (H : Harness S A) (s : S) : List A → S
  | [] => s
  | a :: as => H.run (H.step s a) as

/-- **Definition `recover`.** From state `s` with remaining step budget `k`, some action sequence
of length at most `k` ends the episode with reward `1`: the sequence either issues a terminal
action (`done`) or uses up the whole remaining budget (length exactly `k`), and the reward read off
the state it reaches is `1`. -/
def Harness.Recoverable {S A : Type} (H : Harness S A) (s : S) (k : ℕ) : Prop :=
  ∃ seq : List A, seq.length ≤ k ∧ (H.done (H.run s seq) ∨ seq.length = k) ∧ H.R (H.run s seq)

/-- A logged trajectory: the initial state `s_(0)`, the executed actions (`act t` is the action
`a_{t+1}` executed at step `t+1`), and the number `n` of executed steps. -/
structure Trajectory (S A : Type) where
  s0 : S
  act : ℕ → A
  n : ℕ

/-- `s_(t)`: the underlying browser state after the first `t` actions of `τ`. -/
def Trajectory.state {S A : Type} (τ : Trajectory S A) (H : Harness S A) : ℕ → S
  | 0 => τ.s0
  | t + 1 => H.step (τ.state H t) (τ.act t)

/-- `ν_t` of Definition `recover` for step budget `T`: the state after `t` actions is recoverable
within the remaining budget `T - t`. -/
def Trajectory.nu {S A : Type} (τ : Trajectory S A) (H : Harness S A) (T t : ℕ) : Prop :=
  H.Recoverable (τ.state H t) (T - t)

/-- A failed episode under budget `T`: at most `T` steps were executed, the episode ended (a
terminal action was issued, or the budget was exhausted), and the terminal reward is `0`. -/
structure Trajectory.Failed {S A : Type} (τ : Trajectory S A) (H : Harness S A) (T : ℕ) : Prop where
  le_budget : τ.n ≤ T
  ended : H.done (τ.state H τ.n) ∨ τ.n = T
  no_reward : ¬ H.R (τ.state H τ.n)

/-! ## The weighted objective and the rank map (Eqs. `sw`, `actw-map`) -/

/-- Eq. `sw`: the weighted-mean imitation term `L_SW = ∑ w_i CE_i / ∑ w_i` over the supervised
tokens `T`. -/
noncomputable def LSW {ι : Type} (T : Finset ι) (w CE : ι → ℝ) : ℝ :=
  (∑ i ∈ T, w i * CE i) / (∑ i ∈ T, w i)

/-- The per-token-mean SFT loss on the supervised tokens `T` (Eq. `sft` on one record). -/
noncomputable def LSFT {ι : Type} (T : Finset ι) (CE : ι → ℝ) : ℝ :=
  (1 / (T.card : ℝ)) * ∑ i ∈ T, CE i

/-- Eq. `actw-map`: `v_s = clip_{[φ,1]}(φ + (1-φ)(1/2 + (r_s - 1/2) min(1, σ/τ_dz)))`. -/
noncomputable def rankMap (φ τ σ r : ℝ) : ℝ :=
  max φ (min 1 (φ + (1 - φ) * (1 / 2 + (r - 1 / 2) * min 1 (σ / τ))))

/-- The within-step average rank of segment `i` among the segments `s` scored by `Δ`
(`1`-indexed, ties sharing their mean rank): `#{j : Δ_j < Δ_i} + (#{j : Δ_j = Δ_i} + 1)/2`. -/
noncomputable def avgRank {ι : Type} (s : Finset ι) (Δ : ι → ℝ) (i : ι) : ℝ :=
  ((s.filter fun j => Δ j < Δ i).card : ℝ) + (((s.filter fun j => Δ j = Δ i).card : ℝ) + 1) / 2

/-- The within-step spread `σ = max_s Δ_s - min_s Δ_s` of Eq. `actw-map` (`0` for an empty
step). -/
noncomputable def spread {ι : Type} (s : Finset ι) (Δ : ι → ℝ) : ℝ :=
  if h : s.Nonempty then s.sup' h Δ - s.inf' h Δ else 0

/-- The normalized average rank `r_s = (rank - 1)/(n - 1) ∈ [0,1]` of Eq. `actw-map`. -/
noncomputable def normRank {ι : Type} (s : Finset ι) (Δ : ι → ℝ) (i : ι) : ℝ :=
  (avgRank s Δ i - 1) / (s.card - 1)

/-! ## Softmax cross-entropy (Lemma `sandwich`) -/

/-- Softmax of the logits `z`. -/
noncomputable def softmax {ι : Type} [Fintype ι] (z : ι → ℝ) (v : ι) : ℝ :=
  Real.exp (z v) / ∑ u, Real.exp (z u)

/-- Per-token cross-entropy `CE = -log softmax(z)_y`. -/
noncomputable def nll {ι : Type} [Fintype ι] (y : ι) (z : ι → ℝ) : ℝ := -Real.log (softmax z y)

/-- The Euclidean norm `‖p - e_y‖₂` of the logit gradient `∇_z CE = p - e_y`, `p = softmax z`. -/
noncomputable def logitGradNorm {ι : Type} [Fintype ι] [DecidableEq ι] (y : ι) (z : ι → ℝ) : ℝ :=
  Real.sqrt (∑ v, (softmax z v - if v = y then 1 else 0) ^ 2)

/-! ## The anchored objective (Eqs. `kl`, `kl-grad`, `pstar`, `budget`) -/

/-- Contribution of one supervised position to Eq. `kl` as a function of `p = π_θ(y_i)`:
`L_i(p) = -a log p + b (q/p + log(p/q) - 1)`, with `a = w_i/∑w`, `b = β γ_i/∑γ`,
`q = π_ref(y_i)` (Appendix `proof-target`). -/
noncomputable def anchorLoss (a b q p : ℝ) : ℝ :=
  -a * Real.log p + b * (q / p + Real.log (p / q) - 1)

/-- The throttle `c_i(p) = a - b (1 - q/p)` of Eq. `kl-grad`. -/
noncomputable def throttle (a b q p : ℝ) : ℝ := a - b * (1 - q / p)

/-- The per-token target `p⋆` of Eq. `pstar`. -/
noncomputable def pStar (a b q : ℝ) : ℝ :=
  if a < b then min 1 (q * b / (b - a)) else 1

/-- The critical anchor strength `β̄_i = (w_i/γ_i)(∑γ/∑w)` of Corollary `beta`, with `W = ∑ w`
and `Γ = ∑ γ`. -/
noncomputable def betaBar (w W γ Γ : ℝ) : ℝ := (w / γ) * (Γ / W)

/-! ## Theorem `rank` -/

/-- `μ_j` of Eq. `mu`: `κ (e^Δ - 1)/(1 - κ + κ e^Δ)`. -/
noncomputable def mu (κ Δ : ℝ) : ℝ := κ * (Real.exp Δ - 1) / (1 - κ + κ * Real.exp Δ)

/-! ## The single-round bound (Theorem `bound-formal`, Lemma `imit`, Corollary `regimes`) -/

/-- A finite population (the task distribution) with a probability mass function. -/
structure Population (Ω : Type) [Fintype Ω] where
  P : Ω → ℝ
  nonneg : ∀ ω, 0 ≤ P ω
  total : ∑ ω, P ω = 1

/-- Probability of an event. -/
noncomputable def Population.prob {Ω : Type} [Fintype Ω] (pop : Population Ω) (E : Ω → Prop)
    [DecidablePred E] : ℝ :=
  ∑ ω, if E ω then pop.P ω else 0

/-- Conditional probability `P(E | F)`, with the convention `x / 0 = 0`. -/
noncomputable def Population.cond {Ω : Type} [Fintype Ω] (pop : Population Ω) (E F : Ω → Prop)
    [DecidablePred E] [DecidablePred F] : ℝ :=
  pop.prob (fun ω => E ω ∧ F ω) / pop.prob F

/-- A stochastic policy over a finite set `Y` of structured responses. Histories are response
prefixes: under Assumption `det` the observation history `h_t` of a task is a function of the
task and of the responses `y_{1:t-1}` emitted so far, so a policy of the task is a map from
response prefixes to probability vectors over `Y`. -/
def IsPolicy {Y : Type} [Fintype Y] (π : List Y → Y → ℝ) : Prop :=
  ∀ h, (∀ y, 0 ≤ π h y) ∧ ∑ y, π h y = 1

/-- Total variation distance `TV(p, q) = ½ ∑_y |p(y) - q(y)|` between two distributions over the
full structured response. -/
noncomputable def tv {Y : Type} [Fintype Y] (p q : Y → ℝ) : ℝ := (1 / 2) * ∑ y, |p y - q y|

/-- `expectAt π t f h`: the expectation of `f` at the history reached from history `h` after `t`
more responses drawn from `π`. From the empty history, `expectAt π t f []` is
`𝔼_{h ∼ d^{t+1}_π}[f(h)]` for the step-`(t+1)` history distribution of the paper, and
`expectAt π T R []` is the success probability of `π` when `R` is the terminal reward of the
complete response sequence. -/
noncomputable def expectAt {Y : Type} [Fintype Y] (π : List Y → Y → ℝ) :
    ℕ → (List Y → ℝ) → List Y → ℝ
  | 0, f, h => f h
  | t + 1, f, h => ∑ y, π h y * expectAt π t f (h ++ [y])

/-- The point mass on response `y₀`. -/
def pointMass {Y : Type} [DecidableEq Y] (y₀ y : Y) : ℝ := if y = y₀ then 1 else 0

/-- **The relabeled process `π̃` of Appendix `bound`.** On a task whose logged seed rollout
`logged ω` succeeded, it replays that rollout. On a failed task with localization `t' = tl ω`,
it replays the logged responses through step `t' - 1` and then hands control to the
teacher-as-agent `πT ω`, which plays the episode out. Replay emits the logged response at the
current position (`default` past the end of the log, never reached within the budget). -/
noncomputable def relabel {Ω Y : Type} [DecidableEq Y] [Inhabited Y]
    (solved : Ω → Prop) [DecidablePred solved] (logged : Ω → List Y) (tl : Ω → ℕ)
    (πT : Ω → List Y → Y → ℝ) : Ω → List Y → Y → ℝ :=
  fun ω h y =>
    if solved ω ∨ h.length + 1 < tl ω then pointMass ((logged ω).getD h.length default) y
    else πT ω h y

/-- Step mass of a cell: `∑_{t<T} ∑_ω 𝟙[cell] P(ω)`, the expected number of steps it holds. -/
noncomputable def cellMass {Ω : Type} [Fintype Ω] (pop : Population Ω) (T : ℕ)
    (cell : Ω → ℕ → Prop) [∀ ω t, Decidable (cell ω t)] : ℝ :=
  ∑ t ∈ range T, ∑ ω, if cell ω t then pop.P ω else 0

/-- Per-step average of `e ω t` over the (task, step) pairs of a cell, weighted by the task
distribution: `(∑_{t<T} ∑_ω 𝟙[cell] P(ω) e ω t) / (∑_{t<T} ∑_ω 𝟙[cell] P(ω))` (Corollary
`regimes`). Step `t` here is the paper's step `t + 1`. -/
noncomputable def cellAvg {Ω : Type} [Fintype Ω] (pop : Population Ω) (T : ℕ)
    (cell : Ω → ℕ → Prop) [∀ ω t, Decidable (cell ω t)] (e : Ω → ℕ → ℝ) : ℝ :=
  (∑ t ∈ range T, ∑ ω, if cell ω t then pop.P ω * e ω t else 0) /
    (∑ t ∈ range T, ∑ ω, if cell ω t then pop.P ω else 0)

end WebKD
