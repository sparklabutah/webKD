import WebkdLean.Proofs.FatalStep

set_option autoImplicit false
set_option relaxedAutoImplicit false

/-!
# Lemma `imit` and the relabeled process (Appendix `bound`)

Strategy, following the proof of Lemma `imit` in the paper. Histories are response prefixes:
under Assumption `det` the history of a task is a function of its responses so far, and
`expectAt π t f h` averages `f` over the history reached `t` responses after `h`. One step of
the recursion splits the success gap of two policies at `h` into (i) the gap one response later,
averaged under `π̃`, and (ii) `∑_y (π̃(y|h) - π(y|h)) V_π(h y)`. Because `0 ≤ V_π ≤ 1` and both rows
sum to one, (ii) is at most `TV(π(·|h), π̃(·|h))` (`sum_sub_mul_le_tv`). Unrolling (i) over the
remaining steps gives Lemma `imit` for one task (`value_gap_le`), and averaging over tasks gives
it for the task distribution (`lem_imit`). This induction on the remaining steps (the
performance-difference form) proves the same inequality as a step-wise maximal coupling, in which
`TV(π(·|h), π̃(·|h))` is the probability that the two responses separate at `h`; the paper's proof
now follows the induction.

The relabeled process: replaying a logged point mass for `m` steps reaches the `m`-step prefix of
the log (`expectAt_replay`), after which `π̃` coincides with the teacher
(`expectAt_congr_of_length`). Hence `π̃` succeeds with probability `R(logged)` on a solved task
(`relabel_value_solved`) and with the teacher's success probability from `h_{t'}` on a failed one
(`relabel_value_failed`). A dead prefix has value `0` for every policy
(`expectAt_eq_zero_of_dead`). On the harness, an unrecoverable state leaves a dead prefix
(`dead_prefix_of_not_recoverable`), and Proposition `tstar` makes the state after a late
localization unrecoverable (`late_localization_dead`), so the real-valued reward read off the
harness vanishes on every completion of the replayed prefix (`late_localization_dead_reward`),
which is the hypothesis of Theorem `bound-formal` (2).
-/

namespace WebKD

open Finset

section Imit

variable {Y : Type} [Fintype Y]

lemma expectAt_succ (π : List Y → Y → ℝ) (t : ℕ) (f : List Y → ℝ) (h : List Y) :
    expectAt π (t + 1) f h = ∑ y, π h y * expectAt π t f (h ++ [y]) := rfl

/-- Expectations of `[0, 1]`-valued functions under a policy stay in `[0, 1]`. -/
lemma expectAt_mem_unit (π : List Y → Y → ℝ) (hπ : IsPolicy π) (f : List Y → ℝ)
    (hf : ∀ h, 0 ≤ f h ∧ f h ≤ 1) : ∀ t h, 0 ≤ expectAt π t f h ∧ expectAt π t f h ≤ 1 := by
  intro t
  induction t with
  | zero => intro h; exact hf h
  | succ t ih =>
    intro h
    rw [expectAt_succ]
    constructor
    · exact Finset.sum_nonneg fun y _ => mul_nonneg ((hπ h).1 y) (ih _).1
    · calc ∑ y, π h y * expectAt π t f (h ++ [y]) ≤ ∑ y, π h y * 1 :=
            Finset.sum_le_sum fun y _ => mul_le_mul_of_nonneg_left (ih _).2 ((hπ h).1 y)
        _ = 1 := by simp [(hπ h).2]

/-- Expectations of nonnegative functions under a policy are nonnegative. -/
lemma expectAt_nonneg (π : List Y → Y → ℝ) (hπ : IsPolicy π) (f : List Y → ℝ)
    (hf : ∀ h, 0 ≤ f h) : ∀ t h, 0 ≤ expectAt π t f h := by
  intro t
  induction t with
  | zero => intro h; exact hf h
  | succ t ih =>
    intro h
    rw [expectAt_succ]
    exact Finset.sum_nonneg fun y _ => mul_nonneg ((hπ h).1 y) (ih _)

/-- Total variation is nonnegative. -/
lemma tv_nonneg (p q : Y → ℝ) : 0 ≤ tv p q :=
  mul_nonneg (by norm_num) (Finset.sum_nonneg fun y _ => abs_nonneg _)

/-- **The one-step mismatch is at most the total variation.** For distributions `p, q` on `Y`
and `0 ≤ V ≤ 1`, `∑_y (q y - p y) V y ≤ TV(p, q)`: centering `V` at `1/2` costs nothing because
both rows sum to one, and `|V - 1/2| ≤ 1/2`. -/
lemma sum_sub_mul_le_tv (p q : Y → ℝ) (hp : ∑ y, p y = 1) (hq : ∑ y, q y = 1) (V : Y → ℝ)
    (hV : ∀ y, 0 ≤ V y ∧ V y ≤ 1) : ∑ y, (q y - p y) * V y ≤ tv p q := by
  have h0 : ∑ y, (q y - p y) = 0 := by rw [Finset.sum_sub_distrib, hp, hq, sub_self]
  have hcenter : ∑ y, (q y - p y) * V y = ∑ y, (q y - p y) * (V y - 1 / 2) := by
    have : ∑ y, (q y - p y) * (V y - 1 / 2)
        = ∑ y, (q y - p y) * V y - 1 / 2 * ∑ y, (q y - p y) := by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun y _ => by ring
    rw [this, h0, mul_zero, sub_zero]
  rw [hcenter, tv, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro y _
  have hb : |V y - 1 / 2| ≤ 1 / 2 :=
    abs_le.2 ⟨by linarith [(hV y).1], by linarith [(hV y).2]⟩
  calc (q y - p y) * (V y - 1 / 2) ≤ |(q y - p y) * (V y - 1 / 2)| := le_abs_self _
    _ = |q y - p y| * |V y - 1 / 2| := abs_mul _ _
    _ ≤ |q y - p y| * (1 / 2) := mul_le_mul_of_nonneg_left hb (abs_nonneg _)
    _ = 1 / 2 * |p y - q y| := by rw [abs_sub_comm]; ring

/-- **Lemma `imit` for one task, from any history.** With `k` steps left, the success gap of `π̃`
over `π` is at most the expected total variation, under `π̃`'s own histories, summed over the
remaining steps. -/
theorem value_gap_le (π π' : List Y → Y → ℝ) (hπ : IsPolicy π) (hπ' : IsPolicy π')
    (R : List Y → ℝ) (hR : ∀ h, 0 ≤ R h ∧ R h ≤ 1) :
    ∀ k h, expectAt π' k R h - expectAt π k R h
      ≤ ∑ t ∈ range k, expectAt π' t (fun g => tv (π g) (π' g)) h := by
  intro k
  induction k with
  | zero => intro h; simp [expectAt]
  | succ k ih =>
    intro h
    -- paper step: split the gap at `h` into the gap one response later plus the mismatch at `h`
    have hsplit : expectAt π' (k + 1) R h - expectAt π (k + 1) R h
        = ∑ y, π' h y * (expectAt π' k R (h ++ [y]) - expectAt π k R (h ++ [y]))
          + ∑ y, (π' h y - π h y) * expectAt π k R (h ++ [y]) := by
      rw [expectAt_succ, expectAt_succ, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun y _ => by ring
    -- the later gap is bounded by the induction hypothesis, averaged under `π̃`
    have hlater : ∑ y, π' h y * (expectAt π' k R (h ++ [y]) - expectAt π k R (h ++ [y]))
        ≤ ∑ y, π' h y * ∑ t ∈ range k, expectAt π' t (fun g => tv (π g) (π' g)) (h ++ [y]) :=
      Finset.sum_le_sum fun y _ => mul_le_mul_of_nonneg_left (ih _) ((hπ' h).1 y)
    -- the mismatch at `h` is at most the total variation there, since `0 ≤ V_π ≤ 1`
    have hnow : ∑ y, (π' h y - π h y) * expectAt π k R (h ++ [y]) ≤ tv (π h) (π' h) :=
      sum_sub_mul_le_tv (π h) (π' h) (hπ h).2 (hπ' h).2 _
        (fun y => expectAt_mem_unit π hπ R hR k _)
    -- unrolling: the step-`0` term is the TV at `h`, later terms are averages under `π̃`
    have hunroll : ∑ t ∈ range (k + 1), expectAt π' t (fun g => tv (π g) (π' g)) h
        = ∑ y, π' h y * ∑ t ∈ range k, expectAt π' t (fun g => tv (π g) (π' g)) (h ++ [y])
          + tv (π h) (π' h) := by
      rw [Finset.sum_range_succ']
      have hsucc : ∑ t ∈ range k, expectAt π' (t + 1) (fun g => tv (π g) (π' g)) h
          = ∑ y, π' h y * ∑ t ∈ range k, expectAt π' t (fun g => tv (π g) (π' g)) (h ++ [y]) := by
        simp only [expectAt_succ, Finset.mul_sum]
        exact Finset.sum_comm
      rw [hsucc]
      rfl
    rw [hsplit, hunroll]
    linarith

/-- **Lemma `imit`** (`Statements.lem_imit`): average `value_gap_le` over the task distribution. -/
theorem lem_imit : Statements.lem_imit := by
  intro Ω Y _ _ pop π π' R T hπ hπ' hR
  have hgap := fun ω => value_gap_le (π ω) (π' ω) (hπ ω) (hπ' ω) (R ω) (hR ω) T []
  calc (∑ ω, pop.P ω * expectAt (π' ω) T (R ω) []) - ∑ ω, pop.P ω * expectAt (π ω) T (R ω) []
      = ∑ ω, pop.P ω * (expectAt (π' ω) T (R ω) [] - expectAt (π ω) T (R ω) []) := by
        rw [← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun ω _ => by ring
    _ ≤ ∑ ω, pop.P ω * ∑ t ∈ range T, expectAt (π' ω) t (fun h => tv (π ω h) (π' ω h)) [] :=
        Finset.sum_le_sum fun ω _ => mul_le_mul_of_nonneg_left (hgap ω) (pop.nonneg ω)
    _ = ∑ t ∈ range T, ∑ ω, pop.P ω * expectAt (π' ω) t (fun h => tv (π ω h) (π' ω h)) [] := by
        simp only [Finset.mul_sum]
        exact Finset.sum_comm

end Imit

section Relabel

variable {Y : Type} [Fintype Y] [DecidableEq Y]

lemma pointMass_isDist (y₀ : Y) : (∀ y, 0 ≤ pointMass y₀ y) ∧ ∑ y, pointMass y₀ y = 1 := by
  refine ⟨fun y => ?_, ?_⟩
  · unfold pointMass
    split_ifs <;> norm_num
  · simp [pointMass]

/-- Integrating against a point mass evaluates at its atom. -/
lemma sum_pointMass_mul (y₀ : Y) (g : Y → ℝ) : ∑ y, pointMass y₀ y * g y = g y₀ := by
  simp [pointMass]

omit [DecidableEq Y] in
/-- **A dead prefix has value `0`.** If every completion of `h` by `k` responses has reward `0`,
every policy succeeds from `h` with probability `0`. -/
lemma expectAt_eq_zero_of_dead (π : List Y → Y → ℝ) (f : List Y → ℝ) :
    ∀ k (h : List Y), (∀ c : List Y, c.length = k → f (h ++ c) = 0) → expectAt π k f h = 0 := by
  intro k
  induction k with
  | zero =>
    intro h hdead
    simpa [expectAt] using hdead [] rfl
  | succ k ih =>
    intro h hdead
    rw [expectAt_succ]
    refine Finset.sum_eq_zero fun y _ => ?_
    have hy : expectAt π k f (h ++ [y]) = 0 :=
      ih (h ++ [y]) fun c hc => by
        rw [List.append_assoc, List.singleton_append]
        exact hdead (y :: c) (by simp [hc])
    rw [hy, mul_zero]

omit [DecidableEq Y] in
/-- Two policies that agree on every history of length at least `m` have the same expectations
from such histories. -/
lemma expectAt_congr_of_length (π π' : List Y → Y → ℝ) (m : ℕ)
    (hagree : ∀ h : List Y, m ≤ h.length → π h = π' h) (f : List Y → ℝ) :
    ∀ k (h : List Y), m ≤ h.length → expectAt π k f h = expectAt π' k f h := by
  intro k
  induction k with
  | zero => intro h _; rfl
  | succ k ih =>
    intro h hh
    rw [expectAt_succ, expectAt_succ, hagree h hh]
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [ih (h ++ [y]) (by simp only [List.length_append, List.length_singleton]; omega)]

variable [Inhabited Y]

/-- **Replaying a logged prefix.** If `π` emits the logged response at each of the first `m`
positions of `l`, then `d ≤ m` replayed steps from `l.take (m - d)` lead to `l.take m`. -/
lemma expectAt_replay (π : List Y → Y → ℝ) (f : List Y → ℝ) (l : List Y) (m k : ℕ)
    (hm : m ≤ l.length) (hrep : ∀ j < m, π (l.take j) = pointMass (l.getD j default)) :
    ∀ d, d ≤ m → expectAt π (k + d) f (l.take (m - d)) = expectAt π k f (l.take m) := by
  intro d
  induction d with
  | zero => intro _; rfl
  | succ d ih =>
    intro hd
    have hj : m - (d + 1) < m := by omega
    have hjl : m - (d + 1) < l.length := by omega
    rw [show k + (d + 1) = (k + d) + 1 by omega, expectAt_succ, hrep _ hj, sum_pointMass_mul,
      List.getD_eq_getElem _ _ hjl, List.take_append_getElem hjl,
      show m - (d + 1) + 1 = m - d by omega]
    exact ih (by omega)

omit [Fintype Y] in
/-- The relabeled process replays: in the replay branch it is the point mass on the logged
response at the current position. -/
lemma relabel_of_replay {Ω : Type} (solved : Ω → Prop) [DecidablePred solved]
    (logged : Ω → List Y) (tl : Ω → ℕ) (πT : Ω → List Y → Y → ℝ) (ω : Ω) (h : List Y)
    (hc : solved ω ∨ h.length + 1 < tl ω) :
    relabel solved logged tl πT ω h = pointMass ((logged ω).getD h.length default) := by
  funext y
  simp only [relabel, hc, ↓reduceIte]

omit [Fintype Y] in
/-- Outside the replay branch the relabeled process is the teacher. -/
lemma relabel_of_teacher {Ω : Type} (solved : Ω → Prop) [DecidablePred solved]
    (logged : Ω → List Y) (tl : Ω → ℕ) (πT : Ω → List Y → Y → ℝ) (ω : Ω) (h : List Y)
    (hc : ¬ (solved ω ∨ h.length + 1 < tl ω)) :
    relabel solved logged tl πT ω h = πT ω h := by
  funext y
  simp only [relabel, hc, ↓reduceIte]

/-- The relabeled process is a policy whenever the teacher is. -/
lemma relabel_isPolicy {Ω : Type} (solved : Ω → Prop) [DecidablePred solved]
    (logged : Ω → List Y) (tl : Ω → ℕ) (πT : Ω → List Y → Y → ℝ)
    (hπT : ∀ ω, IsPolicy (πT ω)) (ω : Ω) : IsPolicy (relabel solved logged tl πT ω) := by
  intro h
  by_cases hc : solved ω ∨ h.length + 1 < tl ω
  · rw [relabel_of_replay solved logged tl πT ω h hc]
    exact pointMass_isDist _
  · rw [relabel_of_teacher solved logged tl πT ω h hc]
    exact hπT ω h

/-- **The relabeled process on a solved task** replays the logged rollout and succeeds with
probability `R(logged)`. -/
lemma relabel_value_solved {Ω : Type} (solved : Ω → Prop) [DecidablePred solved]
    (logged : Ω → List Y) (tl : Ω → ℕ) (πT : Ω → List Y → Y → ℝ) (R : List Y → ℝ) (T : ℕ)
    (ω : Ω) (hs : solved ω) (hlen : (logged ω).length = T) :
    expectAt (relabel solved logged tl πT ω) T R [] = R (logged ω) := by
  have hrep : ∀ j < T, relabel solved logged tl πT ω ((logged ω).take j)
      = pointMass ((logged ω).getD j default) := by
    intro j hj
    rw [relabel_of_replay solved logged tl πT ω _ (Or.inl hs), List.length_take,
      min_eq_left (by omega)]
  have h1 := expectAt_replay (relabel solved logged tl πT ω) R (logged ω) T 0 (by omega) hrep
    T le_rfl
  rw [zero_add, Nat.sub_self, List.take_zero, List.take_of_length_le (le_of_eq hlen)] at h1
  rw [h1]
  rfl

/-- **The relabeled process on a failed task** replays the logged prefix through step `t' - 1`
and then follows the teacher, so it succeeds with the teacher's success probability from
`h_{t'}`. -/
lemma relabel_value_failed {Ω : Type} (solved : Ω → Prop) [DecidablePred solved]
    (logged : Ω → List Y) (tl : Ω → ℕ) (πT : Ω → List Y → Y → ℝ) (R : List Y → ℝ) (T : ℕ)
    (ω : Ω) (hs : ¬ solved ω) (hlen : (logged ω).length = T) (htl : tl ω ≤ T) :
    expectAt (relabel solved logged tl πT ω) T R []
      = expectAt (πT ω) (T - (tl ω - 1)) R ((logged ω).take (tl ω - 1)) := by
  set m := tl ω - 1 with hm
  have hrep : ∀ j < m, relabel solved logged tl πT ω ((logged ω).take j)
      = pointMass ((logged ω).getD j default) := by
    intro j hj
    have hlenj : ((logged ω).take j).length = j := by rw [List.length_take]; omega
    rw [relabel_of_replay solved logged tl πT ω _ (Or.inr (by rw [hlenj]; omega)), hlenj]
  have h1 := expectAt_replay (relabel solved logged tl πT ω) R (logged ω) m (T - m)
    (by omega) hrep m le_rfl
  rw [Nat.sub_self, List.take_zero, show T - m + m = T by omega] at h1
  rw [h1]
  -- after the replayed prefix the relabeled process is the teacher
  refine expectAt_congr_of_length _ _ m (fun h hh => ?_) R (T - m) _
    (by rw [List.length_take]; omega)
  exact relabel_of_teacher solved logged tl πT ω h (by rw [not_or, not_lt]; exact ⟨hs, by omega⟩)

/-! ### From Proposition `tstar` to a dead prefix -/

omit [Fintype Y] [DecidableEq Y] [Inhabited Y] in
/-- **Harness bridge.** Responses are parsed into actions by `parse`, and a response sequence is
rewarded by the harness reward of the state its parsed actions reach. If the state reached by a
response prefix is unrecoverable within the remaining budget `k`, no completion of the prefix by
`k` responses reaches a rewarded state. -/
theorem dead_prefix_of_not_recoverable {S A : Type} (H : Harness S A) (s₀ : S) (parse : Y → A)
    (pre : List Y) (k : ℕ) (hnr : ¬ H.Recoverable (H.run s₀ (pre.map parse)) k) :
    ∀ c : List Y, c.length = k → ¬ H.R (H.run s₀ ((pre ++ c).map parse)) := by
  intro c hc hR
  apply hnr
  refine ⟨c.map parse, by simp [hc], Or.inr (by simp [hc]), ?_⟩
  rw [List.map_append, Harness.run_append] at hR
  exact hR

omit [Fintype Y] [DecidableEq Y] in
/-- The state after `j` executed actions is the run of the first `j` parsed logged responses. -/
lemma state_eq_run_take {S A : Type} (H : Harness S A) (τ : Trajectory S A) (parse : Y → A)
    (logged : List Y) (hact : ∀ t, t < logged.length → τ.act t = parse (logged.getD t default)) :
    ∀ j, j ≤ logged.length → τ.state H j = H.run τ.s0 ((logged.take j).map parse) := by
  intro j
  induction j with
  | zero => intro _; rfl
  | succ j ih =>
    intro hj
    have hjl : j < logged.length := by omega
    rw [← List.take_append_getElem hjl, List.map_append, Harness.run_append, ← ih hjl.le]
    simp only [Trajectory.state, List.map_cons, List.map_nil, Harness.run_cons, Harness.run_nil]
    rw [hact j hjl, List.getD_eq_getElem _ _ hjl]

omit [Fintype Y] [DecidableEq Y] in
/-- **Late localization leaves a dead prefix** (the step of the proof of Theorem `bound-formal`
that uses Proposition `tstar`). On a failed trajectory with `ν_0 = 1` that executes the parsed
logged responses, if the localization `t'` comes after the first fatal step `t⋆`, no completion of
the replayed prefix `y_{1:t'-1}` by the remaining `T - (t' - 1)` responses reaches a rewarded
state. -/
theorem late_localization_dead {S A : Type} (H : Harness S A) (τ : Trajectory S A) (T : ℕ)
    (parse : Y → A) (logged : List Y) (hlen : logged.length = T)
    (hact : ∀ t, t < logged.length → τ.act t = parse (logged.getD t default))
    (hf : τ.Failed H T) (h0 : τ.nu H T 0) (t' : ℕ) (ht' : t' - 1 ≤ τ.n)
    (hlate : ∃ tstar, ¬ τ.nu H T tstar ∧ (∀ t, t < tstar → τ.nu H T t) ∧ tstar < t') :
    ∀ c : List Y, c.length = T - (t' - 1) →
      ¬ H.R (H.run τ.s0 ((logged.take (t' - 1) ++ c).map parse)) := by
  obtain ⟨tstar, hspec, hmin, hlt⟩ := hlate
  obtain ⟨_, _, hafter, _, _⟩ := τ.first_fatal_step_of_min H T tstar hf h0 hspec hmin
  -- Proposition `tstar`: the state after `t' - 1 ≥ t⋆` actions is unrecoverable
  have hnu : ¬ τ.nu H T (t' - 1) := τ.late_localization_lost H T tstar t' hafter hlt ht'
  have hstate : τ.state H (t' - 1) = H.run τ.s0 ((logged.take (t' - 1)).map parse) :=
    state_eq_run_take H τ parse logged hact (t' - 1) (by have := hf.le_budget; omega)
  unfold Trajectory.nu at hnu
  rw [hstate] at hnu
  exact dead_prefix_of_not_recoverable H τ.s0 parse _ _ hnu

omit [Fintype Y] [DecidableEq Y] in
/-- **The hypothesis of Theorem `bound-formal` (2) from the harness.** Reward a complete response
sequence by `1` if its parsed actions reach a rewarded harness state and `0` otherwise. After a late
localization every completion of the replayed prefix has reward `0`. -/
theorem late_localization_dead_reward {S A : Type} (H : Harness S A) [DecidablePred H.R]
    (τ : Trajectory S A) (T : ℕ) (parse : Y → A) (logged : List Y) (hlen : logged.length = T)
    (hact : ∀ t, t < logged.length → τ.act t = parse (logged.getD t default))
    (hf : τ.Failed H T) (h0 : τ.nu H T 0) (t' : ℕ) (ht' : t' - 1 ≤ τ.n)
    (hlate : ∃ tstar, ¬ τ.nu H T tstar ∧ (∀ t, t < tstar → τ.nu H T t) ∧ tstar < t') :
    ∀ c : List Y, c.length = T - (t' - 1) →
      (if H.R (H.run τ.s0 ((logged.take (t' - 1) ++ c).map parse)) then (1 : ℝ) else 0) = 0 := by
  intro c hc
  have hdead := late_localization_dead H τ T parse logged hlen hact hf h0 t' ht' hlate c hc
  simp only [hdead, ↓reduceIte]

end Relabel

end WebKD
