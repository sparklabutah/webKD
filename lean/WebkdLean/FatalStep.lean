import Mathlib

/-!
# The first fatal step

Formalizes Definition `recover`, Lemma `mono` (recoverability is absorbing) and
Proposition `tstar` (every failed web trajectory has a unique first fatal step) of the WebKD
paper, together with the two localization facts used in P3 and in the proof of the
single-round bound (late localization is lost, early localization is still recoverable).

The harness is modelled by a deterministic transition `step` (Assumption `det`: the transition
and the parser are functions), a predicate `done` marking that the episode has ended (a terminal
action was issued), an absorbing law for ended episodes (Assumption `budget`), and a terminal
reward predicate `R` read off the final state (Assumption `reward`).
-/

namespace WebKD

/-- A deterministic web harness. -/
structure Harness (S A : Type*) where
  /-- deterministic transition `𝕋` composed with the deterministic parser `f_parse` -/
  step : S → A → S
  /-- the episode has ended (a terminal action was issued) -/
  done : S → Prop
  /-- the terminal binary reward, read off the final state -/
  R : S → Prop
  /-- an ended episode is absorbing, with its reward fixed (Assumption `budget`) -/
  absorbing : ∀ s a, done s → step s a = s

variable {S A : Type*}

/-- Run an action sequence from a state. -/
def Harness.run (H : Harness S A) (s : S) : List A → S
  | [] => s
  | a :: as => H.run (H.step s a) as

@[simp] lemma Harness.run_nil (H : Harness S A) (s : S) : H.run s [] = s := rfl

@[simp] lemma Harness.run_cons (H : Harness S A) (s : S) (a : A) (as : List A) :
    H.run s (a :: as) = H.run (H.step s a) as := rfl

lemma Harness.run_of_done (H : Harness S A) {s : S} (hs : H.done s) :
    ∀ seq : List A, H.run s seq = s
  | [] => rfl
  | a :: as => by rw [Harness.run_cons, H.absorbing s a hs, H.run_of_done hs as]

/-- **Definition `recover`.** From state `s` with remaining step budget `k`, some action sequence
of length at most `k` ends the episode with reward `1`. -/
def Harness.Recoverable (H : Harness S A) (s : S) (k : ℕ) : Prop :=
  ∃ seq : List A, seq.length ≤ k ∧ H.done (H.run s seq) ∧ H.R (H.run s seq)

/-- A logged trajectory: the initial state, the executed actions (`act t` is the action `a_{t+1}`
taken at step `t+1`), and the number `n` of executed steps. -/
structure Trajectory (S A : Type*) where
  s0 : S
  act : ℕ → A
  n : ℕ

/-- `s_(t)`: the underlying browser state after the first `t` actions of `τ`. -/
def Trajectory.state (τ : Trajectory S A) (H : Harness S A) : ℕ → S
  | 0 => τ.s0
  | t + 1 => H.step (τ.state H t) (τ.act t)

/-- `ν_t` (Definition `recover`) for step budget `T`. -/
def Trajectory.nu (τ : Trajectory S A) (H : Harness S A) (T t : ℕ) : Prop :=
  H.Recoverable (τ.state H t) (T - t)

/-- **Lemma `mono`** (recoverability is absorbing): `ν_{t+1} = 1 → ν_t = 1`, equivalently
`ν_t = 0 → ν_{t+1} = 0`, along the trajectory (`t + 1 ≤ T`). -/
lemma Trajectory.nu_of_nu_succ (τ : Trajectory S A) (H : Harness S A) (T t : ℕ)
    (ht : t + 1 ≤ T) (h : τ.nu H T (t + 1)) : τ.nu H T t := by
  obtain ⟨seq, hlen, hdone, hR⟩ := h
  refine ⟨τ.act t :: seq, ?_, ?_, ?_⟩
  · simp only [List.length_cons]; omega
  · exact hdone
  · exact hR

/-- Lemma `mono` in the contrapositive form of the paper. -/
lemma Trajectory.not_nu_succ_of_not_nu (τ : Trajectory S A) (H : Harness S A) (T t : ℕ)
    (ht : t + 1 ≤ T) (h : ¬ τ.nu H T t) : ¬ τ.nu H T (t + 1) :=
  fun h' => h (τ.nu_of_nu_succ H T t ht h')

/-- A failed episode under budget `T`: at most `T` steps were executed, the episode ended
(a terminal action was issued, or the budget was exhausted), and the terminal reward is `0`. -/
structure Trajectory.Failed (τ : Trajectory S A) (H : Harness S A) (T : ℕ) : Prop where
  le_budget : τ.n ≤ T
  ended : H.done (τ.state H τ.n) ∨ τ.n = T
  no_reward : ¬ H.R (τ.state H τ.n)

/-- At the episode's end, `ν = 0`. -/
lemma Trajectory.not_nu_end (τ : Trajectory S A) (H : Harness S A) (T : ℕ)
    (hf : τ.Failed H T) : ¬ τ.nu H T τ.n := by
  rintro ⟨seq, hlen, hdone, hR⟩
  rcases hf.ended with hd | hn
  · rw [H.run_of_done hd] at hR
    exact hf.no_reward hR
  · have h0 : seq.length = 0 := by omega
    rw [List.eq_nil_of_length_eq_zero h0] at hR
    exact hf.no_reward hR

/-- The content of Proposition `tstar` for any `t⋆` that is the least index with `ν = 0`. -/
theorem Trajectory.first_fatal_step_of_min (τ : Trajectory S A) (H : Harness S A) (T tstar : ℕ)
    (hf : τ.Failed H T) (h0 : τ.nu H T 0)
    (hspec : ¬ τ.nu H T tstar) (hmin : ∀ t, t < tstar → τ.nu H T t) :
    1 ≤ tstar ∧ tstar ≤ τ.n ∧
      (∀ t, tstar ≤ t → t ≤ τ.n → ¬ τ.nu H T t) ∧
      (∃ a : A, H.Recoverable (H.step (τ.state H (tstar - 1)) a) (T - tstar)) ∧
      ¬ H.Recoverable (H.step (τ.state H (tstar - 1)) (τ.act (tstar - 1))) (T - tstar) := by
  have h1 : 1 ≤ tstar := by
    by_contra h
    have h' : tstar = 0 := by omega
    rw [h'] at hspec
    exact hspec h0
  have hle : tstar ≤ τ.n := by
    by_contra h
    exact τ.not_nu_end H T hf (hmin _ (by omega))
  have hafter : ∀ k, tstar + k ≤ τ.n → ¬ τ.nu H T (tstar + k) := by
    intro k
    induction k with
    | zero => intro _; simpa using hspec
    | succ k ih =>
      intro hk hnu
      have := τ.nu_of_nu_succ H T (tstar + k) (by have := hf.le_budget; omega) hnu
      exact ih (by omega) this
  obtain ⟨m, rfl⟩ : ∃ m, tstar = m + 1 := ⟨tstar - 1, by omega⟩
  refine ⟨h1, hle, ?_, ?_, ?_⟩
  · intro t ht htn
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le ht
    exact hafter k htn
  · have hnum : τ.nu H T m := hmin m (by omega)
    obtain ⟨seq, hlen, hdone, hR⟩ := hnum
    cases seq with
    | nil =>
      exfalso
      apply hspec
      simp only [Harness.run_nil] at hdone hR
      refine ⟨[], by simp, ?_, ?_⟩
      · show H.done (H.step (τ.state H m) (τ.act m))
        rw [H.absorbing _ _ hdone]
        exact hdone
      · show H.R (H.step (τ.state H m) (τ.act m))
        rw [H.absorbing _ _ hdone]
        exact hR
    | cons a rest =>
      refine ⟨a, rest, ?_, ?_, ?_⟩
      · simp only [List.length_cons] at hlen
        have := hf.le_budget
        omega
      · rw [Nat.add_sub_cancel]
        exact hdone
      · rw [Nat.add_sub_cancel]
        exact hR
  · intro hrec
    apply hspec
    rw [Nat.add_sub_cancel] at hrec
    exact hrec

/-- **Proposition `tstar`.** Along a failed trajectory of a task that is solvable from its initial
state (`ν_0 = 1`), the first fatal step `t⋆ = min {t ≥ 1 : ν_t = 0}` exists, satisfies
`1 ≤ t⋆ ≤ n`, every step before it leaves the task solvable, every step from it on leaves the task
unsolvable, and at step `t⋆` a recoverability-preserving action existed but the executed action
was not one. -/
theorem Trajectory.first_fatal_step (τ : Trajectory S A) (H : Harness S A) (T : ℕ)
    (hf : τ.Failed H T) (h0 : τ.nu H T 0) :
    ∃ tstar : ℕ, 1 ≤ tstar ∧ tstar ≤ τ.n ∧
      (∀ t, t < tstar → τ.nu H T t) ∧
      (∀ t, tstar ≤ t → t ≤ τ.n → ¬ τ.nu H T t) ∧
      (∃ a : A, H.Recoverable (H.step (τ.state H (tstar - 1)) a) (T - tstar)) ∧
      ¬ H.Recoverable (H.step (τ.state H (tstar - 1)) (τ.act (tstar - 1))) (T - tstar) := by
  classical
  have hex : ∃ t, ¬ τ.nu H T t := ⟨τ.n, τ.not_nu_end H T hf⟩
  have hspec : ¬ τ.nu H T (Nat.find hex) := Nat.find_spec hex
  have hmin : ∀ t, t < Nat.find hex → τ.nu H T t := fun t ht => by
    have := Nat.find_min hex ht
    tauto
  obtain ⟨h1, hle, hafter, hexists, hnot⟩ :=
    τ.first_fatal_step_of_min H T (Nat.find hex) hf h0 hspec hmin
  exact ⟨Nat.find hex, h1, hle, hmin, hafter, hexists, hnot⟩

/-- The characterization in Proposition `tstar` determines `t⋆` uniquely. -/
theorem Trajectory.first_fatal_step_unique (τ : Trajectory S A) (H : Harness S A) (T : ℕ)
    {t1 t2 : ℕ}
    (h1 : ¬ τ.nu H T t1) (h1' : ∀ t, t < t1 → τ.nu H T t)
    (h2 : ¬ τ.nu H T t2) (h2' : ∀ t, t < t2 → τ.nu H T t) : t1 = t2 := by
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with h | h
  · exact h1 (h2' t1 h)
  · exact h2 (h1' t2 h)

/-- **P3 asymmetry, late case.** If the localizer returns `t' > t⋆`, the state `h_{t'}` (reached
after `t' - 1` actions) is unrecoverable: no continuation within the budget can succeed. -/
theorem Trajectory.late_localization_lost (τ : Trajectory S A) (H : Harness S A) (T tstar t' : ℕ)
    (hafter : ∀ t, tstar ≤ t → t ≤ τ.n → ¬ τ.nu H T t)
    (hlate : tstar < t') (ht' : t' - 1 ≤ τ.n) : ¬ τ.nu H T (t' - 1) :=
  hafter _ (by omega) ht'

/-- **P3 asymmetry, early case.** If the localizer returns `1 ≤ t' ≤ t⋆`, the state `h_{t'}` is
still recoverable, so the teacher only spends extra edits. -/
theorem Trajectory.early_localization_recoverable (τ : Trajectory S A) (H : Harness S A)
    (T tstar t' : ℕ) (hbefore : ∀ t, t < tstar → τ.nu H T t)
    (hearly : t' ≤ tstar) (h1 : 1 ≤ t') : τ.nu H T (t' - 1) :=
  hbefore _ (by omega)

end WebKD
