import WebkdLean.Statements

set_option autoImplicit false
set_option relaxedAutoImplicit false

/-!
# The first fatal step (Proofs of Lemma `mono` and Proposition `tstar`)

Strategy, following Appendix `tstar` of the paper. Lemma `mono` prepends the executed action to a
successful continuation from the next state (`nu_of_nu_succ`), so recoverability at step `t + 1`
gives recoverability at step `t`. At the episode's end `ν = 0` (`not_nu_end`): an ended episode is
absorbing with reward `0`, and an exhausted budget leaves only the empty continuation. Hence some
index has `ν = 0`, the least one is `t⋆` (`Nat.find`), it is at least `1` because `ν_0 = 1`, and
Lemma `mono` propagates `ν = 0` from `t⋆` to the end (`first_fatal_step_of_min`). The first action
of a successful continuation from `s_(t⋆-1)` is the recoverability-preserving action at step `t⋆`;
that continuation is not empty, since an ended state at `t⋆ - 1` would be absorbing and give
`ν_{t⋆} = 1`, and the budget cannot run out before `t⋆ ≤ n ≤ T`.
The two localization facts used in the bound (late is lost, early is recoverable) are immediate
corollaries.
-/

namespace WebKD

variable {S A : Type}

@[simp] lemma Harness.run_nil (H : Harness S A) (s : S) : H.run s [] = s := rfl

@[simp] lemma Harness.run_cons (H : Harness S A) (s : S) (a : A) (as : List A) :
    H.run s (a :: as) = H.run (H.step s a) as := rfl

/-- Running a concatenation runs the two pieces in turn. -/
lemma Harness.run_append (H : Harness S A) (s : S) (as bs : List A) :
    H.run s (as ++ bs) = H.run (H.run s as) bs := by
  induction as generalizing s with
  | nil => rfl
  | cons a as ih => exact ih (H.step s a)

/-- An ended episode stays where it is (Assumption `budget`). -/
lemma Harness.run_of_done (H : Harness S A) {s : S} (hs : H.done s) :
    ∀ seq : List A, H.run s seq = s
  | [] => rfl
  | a :: as => by rw [Harness.run_cons, H.absorbing s a hs, H.run_of_done hs as]

/-- **Lemma `mono`**, forward form: `ν_{t+1} = 1 → ν_t = 1` along the trajectory (`t + 1 ≤ T`).
Paper step: prepend the executed action `a_{t+1}` to the successful continuation from `s_(t+1)`. -/
lemma Trajectory.nu_of_nu_succ (τ : Trajectory S A) (H : Harness S A) (T t : ℕ)
    (ht : t + 1 ≤ T) (h : τ.nu H T (t + 1)) : τ.nu H T t := by
  obtain ⟨seq, hlen, hend, hR⟩ := h
  refine ⟨τ.act t :: seq, ?_, ?_, ?_⟩
  · -- the longer sequence fits in the larger budget `T - t`
    simp only [List.length_cons]; omega
  · rcases hend with hd | hl
    · exact Or.inl hd
    · -- a continuation that used up budget `T - (t+1)` now uses up `T - t`
      right; simp only [List.length_cons]; omega
  · exact hR

/-- Lemma `mono` in the contrapositive form of the paper. -/
lemma Trajectory.not_nu_succ_of_not_nu (τ : Trajectory S A) (H : Harness S A) (T t : ℕ)
    (ht : t + 1 ≤ T) (h : ¬ τ.nu H T t) : ¬ τ.nu H T (t + 1) :=
  fun h' => h (τ.nu_of_nu_succ H T t ht h')

/-- At the episode's end, `ν = 0`: an ended episode is absorbing with reward `0`, and an exhausted
budget admits only the empty continuation, whose reward is the episode's own reward `0`. -/
lemma Trajectory.not_nu_end (τ : Trajectory S A) (H : Harness S A) (T : ℕ)
    (hf : τ.Failed H T) : ¬ τ.nu H T τ.n := by
  rintro ⟨seq, hlen, _, hR⟩
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
  -- `t⋆ ≥ 1` because `ν_0 = 1`
  have h1 : 1 ≤ tstar := by
    by_contra h
    have h' : tstar = 0 := by omega
    rw [h'] at hspec
    exact hspec h0
  -- `t⋆ ≤ n` because `ν = 0` at the end and `t⋆` is the least such index
  have hle : tstar ≤ τ.n := by
    by_contra h
    exact τ.not_nu_end H T hf (hmin _ (by omega))
  -- Lemma `mono` propagates `ν = 0` from `t⋆` to the end
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
  · -- the first action of a successful continuation from `s_(t⋆-1)` preserves recoverability
    have hnum : τ.nu H T m := hmin m (by omega)
    obtain ⟨seq, hlen, hend, hR⟩ := hnum
    cases seq with
    | nil =>
      exfalso
      simp only [Harness.run_nil, List.length_nil] at hend hR
      rcases hend with hdone | hzero
      · -- an ended episode is absorbing, so `ν_{m+1} = ν_m = 1`, contradicting `hspec`
        apply hspec
        refine ⟨[], by simp, Or.inl ?_, ?_⟩
        · show H.done (H.step (τ.state H m) (τ.act m))
          rw [H.absorbing _ _ hdone]
          exact hdone
        · show H.R (H.step (τ.state H m) (τ.act m))
          rw [H.absorbing _ _ hdone]
          exact hR
      · -- the budget cannot be exhausted at `m < t⋆ ≤ n ≤ T`
        have := hf.le_budget
        omega
    | cons a rest =>
      refine ⟨a, rest, ?_, ?_, ?_⟩
      · simp only [List.length_cons] at hlen
        have := hf.le_budget
        omega
      · rw [Nat.add_sub_cancel]
        rcases hend with hd | hl
        · exact Or.inl hd
        · right; simp only [List.length_cons] at hl; omega
      · rw [Nat.add_sub_cancel]
        exact hR
  · intro hrec
    apply hspec
    rw [Nat.add_sub_cancel] at hrec
    exact hrec

/-- **Proposition `tstar`** (existence): the first fatal step `t⋆ = min {t : ν_t = 0}` exists,
satisfies `1 ≤ t⋆ ≤ n`, every step before it leaves the task solvable, every step from it on leaves
the task unsolvable, and at step `t⋆` a recoverability-preserving action existed but the executed
action was not one. -/
theorem Trajectory.first_fatal_step (τ : Trajectory S A) (H : Harness S A) (T : ℕ)
    (hf : τ.Failed H T) (h0 : τ.nu H T 0) :
    ∃ tstar : ℕ, 1 ≤ tstar ∧ tstar ≤ τ.n ∧ ¬ τ.nu H T tstar ∧
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
  exact ⟨Nat.find hex, h1, hle, hspec, hmin, hafter, hexists, hnot⟩

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

/-! ### The statements -/

/-- **Lemma `mono`** (`Statements.lem_mono`). -/
theorem lem_mono : Statements.lem_mono := by
  intro S A H τ T t ht h
  exact τ.not_nu_succ_of_not_nu H T t ht h

/-- **Proposition `tstar`** (`Statements.prop_tstar`): existence with all its properties, from
`first_fatal_step`, and uniqueness, from `first_fatal_step_unique`. -/
theorem prop_tstar : Statements.prop_tstar := by
  intro S A H τ T hf h0
  refine ⟨τ.not_nu_end H T hf, ?_⟩
  obtain ⟨tstar, h1, hle, hspec, hmin, hafter, hex, hnot⟩ := τ.first_fatal_step H T hf h0
  exact ⟨tstar, h1, hle, hspec, hmin, hafter, hex, hnot,
    fun t' ht' hmin' => τ.first_fatal_step_unique H T ht' hmin' hspec hmin⟩

end WebKD
