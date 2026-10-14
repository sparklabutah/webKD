import WebkdLean

/-! Axiom audit: every declaration in the paper ↔ Lean map of `README.md`. Each line must report
only `propext`, `Classical.choice` and `Quot.sound` (or no axioms). -/

-- the sixteen statements
#print axioms WebKD.lem_mono
#print axioms WebKD.prop_tstar
#print axioms WebKD.prop_target
#print axioms WebKD.eq_kl_grad
#print axioms WebKD.cor_beta
#print axioms WebKD.cor_budget
#print axioms WebKD.eq_sw
#print axioms WebKD.rem_resample
#print axioms WebKD.eq_actw_map
#print axioms WebKD.lem_sandwich
#print axioms WebKD.prop_split
#print axioms WebKD.thm_rank
#print axioms WebKD.lem_imit
#print axioms WebKD.thm_bound_formal
#print axioms WebKD.cor_regimes
#print axioms WebKD.thm_bound
-- Appendix `tstar`
#print axioms WebKD.Trajectory.nu_of_nu_succ
#print axioms WebKD.Trajectory.not_nu_end
#print axioms WebKD.Trajectory.first_fatal_step
#print axioms WebKD.Trajectory.first_fatal_step_of_min
#print axioms WebKD.Trajectory.first_fatal_step_unique
#print axioms WebKD.Trajectory.late_localization_lost
#print axioms WebKD.Trajectory.early_localization_recoverable
-- the anchor
#print axioms WebKD.hasDerivAt_anchorLoss
#print axioms WebKD.hasDerivAt_anchored_coord
#print axioms WebKD.offtarget_gap_rate
#print axioms WebKD.throttle_sign
#print axioms WebKD.anchorLoss_pStar_le
#print axioms WebKD.pStar_pos_le_one
#print axioms WebKD.cap_exists_iff
#print axioms WebKD.pStar_lt_one_iff
#print axioms WebKD.pStar_lt_one_iff_beta
#print axioms WebKD.budget_bound
#print axioms WebKD.budget_eq_of_interior
#print axioms WebKD.budget_uncapped
#print axioms WebKD.budget_anti_in_beta
#print axioms WebKD.budget_mono_in_betaBar
-- weighting
#print axioms WebKD.uniform_weights_recover_sft
#print axioms WebKD.LSW_eq_expectation
#print axioms WebKD.qw_sum_eq_one
#print axioms WebKD.rankMap_zero_spread
#print axioms WebKD.rankMap_mean_of_ranks
#print axioms WebKD.avgRank_sum
#print axioms WebKD.avgRank_lt_of_lt
#print axioms WebKD.rankMap_monovary
#print axioms WebKD.rankMap_top
#print axioms WebKD.rankMap_bottom
#print axioms WebKD.spread_eq_zero_of_tie
#print axioms WebKD.dead_zone_step_not_sft
#print axioms WebKD.pmi_eq
-- premium split and rank weighting
#print axioms WebKD.hasDerivAt_nll_coord
#print axioms WebKD.sandwich
#print axioms WebKD.grad_norm_bounds
#print axioms WebKD.chord_le_one_sub_exp_neg
#print axioms WebKD.loss_share
#print axioms WebKD.grad_share
#print axioms WebKD.grad_share_logits
#print axioms WebKD.deployment_prob_product
#print axioms WebKD.leave_one_out_recovers
#print axioms WebKD.mu_neg
#print axioms WebKD.hasDerivAt_log_deployment
#print axioms WebKD.mu_strictMono
#print axioms WebKD.mu_strictMonoOn_kappa
#print axioms WebKD.mu_strictAntiOn_kappa
#print axioms WebKD.rank_order_optimal
#print axioms WebKD.order_only
-- the single-round bound
#print axioms WebKD.sum_sub_mul_le_tv
#print axioms WebKD.value_gap_le
#print axioms WebKD.expectAt_replay
#print axioms WebKD.relabel_value_solved
#print axioms WebKD.relabel_value_failed
#print axioms WebKD.Population.prob_failed_notlate
#print axioms WebKD.Population.restricted_sum_eq
#print axioms WebKD.expectAt_eq_zero_of_dead
#print axioms WebKD.dead_prefix_of_not_recoverable
#print axioms WebKD.late_localization_dead
#print axioms WebKD.late_localization_dead_reward
#print axioms WebKD.state_eq_run_take
#print axioms WebKD.cell_sum_eq
#print axioms WebKD.regime_decomposition_le
#print axioms WebKD.comparator_failure_eq
#print axioms WebKD.single_round_bound_population
