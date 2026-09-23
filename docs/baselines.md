# Baselines

The repository retains the reusable baseline implementations alongside WebKD.
Names describe behavior rather than internal run identifiers.

## Editing and supervised learning

| Method | How to select it | Behavior |
| --- | --- | --- |
| Behavior cloning / uniform SFT | `webkd train --method sft` | Uniform cross-entropy on the supplied demonstrations |
| Teacher relabeling | `webkd edit --mode teacher_relabel` | Localize an error, then relabel at student-visited states |
| Expert completion | `webkd edit --mode expert_completion` | Replay the prefix, then let the teacher execute the corrected suffix |
| On-policy text distillation | `webkd edit --mode on_policy_distillation` | Relabel unsuccessful student trajectories from the first step; retain successful ones |
| Style-matched teacher | `webkd edit --mode style_matched` | Relabel with student rationale exemplars |
| Frozen-student restyling | `webkd edit --mode student_restyle` | Teacher correction followed by a student rewrite; preserve the teacher action |
| Stepwise takeover | `webkd edit --mode takeover` | Student proposes each step; execute a teacher fix when needed |
| Self-correction | `webkd edit-offline --corrector self` | Use a served student as the offline corrector |

Restyling requires `--restyle-base-url` and `--restyle-model`. Takeover requires
`--student-base-url` and `--student-model`. Offline edits do not re-execute the
browser; they must not be represented as environment-validated trajectories.

State-consistency filtering is available independently through `webkd filter`.
The style-matched baseline becomes the WebKD data stage after filtering;
the full WebKD method additionally requires action-informativeness weights and
the rationale anchor.

## Alternative loss support and weights

All builders accept `--help` and operate on user-supplied data:

| Baseline | Entry point |
| --- | --- |
| Surgical masking of unchanged sections | `python -m webkd.build_surgical` |
| Soft weighting of unchanged sections | `python -m webkd.build_edit_weights` |
| Student-likelihood weighting | `python -m webkd.build_likelihood_weights` |
| Teacher-consistency weighting | `python -m webkd.sample_consistency`, then `python -m webkd.build_consistency_weights` |
| Groundedness weighting | `python -m webkd.build_grounding_weights` |
| Action-role or rationale-role anchor | `webkd build-anchor --roles action,rationale` |

Loss sidecars share the token-alignment implementation under
[`src/webkd/training`](../src/webkd/training). Read the builder's required inputs
before choosing defaults. Alternative weightings and role masks are baseline
components; they are not the default WebKD method.

## Multi-turn policy gradients

[`baselines/rl`](../baselines/rl) preserves the TimeWarp adaptation of
WebAgent-R1's environment-driven GRPO trainer, including GRPO, Dr. GRPO, the
nested turn/token mGRPO normalizer, turn weighting, and role-dependent anchors.

Run this baseline in its own environment with `trl==0.24.0`, a compatible
Transformers/PyTorch stack, vLLM, and the TimeWarp BrowserGym/AgentLab distribution.
It is separate from the WebKD supervised training environment. Set the benchmark
endpoints and judge credentials as described in the [data guide](data.md).

```bash
cd baselines/rl
export MODEL_PATH=your-student-checkpoint
export TW_OUTPUT_DIR=outputs/mgrpo
export TW_LOSS_TYPE=mgrpo
export TW_NUM_GENERATIONS=8
export TW_VLLM_SERVER_HOST=localhost
export TW_VLLM_SERVER_PORT=8000
export TW_REPORT_TO=none
accelerate launch train_timewarp.py
```

Start a compatible TRL vLLM server separately and configure Accelerate for your
assigned resources. The server and training processes must have appropriate,
nonconflicting device assignments. See the trainer and configuration utility for
the retained `TW_*` options. These refer to the TimeWarp benchmark integration.
