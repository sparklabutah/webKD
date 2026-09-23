# Data pipeline

## Inputs

The editing pipeline consumes the TimeWarp/AgentLab trajectory format:

```text
episodes/
  episode_on_timewarp.<task_id>_<seed>/
    summary_info.json
    training_data/
      system_prompt_step_0.json
      user_prompt_step_0.json
      agent_output_step_0.json
      ...
```

Prompt records have `role` and `content` fields. Agent outputs may contain a full
tagged `content` string or the legacy `think` and `action` fields. The user prompt
contains `## Goal:` and `## AXTree:` sections. `summary_info.json` supplies
`cum_reward`, `n_steps`, and an optional `err_msg`. Successful source episodes
have `cum_reward >= 1`. Use training tasks only for editing and style exemplars.

Each converted ShareGPT record contains a `system` string and a `conversations`
array with one `human` input and one `gpt` target. The scoring pipeline expects
this single-response format. Sidecars must be regenerated whenever target text
or tokenization changes.

## Live editing

Install the **TimeWarp-compatible** BrowserGym and AgentLab distribution in a
rollout environment and configure its benchmark servers. Stock AgentLab alone
does not provide TimeWarp's plan/memory prompt extensions or benchmark
registration. WebKD includes its trajectory-recording adapter in
[`recording_agent.py`](../src/webkd/recording_agent.py).

Set `TW_WIKI`, `TW_NEWS`, `TW_WEBSHOP`, and `TW_HOME` to your running benchmark
endpoints. Supply provider credentials through `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`; credentials are never part of a configuration file.

```bash
export WEBKD_CORRECTOR_PROVIDER=openai
export WEBKD_CORRECTOR_MODEL=your-teacher-model
export WEBKD_STUDENT_MODEL=your-student-model

webkd edit --episodes data/student_episodes --out data/edits \
  --mode webkd --style-bank data/student_episodes --style-k 3 --version 6
```

`--mode webkd` generates student-style teacher targets. The filtering and loss
stages below are also required for the full method. Editing keeps the environment
plan-free; optional oracle information is supplied only to the teacher. For
oracle baselines, set `WEBKD_TASK_CONFIG` to the benchmark's raw task-config JSON.
No task configuration is needed for ordinary unprivileged editing.

For a local OpenAI-compatible corrector, use `WEBKD_CORRECTOR_PROVIDER=local`,
`WEBKD_CORRECTOR_BASE_URL`, and an explicit `WEBKD_CORRECTOR_MODEL`.
`--resume` skips completed tasks. `--shard` and `--num-shards` partition a fixed
episode list deterministically; a failed episode causes the editing process to
exit unsuccessfully.

## Filter and convert

Choose a new output directory for filtering:

```bash
webkd filter --edits data/edits/webkd_none/v6 --out data/filtered --thresh 0.8
webkd convert data/filtered --output data/webkd.json --think --plan --memory
webkd prepare-scoring --dataset data/webkd.json --out artifacts/scoring.json
```

The converter preserves the exact structured target format used by the loss
sidecars. The default filter threshold is `0.8`.

## Score action informativeness

Serve the frozen student with a vLLM deployment that accepts completion requests
with `echo=true` and `logprobs=1`. The model and tokenizer must match the student
used for training. Start the service using your deployment's documented command
and assigned resources; WebKD's scoring client does not allocate GPUs or start
servers.

```bash
webkd score --port 8000 --model your-served-student \
  --tokenizer models/student --corpus artifacts/scoring.json \
  --out artifacts/action_scores.jsonl --workers 4 --max-len 30000

webkd build-weights --dataset data/webkd.json \
  --scores artifacts/action_scores.jsonl --out artifacts/webkd.weights.json

webkd build-anchor --corpus data/webkd.json --out-dir artifacts
```

This creates `artifacts/webkd.klgamma_rationale.json`. The weighting builder
rejects incomplete scoring by default. Repair failed requests before continuing;
`--allow-incomplete` explicitly enables the legacy uniform fallback and should
not be used for the full WebKD configuration.

Scores, sidecars, generated datasets, and optional diagnostic reports are runtime
artifacts. They are excluded from version control.
