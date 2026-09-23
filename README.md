# WebKD

**Student-aligned text distillation for web agents.**

WebKD turns a black-box teacher into an editor of a student's failed web
trajectories. It combines **student-style corrections**, **state-consistency
filtering**, **action-informativeness weighting**, and a **rationale anchor to
the frozen student**. The teacher supplies text; teacher logits and a shared
teacher–student tokenizer are not required.

## Navigation

| Start here | Contents |
| --- | --- |
| [Method](docs/method.md) | Editing policy, token weights, and the implemented objective |
| [Data pipeline](docs/data.md) | Trajectory format, teacher setup, filtering, and scoring |
| [Training](docs/training.md) | LLaMA-Factory integration, configuration, and launch commands |
| [Baselines](docs/baselines.md) | Editing, supervised learning, weighting, and policy-gradient baselines |
| [Source](src/webkd) | Reusable WebKD implementation |
| [Tests](tests) | Synthetic regression and training-objective tests |

## Install

Use Python 3.11 or newer in a virtual environment, then install from this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[teacher,scoring,dev]"
webkd --help
```

This installs the data pipeline and clients. Live browser rollouts also require
the TimeWarp distribution of BrowserGym and AgentLab, and running benchmark
servers. Training uses the separately installed [LLaMA-Factory integration](docs/training.md).
A vLLM server with prompt-logprob support is needed for action-informativeness
scoring. Model checkpoints, benchmark data, and credentials are supplied by the user.

## Pipeline

```text
Student trajectories
    → teacher edits in the student's style
    → state-consistency filtering
    → ShareGPT training corpus
    → frozen-student action-informativeness scores
    → token weights + rationale anchor mask
    → WebKD training
```

Given edited trajectories, the CPU preparation stages are:

```bash
webkd filter --edits data/edits/webkd_none/v6 --out data/filtered
webkd convert data/filtered --output data/webkd.json --think --plan --memory
webkd prepare-scoring --dataset data/webkd.json --out artifacts/scoring.json
webkd build-anchor --corpus data/webkd.json --out-dir artifacts
```

Then score the prepared requests under the frozen student, build weights, and
train. See the [complete data workflow](docs/data.md) and [training commands](docs/training.md).
Every stage has its own `--help`.

## Repository layout

```text
src/webkd/                  Editing, filtering, scoring, and data tools
src/webkd/training/         Token alignment, masking, and weighted loss
integrations/llamafactory/  Preprocessing, collation, and trainer integration
baselines/rl/               Multi-turn GRPO and related policy-gradient objectives
configs/                   General training configuration
scripts/                   Integration installation utility
tests/                     Synthetic regression tests
docs/                      Method, data, training, and baseline guides
```

## Tests

```bash
python -m pytest tests
ruff check src integrations scripts tests baselines
ruff format --check src integrations scripts tests baselines
```

The core tests use synthetic trajectories and mocked teachers, without a browser,
network calls, or GPU. Training-objective tests additionally require PyTorch and
the documented training dependencies. No performance results, checkpoints,
trajectory dumps, or run-specific configurations are distributed in this repository.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for licensing and upstream attribution.
