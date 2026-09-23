# Training

## Install the integration

WebKD extends LLaMA-Factory's supervised preprocessing, data collation, and loss
computation. Install the integration in a separate training environment. The
installer checks both the upstream revision and the original file contents;
it refuses unrelated or locally modified versions.

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git vendor/LLaMA-Factory
git -C vendor/LLaMA-Factory checkout 97b32d3133b501432141a82949d5c7bc4d94f23a
python scripts/install_llamafactory.py vendor/LLaMA-Factory
python -m pip install -e vendor/LLaMA-Factory
python -m pip install -e .
```

Install a PyTorch build appropriate for your accelerator using PyTorch's
installation instructions. Optional DeepSpeed, attention kernels, and serving
dependencies depend on the selected model and deployment. Use an environment
separate from the policy-gradient baseline, whose serving and environment
dependencies differ.

The CPU integration test was exercised with Transformers 5.8.0, PyTorch 2.12.1,
TRL 0.24.0, Accelerate 1.11.0, and Datasets 4.0.0. The pinned upstream checkout
defines the supported dependency ranges. This is a software compatibility check,
not a claim that every model or accelerator combination has been validated.

## WebKD

After the [data pipeline](data.md), run:

```bash
webkd train --method webkd --model models/student \
  --dataset data/webkd.json --output outputs/webkd \
  --template qwen3_5 --config configs/train.yaml \
  --weights artifacts/webkd.weights.json \
  --anchor artifacts/webkd.klgamma_rationale.json --kl-beta 0.03
```

Select the chat template for your model. The reference defaults to `--model`;
`--reference` can name the same frozen student checkpoint explicitly. Adjust the
general configuration for context length, batch size, epochs, precision, and
available memory. The configuration is an editable starting point, not a
model-specific reproduction claim.

Add `--dry-run` to validate data and sidecars and write the resolved configuration
without loading any model. Generated configuration, dataset registration, and
method options live inside `--output`. Credentials are not serialized.

The integration uses these environment variables internally:

| Variable | Meaning |
| --- | --- |
| `WEBKD_TOKEN_WEIGHTS` | Action-informativeness CE sidecar |
| `WEBKD_KL_GAMMA_WEIGHTS` | Separate role mask for the anchor |
| `WEBKD_KL_REF` | Frozen student checkpoint |
| `WEBKD_KL_BETA` | Anchor coefficient |
| `WEBKD_TOKEN_WEIGHT_NORM` | CE-only normalization: `batch`, `window`, or diagnostic `count` |
| `WEBKD_SURGICAL_MASK` | Unchanged-section hard masks for the surgical baseline |

For the combined WebKD objective, normalization is per microbatch as described in
the [method guide](method.md). Response packing, history masking, and prompt
supervision are unsupported by the supplied launch configuration. Weighted CE
and surgical hard-mask modes use different input contracts and must not be
enabled together.

The frozen reference is a dense replica on every training process. It needs the
same vocabulary as the trainable student. Fused losses that omit logits are not
compatible with the combined objective. Use the default logits-producing model
forward path and allow memory for both models.

## Supervised baselines

Behavior cloning and uniform fine-tuning use the same entry point with no
weighting or anchor:

```bash
webkd train --method sft --model models/student \
  --dataset data/baseline.json --output outputs/baseline \
  --template qwen3_5 --config configs/train.yaml
```

The selected corpus determines the editing baseline. See [baselines](baselines.md).
For specialized weighting or hard masking, use the corresponding sidecar with
the integrated LLaMA-Factory launcher directly.

## Distributed execution and evaluation

Launch LLaMA-Factory inside your scheduler allocation with its supported
distributed launcher. Match per-device batch size and gradient accumulation to
the intended effective batch. WebKD does not change device visibility, acquire
resources, or stop other processes.

Evaluate exported checkpoints using the same benchmark distribution, prompt
configuration, task split, and decoding settings as the source student. Browser
infrastructure, model downloads, and full GPU training are separate from the
synthetic CPU regression suite.

To include the random-model CPU integration test after installation:

```bash
WEBKD_TEST_INTEGRATION=1 CUDA_VISIBLE_DEVICES='' python -m pytest tests
```

It constructs its tokenizer and model locally, then checks preprocessing,
collation, the weighted objective, and frozen-reference gradients without
downloading a checkpoint.
