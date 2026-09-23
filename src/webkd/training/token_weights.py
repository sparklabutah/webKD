# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training-time token alignment and loss utilities."""

import hashlib
import json
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch

IGNORE_INDEX = -100

SIDECAR_VERSION = 2
DEFAULT_FLOOR = 0.1
MAX_WEIGHT = 1.0

ENV_WEIGHTS = "WEBKD_TOKEN_WEIGHTS"
ENV_FLOOR = "WEBKD_TOKEN_WEIGHT_FLOOR"
ENV_NORM = "WEBKD_TOKEN_WEIGHT_NORM"
ENV_STRICT = "WEBKD_TOKEN_WEIGHTS_STRICT"

_TABLE = None

_MAX_TEMPLATE_PREFIX_TOKENS = 32

_TABLE_META: dict[str, Any] = {}
_TABLE_PATH = None


def token_weights_enabled() -> bool:
    """True only when WEBKD_TOKEN_WEIGHTS names a side-car. Cheap enough to call per example."""
    return bool(os.environ.get(ENV_WEIGHTS, "").strip())


def _strict() -> bool:
    return os.environ.get(ENV_STRICT, "1").strip() not in ("0", "false", "False", "")


def normalization_mode() -> str:
    """``batch`` (default) | ``window`` | ``count`` -- see the NORMALISATION section above."""
    mode = os.environ.get(ENV_NORM, "batch").strip() or "batch"
    if mode not in ("batch", "window", "count"):
        raise ValueError(f"{ENV_NORM} must be 'batch', 'window' or 'count', got {mode!r}")

    return mode


def env_floor() -> Optional[float]:
    """The floor from the environment, or None when unset (then the side-car / default decides)."""
    raw = os.environ.get(ENV_FLOOR, "").strip()
    if not raw:
        return None

    value = float(raw)
    if not 0.0 <= value <= MAX_WEIGHT:
        raise ValueError(f"{ENV_FLOOR}={raw} must lie in [0, {MAX_WEIGHT}]")

    return value


def resolve_floor(entry: Optional[dict] = None, meta: Optional[dict] = None) -> float:
    """Floor precedence: env var > per-entry "floor" > side-car "floor" > DEFAULT_FLOOR."""
    value = env_floor()
    if value is not None:
        return value

    if entry is not None and entry.get("floor") is not None:
        return float(entry["floor"])

    if meta is not None and meta.get("floor") is not None:
        return float(meta["floor"])

    return DEFAULT_FLOOR


def content_key(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def load_table(path: str) -> dict:
    """Read (and cache) the side-car. Accepts {"entries": {...}} or a bare {sha1: entry}."""
    global _TABLE, _TABLE_META, _TABLE_PATH
    if _TABLE is not None and _TABLE_PATH == path:
        return _TABLE

    with open(path, encoding="utf-8") as f:
        blob = json.load(f)

    if not isinstance(blob, dict):
        raise ValueError(f"{ENV_WEIGHTS}={path} is not a token-weight side-car")

    entries = blob.get("entries", blob)
    if not isinstance(entries, dict):
        raise ValueError(f"{ENV_WEIGHTS}={path} is not a token-weight side-car")

    version = int(blob.get("version", 1)) if "entries" in blob else 1
    if version > SIDECAR_VERSION:
        raise ValueError(
            f"{path} is a v{version} side-car but this build only understands v{SIDECAR_VERSION}"
        )

    meta = {k: v for k, v in blob.items() if k != "entries"} if "entries" in blob else {}
    _TABLE, _TABLE_META, _TABLE_PATH = entries, meta, path
    return entries


def set_table(entries: Optional[dict], path: str = "<inline>", meta: Optional[dict] = None) -> None:
    """Inject a side-car table directly."""
    global _TABLE, _TABLE_META, _TABLE_PATH
    _TABLE, _TABLE_META, _TABLE_PATH = entries, meta or {}, path


ENV_KL_GAMMA = "WEBKD_KL_GAMMA_WEIGHTS"
_GTABLE = None
_GTABLE_META: dict[str, Any] = {}
_GTABLE_PATH = None


def kl_gamma_enabled() -> bool:
    """True only when WEBKD_KL_GAMMA_WEIGHTS names a side-car."""
    return bool(os.environ.get(ENV_KL_GAMMA, "").strip())


def _gamma_table() -> tuple[dict, dict]:
    global _GTABLE, _GTABLE_META, _GTABLE_PATH
    path = os.environ[ENV_KL_GAMMA]
    if _GTABLE is not None and _GTABLE_PATH == path:
        return _GTABLE, _GTABLE_META

    with open(path, encoding="utf-8") as f:
        blob = json.load(f)

    if not isinstance(blob, dict):
        raise ValueError(f"{ENV_KL_GAMMA}={path} is not a token-weight side-car")

    entries = blob.get("entries", blob)
    if not isinstance(entries, dict):
        raise ValueError(f"{ENV_KL_GAMMA}={path} is not a token-weight side-car")

    _GTABLE = entries
    _GTABLE_META = {k: v for k, v in blob.items() if k != "entries"} if "entries" in blob else {}
    _GTABLE_PATH = path
    return _GTABLE, _GTABLE_META


def target_kl_gamma(
    tokenizer, content: str, target_ids: list[int], target_label: list[int]
) -> list[float]:
    """gamma_i for one assistant turn, from the role-mask side-car."""
    weights, _labels = target_token_weights(
        tokenizer, content, target_ids, target_label, tables=_gamma_table()
    )
    return weights


def _current_table() -> tuple[dict, dict]:
    if _TABLE is not None and _TABLE_PATH == "<inline>":
        return _TABLE, _TABLE_META

    table = load_table(os.environ[ENV_WEIGHTS])
    return table, _TABLE_META


def flatten_weight_spans(spans) -> list[list[float]]:
    """Sort + de-overlap ``[[start, end, weight], ...]`` into DISJOINT ascending spans."""
    clean = []
    for i, span in enumerate(spans or []):
        if len(span) != 3:
            raise ValueError(f"weight span {span!r} must be [start, end, weight]")

        start, end, weight = int(span[0]), int(span[1]), float(span[2])
        if start < 0 or end < start:
            raise ValueError(f"weight span {span!r} is not a valid half-open char range")

        if end > start:
            clean.append((start, end, weight, i))

    if not clean:
        return []

    clean.sort(key=lambda s: (s[0], s[3]))
    if all(clean[k][0] >= clean[k - 1][1] for k in range(1, len(clean))):
        return [[s[0], s[1], s[2]] for s in clean]

    points = sorted({p for s in clean for p in (s[0], s[1])})
    out: list[list[float]] = []
    for lo, hi in zip(points[:-1], points[1:]):
        winner = None
        for start, end, weight, idx in clean:
            if start <= lo and end >= hi and (winner is None or idx > winner[0]):
                winner = (idx, weight)

        if winner is None:
            continue

        if out and out[-1][1] == lo and out[-1][2] == winner[1]:
            out[-1][1] = hi
        else:
            out.append([lo, hi, winner[1]])

    return out


def token_weight_vector(
    offsets,
    weight_spans=None,
    default_weight: float = 1.0,
    floor: float = DEFAULT_FLOOR,
    mask_spans=None,
    max_weight: float = MAX_WEIGHT,
) -> list[float]:
    """Per-token weights for `offsets` (the tokenizer's own char offset mapping)."""
    spans = weight_spans or []
    masks = mask_spans or []
    out = []
    j = 0
    for start, end in offsets:
        if end <= start:
            weight = default_weight
        else:
            while j < len(spans) and spans[j][1] <= start:
                j += 1

            acc, covered, k = 0.0, 0, j
            while k < len(spans) and spans[k][0] < end:
                lo, hi = max(start, spans[k][0]), min(end, spans[k][1])
                if hi > lo:
                    acc += (hi - lo) * spans[k][2]
                    covered += hi - lo

                k += 1

            acc += (end - start - covered) * default_weight
            weight = acc / float(end - start)

        weight = min(max(weight, floor), max_weight)
        if end > start and any(start >= lo and end <= hi for lo, hi in masks):
            weight = 0.0

        out.append(weight)

    return out


def target_token_weights(
    tokenizer, content: str, target_ids: list[int], target_label: list[int], tables=None
) -> tuple[list[float], list[int]]:
    """Return ``(weights, labels)`` for one assistant turn."""
    table, meta = tables if tables is not None else _current_table()
    entry = table.get(content_key(content))
    if entry is None:
        if _strict():
            raise ValueError(
                "token weights: assistant message not found in the side-car "
                f"({os.environ.get(ENV_WEIGHTS, _TABLE_PATH)}). The dataset and the weight file must come from "
                f"the same build. Set {ENV_STRICT}=0 to train with uniform weights."
            )

        return [1.0] * len(target_label), list(target_label)

    if "len" in entry and entry["len"] != len(content):
        raise ValueError("token weights: SHA1 hit but content length differs -- stale side-car?")

    floor = resolve_floor(entry, meta)
    default_weight = float(entry.get("default_weight", meta.get("default_weight", 1.0)))
    envelope_weight = float(entry.get("envelope_weight", meta.get("envelope_weight", 1.0)))
    spans = flatten_weight_spans(entry.get("weight_spans") or [])
    masks = entry.get("mask_spans") or []

    weights = [envelope_weight] * len(target_label)
    labels = list(target_label)
    if not spans and not masks and default_weight == 1.0 and envelope_weight == 1.0:
        for i, label in enumerate(labels):
            if label == IGNORE_INDEX:
                weights[i] = 0.0

        return weights, labels

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("token weights need a fast tokenizer (offset mapping)")

    encoded = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])

    target_ids = list(target_ids)
    start, limit = -1, min(len(target_ids), _MAX_TEMPLATE_PREFIX_TOKENS)
    for k in range(limit + 1):
        m = min(len(ids), len(target_ids) - k)
        if m <= 0:
            break
        if target_ids[k : k + m] == ids[:m]:
            start = k
            break
    if start < 0:
        raise ValueError(
            "token weights: the assistant message does not tokenize to the head of the encoded target -- "
            "the template rewrote the content, so the character spans are meaningless."
        )

    n = min(len(ids), len(target_label) - start)
    if n <= 0:
        raise ValueError(
            "token weights: the encoded target is shorter than the template prefix it starts with"
        )

    vector = token_weight_vector(offsets[:n], spans, default_weight, floor, masks)
    for i in range(start):
        weights[i] = default_weight
    for i in range(n):
        weights[start + i] = vector[i]
        if vector[i] <= 0.0:
            labels[start + i] = IGNORE_INDEX

    for i, label in enumerate(labels):
        if label == IGNORE_INDEX:
            weights[i] = 0.0

    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError(
            "token weights: every label of this sample is masked -- such samples must be dropped"
        )

    return weights, labels


def applied_weight_sum(labels: "torch.Tensor", weights: "torch.Tensor") -> "torch.Tensor":
    """Sum of the weights that will actually multiply a cross-entropy term."""
    import torch

    shift_labels = labels[..., 1:]
    shift_weights = weights[..., 1:].to(dtype=torch.float32)
    return (shift_weights * shift_labels.ne(IGNORE_INDEX)).sum()


def valid_token_count(labels: "torch.Tensor") -> "torch.Tensor":
    """Number of positions that carry a loss term, i.e. stock's denominator (``count`` mode)."""
    return labels[..., 1:].ne(IGNORE_INDEX).sum()


def weighted_lm_loss(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    weights: "torch.Tensor",
    normalizer: Optional["torch.Tensor"] = None,
    check_alignment: bool = True,
) -> "torch.Tensor":
    """``sum_i w_i * CE_i / normalizer`` over the causal-shifted positions."""
    import torch
    import torch.nn.functional as F

    if weights.shape != labels.shape:
        raise ValueError(
            f"token weights {tuple(weights.shape)} do not match labels {tuple(labels.shape)}"
        )

    shift_logits = logits[..., :-1, :].float()
    shift_labels = labels[..., 1:]
    shift_weights = weights[..., 1:].to(dtype=torch.float32)
    valid = shift_labels.ne(IGNORE_INDEX)
    if check_alignment and not torch.equal(shift_weights.gt(0.0), valid):
        raise ValueError(
            "token weights are misaligned with the labels (weight>0 must hold exactly where "
            "label != -100). Check the collator's padding side / the side-car build."
        )

    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1).masked_fill(~valid.reshape(-1), 0),
        reduction="none",
    ).view_as(shift_labels)
    numerator = (ce * shift_weights * valid).sum()
    if normalizer is None:
        denominator = (shift_weights * valid).sum()
    elif torch.is_tensor(normalizer):
        denominator = normalizer.to(device=numerator.device, dtype=numerator.dtype)
    else:
        denominator = torch.tensor(
            float(normalizer), device=numerator.device, dtype=numerator.dtype
        )

    return numerator / torch.clamp(denominator, min=1e-8)


def pad_token_weights(rows, length: int, padding_side: str = "right") -> "torch.Tensor":
    """Pad per-sample weight lists to `length` with 0.0, on `padding_side`."""
    import torch

    padded = []
    for row in rows:
        row = [float(x) for x in row]
        if len(row) > length:
            raise ValueError(
                f"token weight row of length {len(row)} does not fit a batch of width {length}"
            )

        pad = [0.0] * (length - len(row))
        padded.append(row + pad if padding_side == "right" else pad + row)

    return torch.tensor(padded, dtype=torch.float32)
