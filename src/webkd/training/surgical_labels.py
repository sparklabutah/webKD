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

IGNORE_INDEX = -100

ENV_MASK = "WEBKD_SURGICAL_MASK"
ENV_STRICT = "WEBKD_SURGICAL_STRICT"

_TABLE = None
_TABLE_PATH = None


def surgical_enabled() -> bool:
    """True only when WEBKD_SURGICAL_MASK names a side-car. Cheap enough to call per example."""
    return bool(os.environ.get(ENV_MASK, "").strip())


def _strict() -> bool:
    return os.environ.get(ENV_STRICT, "1").strip() not in ("0", "false", "False", "")


def load_table(path: str) -> dict:
    """Read (and cache) the side-car. Accepts either {"entries": {...}} or a bare {sha1: entry}."""
    global _TABLE, _TABLE_PATH
    if _TABLE is not None and _TABLE_PATH == path:
        return _TABLE

    with open(path, encoding="utf-8") as f:
        blob = json.load(f)

    entries = blob.get("entries", blob) if isinstance(blob, dict) else blob
    if not isinstance(entries, dict):
        raise ValueError(f"{ENV_MASK}={path} is not a surgical mask side-car")

    _TABLE, _TABLE_PATH = entries, path
    return entries


def set_table(entries: dict, path: str = "<inline>") -> None:
    global _TABLE, _TABLE_PATH
    _TABLE, _TABLE_PATH = entries, path


def _current_table() -> dict:
    if _TABLE is not None and _TABLE_PATH == "<inline>":
        return _TABLE

    return load_table(os.environ[ENV_MASK])


def content_key(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def token_keep_mask(offsets, spans) -> list[bool]:
    """Per-token keep flags. A token is dropped ONLY if its whole char span sits inside a masked"""
    keep = []
    for start, end in offsets:
        drop = False
        if end > start:
            for lo, hi in spans:
                if start >= lo and end <= hi:
                    drop = True
                    break

        keep.append(not drop)

    return keep


def mask_target_label(
    tokenizer, content: str, target_ids: list[int], target_label: list[int]
) -> list[int]:
    """Return a NEW label list with the unchanged sections of `content` set to IGNORE_INDEX."""
    table = _current_table()
    entry = table.get(content_key(content))
    if entry is None:
        if _strict():
            raise ValueError(
                "surgical mask: assistant message not found in the side-car "
                f"({os.environ.get(ENV_MASK, _TABLE_PATH)}). The dataset and the mask file must come "
                f"from the same build_surgical.py run. Set {ENV_STRICT}=0 to train unmasked."
            )

        return list(target_label)

    if "len" in entry and entry["len"] != len(content):
        raise ValueError("surgical mask: SHA1 hit but content length differs -- stale side-car?")

    if entry.get("changed", None) == []:
        raise ValueError(
            "surgical mask: sample has no changed section -- rebuild without --keep-unchanged"
        )

    spans = entry.get("mask_spans") or []
    if not spans:
        return list(target_label)

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("surgical mask needs a fast tokenizer (offset mapping)")

    encoded = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])

    n = min(len(ids), len(target_label))
    if ids[:n] != list(target_ids[:n]):
        raise ValueError(
            "surgical mask: the assistant message does not tokenize to the head of the encoded "
            "target -- the template rewrote the content, so the character spans are meaningless."
        )

    keep = token_keep_mask(offsets[:n], spans)
    labels = list(target_label)
    for i in range(n):
        if not keep[i]:
            labels[i] = IGNORE_INDEX

    if labels and all(label == IGNORE_INDEX for label in labels):
        raise ValueError(
            "surgical mask: every label masked -- all-unchanged samples must be dropped at build time"
        )

    return labels
