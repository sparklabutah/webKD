"""Build and validate character-span sidecars shared with the training loss."""

from __future__ import print_function

import json
import os
import re

from webkd.training import token_weights as TW

from webkd import surgical as SU

SIDECAR_VERSION = TW.SIDECAR_VERSION
DEFAULT_FLOOR = TW.DEFAULT_FLOOR
MAX_WEIGHT = TW.MAX_WEIGHT
IGNORE_INDEX = TW.IGNORE_INDEX
ENV_WEIGHTS = TW.ENV_WEIGHTS
ENV_FLOOR = TW.ENV_FLOOR
ENV_NORM = TW.ENV_NORM
ENV_STRICT = TW.ENV_STRICT
content_key = TW.content_key
flatten_weight_spans = TW.flatten_weight_spans
token_weight_vector = TW.token_weight_vector

TAG_ORDER = SU.TAG_ORDER


def load_conversations(dataset_path):
    with open(dataset_path) as f:
        blob = json.load(f)

    out = []
    for i, sample in enumerate(blob):
        turns = sample.get("conversations") or []
        human = ""
        for turn in turns:
            if turn.get("from") in ("human", "user"):
                human = turn.get("value", "")
            elif turn.get("from") in ("gpt", "assistant"):
                content = turn.get("value", "")
                out.append(
                    {"index": i, "content": content, "prompt": human, "sha1": content_key(content)}
                )
    return out


def tag_spans(content):
    """{tag: {"outer": (s, e), "inner": (s, e)}} character spans of the five sections IN `content`."""
    out = {}
    for tag in TAG_ORDER:
        m = re.search(SU.STRICT_PATTERNS[tag], content, re.DOTALL)
        if not m:
            continue
        text = m.group(1)
        i, j = m.span(1)
        lead = len(text) - len(text.lstrip())
        trail = len(text) - len(text.rstrip())
        out[tag] = {"outer": m.span(0), "inner": (i + lead, j - trail)}
    return out


def weight_spans_from_tags(content, weight_by_tag, policy="section", spans=None):
    """[[start, end, weight], ...] for whole sections. `policy`: "section" (tags included) | "inner"."""
    key = "inner" if policy == "inner" else "outer"
    found = spans if spans is not None else tag_spans(content)
    out = []
    for tag in TAG_ORDER:
        if tag not in weight_by_tag or tag not in found:
            continue
        s, e = found[tag][key]
        if e > s:
            out.append([int(s), int(e), float(weight_by_tag[tag])])
    return out


def char_spans_from_token_weights(offsets, weights):
    """Turn per-token (offset_mapping, weight) into merged char spans."""
    out = []
    for (s, e), w in zip(offsets, weights):
        if e <= s:
            continue
        w = float(w)
        if out and out[-1][1] == s and out[-1][2] == w:
            out[-1][1] = e
        else:
            out.append([int(s), int(e), w])
    return out


def build_entry(
    content,
    weight_spans=None,
    mask_spans=None,
    default_weight=1.0,
    floor=None,
    extra=None,
    envelope_weight=None,
):
    """One side-car entry for one assistant turn, validated against `content`."""
    spans = flatten_weight_spans(weight_spans or [])
    for s, e, w in spans:
        if e > len(content):
            raise ValueError(
                "weight span [%d, %d) runs past the end of the %d-char target"
                % (s, e, len(content))
            )
        if not (0.0 <= w <= MAX_WEIGHT):
            raise ValueError("weight %r outside [0, %s]" % (w, MAX_WEIGHT))

    entry = {"len": len(content)}
    if spans:
        entry["weight_spans"] = spans
    if default_weight != 1.0:
        entry["default_weight"] = float(default_weight)
    if mask_spans:
        for s, e in mask_spans:
            if e > len(content):
                raise ValueError("mask span [%d, %d) runs past the end of the target" % (s, e))
        entry["mask_spans"] = [[int(s), int(e)] for s, e in mask_spans]
    if floor is not None:
        entry["floor"] = float(floor)
    if envelope_weight is not None:
        entry["envelope_weight"] = float(envelope_weight)
    if extra:
        entry.update(extra)
    return entry


def build_entries(
    spans_by_content,
    default_weight=1.0,
    floor=None,
    mask_spans_by_content=None,
    extra_by_content=None,
    envelope_weight=None,
):
    """{sha1: entry} from {content -> [(start, end, weight), ...]}."""
    entries = {}
    for content, spans in spans_by_content.items():
        entries[content_key(content)] = build_entry(
            content,
            weight_spans=spans,
            mask_spans=(mask_spans_by_content or {}).get(content),
            default_weight=default_weight,
            floor=floor,
            extra=(extra_by_content or {}).get(content),
            envelope_weight=envelope_weight,
        )
    return entries


def write_sidecar(
    path,
    spans_by_content,
    dataset=None,
    default_weight=1.0,
    floor=None,
    mask_spans_by_content=None,
    extra_by_content=None,
    meta=None,
    entries=None,
    envelope_weight=None,
):
    """Write a v2 side-car. Returns the {sha1: entry} table it wrote."""
    if entries is None:
        entries = build_entries(
            spans_by_content,
            default_weight,
            floor,
            mask_spans_by_content,
            extra_by_content,
            envelope_weight,
        )

    blob = {"version": SIDECAR_VERSION, "entries": entries}
    if dataset is not None:
        blob["dataset"] = dataset
    if default_weight != 1.0:
        blob["default_weight"] = float(default_weight)
    if floor is not None:
        blob["floor"] = float(floor)
    if envelope_weight is not None:
        blob["envelope_weight"] = float(envelope_weight)
    if meta:
        blob.update(meta)

    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as f:
        json.dump(blob, f)
    return entries


def validate(entries, conversations, require_all=True):
    """Raise unless every conversation is covered and every span fits its target. Returns stats."""
    missing = [c["sha1"] for c in conversations if c["sha1"] not in entries]
    if require_all and missing:
        raise ValueError(
            "%d/%d conversations have no weight entry (first: %s)"
            % (len(missing), len(conversations), missing[0])
        )

    stale = []
    for c in conversations:
        entry = entries.get(c["sha1"])
        if entry is None:
            continue
        if entry.get("len") is not None and entry["len"] != len(c["content"]):
            stale.append(c["sha1"])
        for span in entry.get("weight_spans") or []:
            if span[1] > len(c["content"]):
                raise ValueError("entry %s has a span past the end of its target" % c["sha1"])
    if stale:
        raise ValueError(
            "%d entries do not match their conversation's length (stale build)" % len(stale)
        )

    return {"conversations": len(conversations), "entries": len(entries), "missing": len(missing)}


def preview_weights(tokenizer, content, entry, floor=None, group=True):
    """[(text, weight), ...] -- what the TRAINER will actually apply, decoded back to text."""
    enc = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(enc["offset_mapping"])
    weights = token_weight_vector(
        offsets,
        flatten_weight_spans(entry.get("weight_spans") or []),
        float(entry.get("default_weight", 1.0)),
        DEFAULT_FLOOR if floor is None else floor,
        entry.get("mask_spans") or [],
    )
    rows = []
    for (s, e), w in zip(offsets, weights):
        piece = content[s:e]
        if group and rows and abs(rows[-1][1] - w) < 1e-9:
            rows[-1][0] += piece
        else:
            rows.append([piece, w])
    return [(text, weight) for text, weight in rows]


def summarize(entries, conversations=None):
    n_spans, total_span_chars, weights = 0, 0, []
    for entry in entries.values():
        for s, e, w in entry.get("weight_spans") or []:
            n_spans += 1
            total_span_chars += e - s
            weights.append(w)

    out = {"entries": len(entries), "weight_spans": n_spans, "weighted_chars": total_span_chars}
    if weights:
        weights.sort()
        out["weight_min"] = weights[0]
        out["weight_max"] = weights[-1]
        out["weight_mean"] = sum(weights) / float(len(weights))
        out["weight_median"] = weights[len(weights) // 2]
    if conversations:
        out["target_chars"] = sum(len(c["content"]) for c in conversations)
    return out
