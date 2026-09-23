"""Segment rationales and map leave-one-out action contributions to token weights."""

from __future__ import print_function

import re

TAG_ORDER = ("think", "plan", "step", "memory", "action")
RATIONALE_TAGS = ("think", "plan", "memory")

MIN_SEG_CHARS = 24
MAX_SEGS_PER_STEP = 24
DEFAULT_TAU = 0.05
DEFAULT_FLOOR = 0.1

_BOUNDARY = re.compile(r"(?:\n+|(?<=[.!?;:])[ \t]+)")


def _nonspace(text):
    return len(re.sub(r"\s+", "", text))


def segment_inner(text, min_chars=MIN_SEG_CHARS, max_segs=None):
    """Split one rationale section's INNER text into [(start, end), ...] covering it exactly."""
    if not text:
        return []
    cuts = []
    pos = 0
    for m in _BOUNDARY.finditer(text):
        end = m.end()
        if end > pos:
            cuts.append((pos, end))
            pos = end
    if pos < len(text):
        cuts.append((pos, len(text)))
    if not cuts:
        cuts = [(0, len(text))]

    merged = []
    for s, e in cuts:
        if merged and _nonspace(text[s:e]) < min_chars:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    while len(merged) > 1 and _nonspace(text[merged[0][0] : merged[0][1]]) < min_chars:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)

    if max_segs:
        merged = _cap_segments(text, merged, max_segs)
    return merged


def _cap_segments(text, segs, max_segs):
    """Repeatedly merge the shortest adjacent pair until at most `max_segs` remain."""
    segs = list(segs)
    while len(segs) > max_segs:
        best, best_len = 0, None
        for i in range(len(segs) - 1):
            n = _nonspace(text[segs[i][0] : segs[i][1]]) + _nonspace(
                text[segs[i + 1][0] : segs[i + 1][1]]
            )
            if best_len is None or n < best_len:
                best, best_len = i, n
        segs[best] = (segs[best][0], segs[best + 1][1])
        segs.pop(best + 1)
    return segs


def step_segments(content, tag_spans_fn, min_chars=MIN_SEG_CHARS, max_segs=MAX_SEGS_PER_STEP):
    """[{"tag","start","end","text"}, ...] -- the ablatable rationale segments of one turn."""
    spans = tag_spans_fn(content)
    out = []
    for tag in RATIONALE_TAGS:
        if tag not in spans:
            continue
        a, b = spans[tag]["inner"]
        inner = content[a:b]
        for s, e in segment_inner(inner, min_chars=min_chars):
            out.append({"tag": tag, "start": a + s, "end": a + e, "text": content[a + s : a + e]})
    out.sort(key=lambda d: d["start"])
    if max_segs and len(out) > max_segs:
        out = _cap_step(content, out, max_segs)
    return out


def _cap_step(content, segs, max_segs):
    """Cap the whole step, merging only WITHIN a tag (never across a section boundary)."""
    segs = list(segs)
    while len(segs) > max_segs:
        best, best_len = None, None
        for i in range(len(segs) - 1):
            if segs[i]["tag"] != segs[i + 1]["tag"]:
                continue
            if segs[i]["end"] != segs[i + 1]["start"]:
                continue
            n = _nonspace(segs[i]["text"]) + _nonspace(segs[i + 1]["text"])
            if best_len is None or n < best_len:
                best, best_len = i, n
        if best is None:
            break
        segs[best] = {
            "tag": segs[best]["tag"],
            "start": segs[best]["start"],
            "end": segs[best + 1]["end"],
            "text": content[segs[best]["start"] : segs[best + 1]["end"]],
        }
        segs.pop(best + 1)
    return segs


def ablate(content, seg):
    """The target with `seg`'s characters deleted. Nothing else moves."""
    return content[: seg["start"]] + content[seg["end"] :]


def average_ranks(values):
    """Ascending average ranks (ties share the mean rank), 0-based."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def weights_from_contributions(
    contribs, floor=DEFAULT_FLOOR, tau=DEFAULT_TAU, transform="rank", mask_negative=False
):
    """Within-step normalisation onto [floor, 1] with the dead-zone guard."""
    n = len(contribs)
    if n == 0:
        return [], {"n": 0, "spread": 0.0, "shrink": 0.0}
    if n == 1:
        return (
            [0.0] if (mask_negative and contribs[0] < 0) else [1.0],
            {"n": 1, "spread": 0.0, "shrink": 0.0},
        )

    spread = max(contribs) - min(contribs)
    shrink = 1.0 if tau <= 0 else min(1.0, spread / float(tau))

    if transform == "minmax":
        lo, hi = min(contribs), max(contribs)
        rng = hi - lo
        fracs = [0.5 if rng <= 0 else (c - lo) / float(rng) for c in contribs]
    elif transform == "rank":
        ranks = average_ranks(contribs)
        fracs = [r / float(n - 1) for r in ranks]
    else:
        raise ValueError("unknown transform %r" % (transform,))

    weights = []
    for frac, c in zip(fracs, contribs):
        frac = 0.5 + (frac - 0.5) * shrink
        w = floor + (1.0 - floor) * frac
        w = min(1.0, max(floor, w))
        if mask_negative and c < 0:
            w = 0.0
        weights.append(w)
    return weights, {
        "n": n,
        "spread": spread,
        "shrink": shrink,
        "transform": transform,
        "mask_negative": bool(mask_negative),
        "n_negative": sum(1 for c in contribs if c < 0),
    }


def weight_spans_for_step(
    content,
    segs,
    contribs,
    floor=DEFAULT_FLOOR,
    tau=DEFAULT_TAU,
    transform="rank",
    mask_negative=False,
):
    """[[start, end, weight], ...] for one turn, plus the per-step info dict."""
    weights, info = weights_from_contributions(
        contribs, floor=floor, tau=tau, transform=transform, mask_negative=mask_negative
    )
    spans = []
    for seg, w in zip(segs, weights):
        if seg["end"] > seg["start"]:
            spans.append([int(seg["start"]), int(seg["end"]), float(w)])
    return spans, info
