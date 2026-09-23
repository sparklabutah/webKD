"""Teacher consistency weighting with action-conditioned rationale matching."""

from __future__ import print_function

import math
import re

TAG_ORDER = ("think", "plan", "step", "memory", "action")

SCORED_TAGS = ("think", "plan", "memory")

DEFAULT_FLOOR = 0.1

STOPWORDS = set(
    """
a an and are as at be been being but by can could did do does doing done for from had has have
having he her here hers him his how i if in into is it its itself just me my no nor not of off on
once only or other our ours out over own same she should so some such than that the their theirs
them then there these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours it's i'm we'll don't didn't isn't
""".split()
)

_WORD_RE = re.compile(r"[a-z0-9_]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")

MAX_UNIT_CHARS = 220


def _line_spans(text, offset=0):
    """[(start, end)] of the non-blank LINES of `text`, offsets shifted by `offset`."""
    out = []
    pos = 0
    for line in text.split("\n"):
        s, e = pos, pos + len(line)
        pos = e + 1
        stripped = line.strip()
        if not stripped:
            continue
        lead = len(line) - len(line.lstrip())
        trail = len(line) - len(line.rstrip())
        out.append((offset + s + lead, offset + e - trail))
    return out


def segment(text, offset=0, max_unit_chars=MAX_UNIT_CHARS):
    """Cut `text` into scoring UNITS: [(start, end)] char spans, offsets shifted by `offset`."""
    units = []
    for s, e in _line_spans(text, offset):
        line = text[s - offset : e - offset]
        if len(line) <= max_unit_chars:
            units.append((s, e))
            continue
        pos = 0
        for piece in _SENT_SPLIT_RE.split(line):
            i = line.find(piece, pos)
            if i < 0:
                continue
            pos = i + len(piece)
            if piece.strip():
                units.append((s + i, s + i + len(piece.rstrip())))
    return units


def partition(units, start, end):
    """Grow `units` into a CONTIGUOUS cover of [start, end) so no character is left unweighted."""
    if not units:
        return []
    out = []
    for i, (s, e) in enumerate(units):
        lo = start if i == 0 else out[-1][1]
        hi = end if i == len(units) - 1 else units[i + 1][0]
        hi = max(hi, e)
        out.append([lo, hi])
    return [(s, e) for s, e in out if e > s]


def content_tokens(text):
    """Lowercased word tokens with function words removed. Element ids and numbers are KEPT."""
    return [t for t in _WORD_RE.findall((text or "").lower()) if t not in STOPWORDS]


def build_idf(documents):
    """{token: idf} with smoothed inverse document frequency over `documents` (lists of strings)."""
    n = 0
    df = {}
    for doc in documents:
        n += 1
        for t in set(content_tokens(doc)):
            df[t] = df.get(t, 0) + 1
    idf = {}
    for t, d in df.items():
        idf[t] = math.log((n + 1.0) / (d + 1.0)) + 1.0
    return {"idf": idf, "n": n, "default": math.log(n + 1.0) + 1.0 if n else 1.0}


def idf_of(table, token):
    return table["idf"].get(token, table["default"])


def mass(table, tokens):
    return sum(idf_of(table, t) for t in set(tokens))


def containment(table, unit_tokens, cand_tokens):
    """IDF-weighted fraction of the unit's content that also occurs in the candidate. None if the"""
    uniq = set(unit_tokens)
    denom = mass(table, uniq)
    if denom <= 0:
        return None
    cand = set(cand_tokens)
    num = sum(idf_of(table, t) for t in uniq if t in cand)
    return num / denom


def candidate_windows(text, window=2, max_unit_chars=MAX_UNIT_CHARS):
    """[(window_text, token_set)] for every 1..`window`-unit sliding window of a resample's rationale."""
    spans = segment(text, 0, max_unit_chars)
    out = []
    for i in range(len(spans)):
        acc = []
        for j in range(i, min(i + window, len(spans))):
            acc = acc + content_tokens(text[spans[j][0] : spans[j][1]])
            out.append((text[spans[i][0] : spans[j][1]], set(acc)))
    return out


def candidate_token_sets(text, window=2, max_unit_chars=MAX_UNIT_CHARS):
    """Just the token sets of `candidate_windows` (what the scorer needs)."""
    return [t for _text, t in candidate_windows(text, window, max_unit_chars)]


def rationale_text(sections):
    """The resample's think+plan+memory as one string (the pool a claim may recur in)."""
    return "\n".join((sections.get(t) or "").strip() for t in SCORED_TAGS).strip()


def best_containment(table, unit_tokens, candidate_sets):
    """max containment of `unit_tokens` over the candidate windows; None if unscorable."""
    best = None
    for cand in candidate_sets:
        c = containment(table, unit_tokens, cand)
        if c is None:
            return None
        if best is None or c > best:
            best = c
    return best if best is not None else 0.0


def best_window(table, unit_tokens, windows):
    """(score, window_text) of the best-matching window -- the audit view of `best_containment`."""
    best = (None, "")
    for text, cand in windows:
        c = containment(table, unit_tokens, cand)
        if c is None:
            return (None, "")
        if best[0] is None or c > best[0]:
            best = (c, text)
    return best


def consistency(table, unit_tokens, sample_candidate_sets):
    """c = mean over resamples of the best window containment. None if the unit is unscorable."""
    if not sample_candidate_sets:
        return None
    vals = []
    for cands in sample_candidate_sets:
        c = best_containment(table, unit_tokens, cands)
        if c is None:
            return None
        vals.append(c)
    return sum(vals) / float(len(vals))


def adjust(c_same, c_null):
    """Excess of `c_same` over the same-episode null, renormalised onto [0, 1]."""
    if c_same is None:
        return None
    if c_null is None:
        c_null = 0.0
    denom = 1.0 - c_null
    if denom <= 1e-9:
        return 0.0
    x = (c_same - c_null) / denom
    if x < 0.0:
        return 0.0
    return 1.0 if x > 1.0 else x


def weight_of(c_adj, floor=DEFAULT_FLOOR):
    if c_adj is None:
        return 1.0
    w = floor + (1.0 - floor) * c_adj
    return 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)


def score_units(
    table,
    content,
    tag_spans,
    samples,
    null_samples,
    floor=DEFAULT_FLOOR,
    window=2,
    max_unit_chars=MAX_UNIT_CHARS,
):
    """Score every unit of one target's rationale sections."""
    cand_by_sample = [
        candidate_token_sets(rationale_text(s), window, max_unit_chars) for s in samples
    ]

    null_pools = [
        [candidate_token_sets(rationale_text(s), window, max_unit_chars) for s in grp]
        for grp in null_samples
    ]

    rows = []
    for tag in SCORED_TAGS:
        if tag not in tag_spans:
            continue
        lo, hi = tag_spans[tag]["inner"]
        inner = content[lo:hi]
        spans = partition(segment(inner, lo, max_unit_chars), lo, hi)
        scored = []
        for s, e in spans:
            toks = content_tokens(content[s:e])
            c_same = consistency(table, toks, cand_by_sample)
            c_null = None
            if c_same is not None and null_pools:
                vals = [consistency(table, toks, pool) for pool in null_pools]
                vals = [v for v in vals if v is not None]
                if vals:
                    c_null = sum(vals) / float(len(vals))
            c_adj = adjust(c_same, c_null) if c_same is not None else None
            row = {
                "tag": tag,
                "start": int(s),
                "end": int(e),
                "text": content[s:e],
                "c_same": c_same,
                "c_null": c_null,
                "c_adj": c_adj,
                "weight": weight_of(c_adj, floor) if c_adj is not None else None,
            }
            rows.append(row)
            scored.append(row)
        known = [r["weight"] for r in scored if r["weight"] is not None]
        fill = sum(known) / float(len(known)) if known else 1.0
        for r in scored:
            if r["weight"] is None:
                r["weight"] = fill
    return rows
