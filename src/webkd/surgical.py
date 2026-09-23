"""Identify unchanged response sections for the surgical masking baseline."""

from __future__ import print_function

import difflib
import re

TAG_ORDER = ("think", "plan", "step", "memory", "action")

STRICT_PATTERNS = {
    "think": r"<think>\s*\n(.*?)\n\s*</think>",
    "plan": r"<plan>\s*\n(.*?)\n\s*</plan>",
    "step": r"<step>(.*?)</step>",
    "memory": r"<memory>\s*\n(.*?)\n\s*</memory>",
    "action": r"<action>\s*\n(.*?)\n\s*</action>",
}


def parse_edit_sections(content):
    """Strict tag parse, identical to convert.parse_content_sections."""
    sections = dict((k, "") for k in TAG_ORDER)
    for name, pattern in STRICT_PATTERNS.items():
        m = re.search(pattern, content, re.DOTALL)
        if m:
            sections[name] = m.group(1).strip()
    return sections


def assemble_target(sections, include_think=True, include_plan=True, include_memory=True):
    """Render the assistant target exactly as convert does, plus per-tag char spans."""
    parts = []
    if include_think and sections.get("think"):
        parts.append(("think", "<think>\n%s\n</think>" % sections["think"], len("<think>\n")))
    if include_plan:
        if sections.get("plan"):
            parts.append(("plan", "<plan>\n%s\n</plan>" % sections["plan"], len("<plan>\n")))
        if sections.get("step"):
            parts.append(("step", "<step>%s</step>" % sections["step"], len("<step>")))
    if include_memory and sections.get("memory"):
        parts.append(("memory", "<memory>\n%s\n</memory>" % sections["memory"], len("<memory>\n")))
    if sections.get("action"):
        parts.append(("action", "<action>\n%s\n</action>" % sections["action"], len("<action>\n")))

    value = ""
    spans = {}
    for i, (tag, text, inner_off) in enumerate(parts):
        if i:
            value += "\n\n"
        start = len(value)
        value += text
        inner_start = start + inner_off

        inner_end = inner_start + len(sections[tag])
        spans[tag] = {"outer": (start, start + len(text)), "inner": (inner_start, inner_end)}
    return value, spans


def normalize(text):
    """Whitespace-insensitive comparison form. Two sections that differ only in wrapping are the"""
    return re.sub(r"\s+", " ", (text or "").strip())


def section_similarity(a, b):
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def changed_tags(orig_sections, edit_sections, present_tags, sim_threshold=1.0):
    """Tags the TEACHER ACTUALLY CHANGED, restricted to tags present in the target."""
    changed = []
    for tag in TAG_ORDER:
        if tag not in present_tags:
            continue
        a = orig_sections.get(tag) if orig_sections else None
        b = edit_sections.get(tag)
        if orig_sections is None:
            changed.append(tag)
        elif sim_threshold >= 1.0:
            if normalize(a) != normalize(b):
                changed.append(tag)
        elif section_similarity(a, b) < sim_threshold:
            changed.append(tag)
    return changed


def mask_spans(spans, masked_tags, policy="section"):
    """Character spans to give ZERO loss, sorted and merged. policy: "section" | "inner"."""
    key = "inner" if policy == "inner" else "outer"
    out = []
    for tag in masked_tags:
        if tag in spans:
            s, e = spans[tag][key]
            if e > s:
                out.append([s, e])
    out.sort()
    merged = []
    for s, e in out:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def token_keep_mask(offsets, spans):
    """[bool] per token: True = keep the loss on this token."""
    keep = []
    for s, e in offsets:
        drop = False
        if e > s:
            for a, b in spans:
                if s >= a and e <= b:
                    drop = True
                    break
        keep.append(not drop)
    return keep


def apply_keep_mask(labels, keep, ignore_index=-100):
    """New label list with `ignore_index` wherever keep[i] is False. Never mutates in place."""
    out = list(labels)
    n = min(len(out), len(keep))
    for i in range(n):
        if not keep[i]:
            out[i] = ignore_index
    return out
