"""Stepwise expert supervision of a live student policy."""

from __future__ import print_function

import re

from webkd import trajectory as T

OK = "ok"
FIX = "fix"
FIX_UNUSABLE = "fix_unusable"
UNPARSED = "unparsed"

_VERDICT_RE = re.compile(r"<verdict>\s*(ok|fix)\b", re.IGNORECASE)

_VERDICT_LOOSE_RE = re.compile(r"\bverdict\b\s*[:\-]?\s*\**\s*(ok|fix)\b", re.IGNORECASE)

STUDENT_STEP_HEADER = (
    "\n\n============================================================\n"
    "[THE STUDENT'S PROPOSED STEP FOR THIS STATE -- your job is to judge it]\n"
    "============================================================\n"
)

JUDGE_INSTR = """
You are supervising this student web agent one step at a time, in the LIVE environment. Decide
whether the proposed step above may EXECUTE from the state shown in the observation.

Judge the ACTION, using the observation as the only source of truth:
  * does it make real progress toward the goal from THIS state?
  * does it target an element that actually exists in the AXTree above?
  * is it exactly one valid action, and is the reasoning consistent with what is visible (no
    invented page content, no answer asserted from facts that are not in the observation)?
  * if it submits a final answer with send_msg_to_user, is that answer actually correct and
    complete for the goal?

Reply in EXACTLY one of these two forms.

FORM 1 -- the step is acceptable:
<verdict>OK</verdict>
Use this whenever the step makes progress. It does NOT have to be the step you would have chosen:
do not correct a different-but-reasonable route, a slower-but-working route, wording, verbosity,
or plan formatting. If you are unsure, answer OK.

FORM 2 -- the step is wrong and you are replacing it:
<verdict>FIX</verdict>
followed by the corrected step for THIS state in the five-tag format specified below.
Use this ONLY when the step is genuinely wrong: it makes no progress (repeats an action that
already failed, loops, re-navigates to a page it just left), targets an element that is not in the
observation, is malformed, or submits a wrong or premature final answer. Your action will be
EXECUTED in the live browser and the student continues from the resulting state, so write an action
that is valid right now and that you would be willing to own.

*** WRITE THE REPLACEMENT AS THE AGENT, NOT AS A REVIEWER. ***
Do the judging silently. The five tags you emit become the agent's OWN record of this step and are
replayed in a world where no proposal ever existed, so they must read as the agent thinking for
itself in the first person: what is on the page, what it is doing next, why. Never mention the
proposal or its faults, and never use "the proposed", "the student", "instead of", "rather than",
"the previous suggestion" or any equivalent. Write the step, not a critique of a step.
"""

_JUDGE_VOICE_RE = re.compile(
    r"\b(the proposed|proposed action|proposed step|proposed memory|proposed plan|the student|"
    r"instead of the|rather than the proposed|the agent's proposal|this proposal|the suggestion)\b",
    re.IGNORECASE,
)


def has_judge_voice(sections):
    """True if a target's rationale talks ABOUT a proposal instead of taking the step."""
    sections = sections or {}
    text = " ".join((sections.get(t) or "") for t in ("think", "plan", "memory"))
    return bool(_JUDGE_VOICE_RE.search(text))


def judge_query(student_sections, section_contract="", student_broken=False):
    """The text appended to the student's OWN rendered prompt to turn the teacher into a judge."""
    body = T.sections_to_content(student_sections or {})
    if not body.strip():
        body = "(the student produced no parseable step)"
    parts = [STUDENT_STEP_HEADER, body, "\n", JUDGE_INSTR]
    if student_broken:
        parts.append(
            "\nNOTE: the student did not produce a usable action for this state, so there is nothing\n"
            "to accept. You MUST answer <verdict>FIX</verdict> and supply the corrected step.\n"
        )
    parts.append(section_contract or "")
    return "".join(parts)


def parse_verdict(text):
    """-> (verdict, sections, note). Three-valued and never guesses silently."""
    text = text or ""
    sections = T.parse_sections(text)
    has_action = bool((sections.get("action") or "").strip())
    m = _VERDICT_RE.search(text) or _VERDICT_LOOSE_RE.search(text)
    verdict = m.group(1).lower() if m else ""

    if verdict == "ok":
        return OK, {}, ("ok_with_step" if has_action else "")
    if verdict == "fix":
        if has_action:
            return FIX, sections, ""
        return FIX_UNUSABLE, sections, "fix_without_action"
    if has_action:
        return FIX, sections, "verdict_inferred_from_action"
    return UNPARSED, {}, "no_verdict_no_action"


def merge_fix(student_sections, fix_sections):
    """Build the step that EXECUTES and is DUMPED when the teacher intervenes -> (final, info)."""
    student_sections = dict(student_sections or {})
    fix_sections = dict(fix_sections or {})
    final, filled = {}, []
    for tag in T.TAGS:
        val = (fix_sections.get(tag) or "").strip()
        if not val:
            val = (student_sections.get(tag) or "").strip()
            if val:
                filled.append(tag)
        final[tag] = val
    info = {
        "filled_from_student": filled,
        "action_changed": normalize_action(final.get("action"))
        != normalize_action(student_sections.get("action")),
        "student_action": (student_sections.get("action") or "").strip(),
        "teacher_action": (final.get("action") or "").strip(),
        "sim_think": similarity(student_sections.get("think"), final.get("think")),
    }
    return final, info


def normalize_action(action):
    """Whitespace/quote-insensitive form, for deciding whether a fix changed the DECISION."""
    a = (action or "").strip()
    a = re.sub(r"\s+", " ", a)
    return a.replace('"', "'").rstrip(";").strip()


def similarity(a, b):
    ta = set((a or "").lower().split())
    tb = set((b or "").lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb)) / float(len(ta | tb))


def summarize(records):
    """Per-episode (or per-cell) diagnostics from edit_log.jsonl records -> dict."""
    steps = [r for r in (records or []) if r.get("phase") == "takeover"]
    n = len(steps)
    out = {
        "n_steps": n,
        "n_interventions": 0,
        "n_action_changed": 0,
        "n_inferred": 0,
        "n_unparsed": 0,
        "n_student_broken": 0,
        "n_fix_unusable": 0,
        "n_judge_voice": 0,
    }
    for r in steps:
        tk = r.get("takeover") or {}
        if tk.get("intervened"):
            out["n_interventions"] += 1
            if tk.get("action_changed"):
                out["n_action_changed"] += 1
            if tk.get("judge_voice"):
                out["n_judge_voice"] += 1
        if (tk.get("note") or "").startswith("verdict_inferred"):
            out["n_inferred"] += 1
        if tk.get("verdict") == UNPARSED:
            out["n_unparsed"] += 1
        if tk.get("student_broken"):
            out["n_student_broken"] += 1
        if tk.get("verdict") == FIX_UNUSABLE:
            out["n_fix_unusable"] += 1
    denom = float(n) if n else 1.0
    out["intervention_rate"] = out["n_interventions"] / denom
    out["action_change_rate"] = out["n_action_changed"] / denom
    out["inferred_verdict_rate"] = out["n_inferred"] / denom
    out["unparsed_rate"] = out["n_unparsed"] / denom

    out["judge_voice_rate"] = (
        out["n_judge_voice"] / float(out["n_interventions"]) if out["n_interventions"] else 0.0
    )
    return out
