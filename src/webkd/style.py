"""Mine student rationale exemplars and preserve teacher actions during restyling."""

from __future__ import print_function

import difflib
import glob
import json
import os
import random
import re

from webkd import trajectory as T

STYLE_TAGS = ("think", "plan", "step", "memory")

MAX_SECTION_CHARS = int(os.environ.get("WEBKD_STYLE_MAX_SECTION_CHARS", "1200"))

STYLE_HEADER = """

[STYLE -- write this step in the student agent's own voice]
The five sections you emit become training data for the agent whose steps appear above. Match ITS
writing style: same register, same typical length, same amount of detail, the same way of phrasing
a plan and of noting a memory. Below are verbatim examples of how THIS agent writes.

Copy the STYLE only. Do NOT copy their content, their decisions, or their conclusions -- they are
unrelated steps from other tasks. Your decision for this step must remain exactly the one you would
have made without these examples; only the wording changes."""

STYLE_EXAMPLE_OPEN = "\n--- how the student writes, example %d ---\n"
STYLE_EXAMPLE_CLOSE = "\n--- end example %d ---\n"

RESTYLE_HEADER = """

[REWRITE THIS STEP IN YOUR OWN WORDS]
A more experienced agent looked at the exact page shown above and wrote the following step. Its
decision is correct: keep the action, keep the element it targets, and keep every fact it relies on.

Rewrite it the way YOU would have written it -- your own phrasing, your own level of detail, your own
way of laying out a plan and a memory. Do not change the action. Do not add any claim about the page
that is not already there, and do not drop the reasons the step gives for what it does.

--- the step to rewrite ---
%s
--- end of the step to rewrite ---

Reply with ALL FIVE tags (<think>, <plan>, <step>, <memory>, <action>) and nothing else. The
<action> must be byte-identical to the one above."""


def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _step_num(path):
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def iter_episode_dirs(root, max_depth=3):
    """Yield every directory under `root` that holds a training_data/ dump."""
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _files in os.walk(root):
        if dirpath.rstrip(os.sep).count(os.sep) - base_depth > max_depth:
            dirnames[:] = []
            continue
        if "training_data" in dirnames:
            yield dirpath


def recover_sections(sec):
    """Un-pack the LEGACY dump format, where every tag lands inside the `think` field."""
    think = sec.get("think") or ""
    if not any(marker in think for marker in ("</think>", "<plan>", "<memory>")):
        return sec
    blob = think if think.lstrip().startswith("<think>") else ("<think>\n" + think)
    rec = T.parse_sections(blob)
    for k in T.TAGS:
        if not rec.get(k) and sec.get(k):
            rec[k] = sec[k]
    if sec.get("action"):
        rec["action"] = sec["action"]
    return rec


def episode_sections(ep_dir):
    """[(step, sections)] for one episode, reading ONLY the agent_output dumps."""
    td = os.path.join(ep_dir, "training_data")
    out = []
    for path in glob.glob(os.path.join(td, "agent_output_step_*.json")):
        n = _step_num(path)
        if n < 0:
            continue
        data = _load_json(path)
        if data is None:
            continue
        if "content" in data:
            sec = T.parse_sections(T.get_text(data.get("content", "")))
        else:
            sec = recover_sections(dict((k, str(data.get(k) or "").strip()) for k in T.TAGS))
        out.append((n, sec))
    out.sort()
    return out


REQUIRE_TAGS = tuple(
    t.strip()
    for t in os.environ.get("WEBKD_STYLE_REQUIRE_TAGS", "think,plan,memory").split(",")
    if t.strip()
)


def _usable_exemplar(step, sections):
    """A step is a usable style exemplar only if it shows the student's full voice."""
    if step <= 0:
        return False
    return all((sections.get(t) or "").strip() for t in REQUIRE_TAGS)


_BANK_CACHE = {}


def build_style_bank(root, prefer_solved=True, max_episodes=0):
    """Mine the student's own rollout for exemplars of how it writes. Cached per (root, args)."""
    key = (os.path.abspath(root), bool(prefer_solved), int(max_episodes))
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]

    solved_recs, other_recs = [], []
    for i, ep_dir in enumerate(sorted(iter_episode_dirs(root))):
        if max_episodes and i >= max_episodes:
            break
        summary = _load_json(os.path.join(ep_dir, "summary_info.json")) or {}
        reward = summary.get("cum_reward", None)
        solved = reward is not None and reward >= 1.0
        task_id = T.task_id_from_dir(ep_dir)
        if task_id is None:
            task_id = T.task_id_from_dir(os.path.dirname(ep_dir))
        for step, sec in episode_sections(ep_dir):
            if not _usable_exemplar(step, sec):
                continue
            rec = {
                "task_id": task_id,
                "step": step,
                "solved": solved,
                "sections": dict((k, sec.get(k) or "") for k in T.TAGS),
            }
            (solved_recs if solved else other_recs).append(rec)

    bank = solved_recs if (prefer_solved and solved_recs) else (solved_recs + other_recs)
    bank.sort(key=lambda r: (r["task_id"] if r["task_id"] is not None else -1, r["step"]))
    _BANK_CACHE[key] = bank
    return bank


def pick_exemplars(bank, task_id, k, exclude_task=True, seed_salt=0):
    """k exemplars for `task_id`: deterministic, at most one per source task, never the task itself."""
    if k <= 0 or not bank:
        return []
    pool = [r for r in bank if not (exclude_task and r["task_id"] == task_id)]
    if not pool:
        pool = list(bank)

    by_task = {}
    for r in pool:
        by_task.setdefault(r["task_id"], []).append(r)
    task_keys = sorted(by_task, key=lambda t: (t is None, t))
    rnd = random.Random(1000003 * int(task_id or 0) + int(seed_salt))
    rnd.shuffle(task_keys)

    out = []
    for t in task_keys[:k]:
        steps = by_task[t]
        out.append(steps[rnd.randrange(len(steps))])
    return out


def prefix_exemplars(episode, t_star, k):
    """Up to k of the student's OWN steps from this episode's replay prefix (steps < t*)."""
    if k <= 0 or not episode:
        return []
    out = []
    for st in reversed(episode.get("steps", [])):
        if st["step"] >= t_star:
            continue
        if not _usable_exemplar(st["step"], st["sections"]):
            continue
        out.append(
            {
                "task_id": episode.get("task_id"),
                "step": st["step"],
                "solved": False,
                "sections": dict(st["sections"]),
            }
        )
        if len(out) >= k:
            break
    return list(reversed(out))


def _truncate(text, limit=None):
    limit = MAX_SECTION_CHARS if limit is None else limit
    text = (text or "").strip()
    if limit and len(text) > limit:
        return text[:limit].rstrip() + " ...[truncated]"
    return text


def format_exemplar(sections):
    """Render one exemplar as the tag block the teacher should imitate (no <action>)."""
    parts = []
    for tag in STYLE_TAGS:
        val = _truncate(sections.get(tag, ""))
        if not val:
            continue
        if tag == "step":
            parts.append("<step>%s</step>" % val)
        else:
            parts.append("<%s>\n%s\n</%s>" % (tag, val, tag))
    return "\n\n".join(parts)


def style_block(exemplars):
    rendered = []
    for i, ex in enumerate(exemplars or []):
        body = format_exemplar(ex.get("sections", {}))
        if not body:
            continue
        n = len(rendered) + 1
        rendered.append(STYLE_EXAMPLE_OPEN % n + body + STYLE_EXAMPLE_CLOSE % n)
    if not rendered:
        return ""
    return STYLE_HEADER + "\n" + "".join(rendered)


def restyle_query(teacher_sections):
    return RESTYLE_HEADER % T.sections_to_content(teacher_sections)


def recover_open_think(text):
    """Restore an IMPLIED opening <think> in a model reply, so parse_sections can see it."""
    text = text or ""
    if "</think>" in text and "<think>" not in text:
        return "<think>\n" + text.lstrip()
    return text


def normalize_action(action):
    """Whitespace-insensitive form for comparing two actions (quoting/spacing varies)."""
    return re.sub(r"\s+", "", (action or "").strip())


def similarity(a, b):
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def merge_restyled(teacher_sections, restyled_sections, keep_teacher_action=True):
    """Combine the teacher's DECISION with the student's VOICE. Returns (sections, info)."""
    teacher_sections = dict(teacher_sections or {})
    restyled_sections = dict(restyled_sections or {})
    info = {
        "restyle_parsed": bool(restyled_sections.get("action") or restyled_sections.get("think")),
        "action_changed": False,
        "action_overridden": False,
        "filled_tags": [],
    }

    t_act = teacher_sections.get("action", "")
    r_act = restyled_sections.get("action", "")
    info["action_changed"] = bool(r_act) and normalize_action(r_act) != normalize_action(t_act)

    final = {}
    for tag in T.TAGS:
        val = (restyled_sections.get(tag) or "").strip()
        if not val:
            val = (teacher_sections.get(tag) or "").strip()
            if val:
                info["filled_tags"].append(tag)
        final[tag] = val

    if keep_teacher_action and t_act:
        if info["action_changed"]:
            info["action_overridden"] = True
        final["action"] = t_act

    info["sim_think"] = similarity(teacher_sections.get("think", ""), final.get("think", ""))
    info["sim_rationale"] = similarity(
        "\n".join(teacher_sections.get(t, "") for t in ("think", "plan", "memory")),
        "\n".join(final.get(t, "") for t in ("think", "plan", "memory")),
    )
    tl = len("".join(teacher_sections.get(t, "") for t in ("think", "plan", "memory")))
    fl = len("".join(final.get(t, "") for t in ("think", "plan", "memory")))
    info["len_ratio"] = (float(fl) / tl) if tl else 0.0
    return final, info


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the style-exemplar bank for a P-0 root.")
    ap.add_argument("root", help="P-0 episode root, e.g. data/episodes")
    ap.add_argument("--task-id", type=int, default=104)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--all-episodes", action="store_true", help="do not prefer solved episodes")
    a = ap.parse_args()
    bank = build_style_bank(a.root, prefer_solved=not a.all_episodes)
    tasks = sorted(set(r["task_id"] for r in bank))
    print(
        "bank: %d exemplars from %d tasks (solved-only=%s)"
        % (len(bank), len(tasks), not a.all_episodes)
    )
    picked = pick_exemplars(bank, a.task_id, a.k)
    print("picked for task %s: %s" % (a.task_id, [(p["task_id"], p["step"]) for p in picked]))
    print(style_block(picked))
