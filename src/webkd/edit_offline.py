"""Localize failures and construct offline trajectory edits or self-corrections."""

from __future__ import print_function

import argparse
import glob
import json
import os
import re

from webkd import trajectory as T
from webkd import corrector as C

TEST_RAW = os.environ.get("WEBKD_TASK_CONFIG", "")

_TASKS = None


def task_config(task_id):
    """Return the raw task config (intent, additional_instructions, eval, ...) for a task_id."""
    global _TASKS
    if _TASKS is None:
        _TASKS = {}
        if os.path.isfile(TEST_RAW):
            for c in json.load(open(TEST_RAW)):
                _TASKS[c.get("task_id")] = c
    return _TASKS.get(task_id, {})


def strip_answer_line(instructions):
    """gold_noans: remove the trailing 'Send ... a message to the user: "X".' directive(s)."""
    if not instructions:
        return instructions
    out = re.sub(r"(?im)^\s*\d*\.?\s*(send|tell|provide).*?message.*?(user|:).*$", "", instructions)
    out = re.sub(r'(?is)send a message to the user:\s*["\'].*?["\']\.?', "", out)
    return out.strip()


def _numbered_lines(instructions):
    return re.findall(r"\d+\.\s+[^\n]+", strip_answer_line(instructions) or "")


def plan_prefix(task, frac):
    lines = _numbered_lines(task.get("additional_instructions", "") or "")
    k = max(1, int(round(frac * len(lines))))
    return "\n".join(lines[:k])


def plan_shuffle(task):
    """Deterministic permutation (reverse) of the numbered steps -- content kept, order broken."""
    return "\n".join(reversed(_numbered_lines(task.get("additional_instructions", "") or "")))


def wrong_plan(task):
    """Another v6 task's plan (deterministic: next task_id with instructions, answer stripped)."""
    tid = task.get("task_id")
    if _TASKS is None:
        task_config(0)
    ids = sorted(k for k in _TASKS if k and k >= 104)
    if tid in ids:
        i = ids.index(tid)
        other = ids[(i + 7) % len(ids)]
        return "\n".join(_numbered_lines(_TASKS[other].get("additional_instructions", "") or ""))
    return ""


PRIVILEGES = [
    "gold",
    "gold_noans",
    "none",
    "self",
    "gold_prefix25",
    "gold_prefix50",
    "plan_wrong",
    "plan_shuffle",
]

LOCALIZE_INSTR = (
    "\n\nThe agent's trajectory above FAILED. Identify the FIRST erroneous step -- the "
    "earliest step whose action set the trajectory on the path to failure. Reply with ONLY "
    "the integer step index (0-based). Nothing else."
)

REWRITE_INSTR = {
    "sparse": (
        "\n\nRewrite ONLY the single <action> for step {t}, correcting the error. "
        "Reply with exactly one <action>...</action> block."
    ),
    "local": (
        "\n\nRewrite step {t}: a corrected <think>...</think> and the corrected "
        "<action>...</action>. Reply with those two blocks only."
    ),
    "dense": (
        "\n\nRewrite the trajectory FROM step {t} to completion. For EACH step emit a "
        "<think>...</think>, optional <plan>/<memory>, and an <action>...</action>. The "
        "final action must submit the answer via send_msg_to_user('...'). Separate steps "
        "with a line containing only ---."
    ),
}

CORRECTOR_SYSTEM = (
    "You are an expert web-agent trainer. You review a failed browser-agent trajectory (goal, "
    "per-step accessibility-tree observations, and the agent's think/action) and produce a "
    "corrected continuation using the SAME <think>/<plan>/<memory>/<action> format and the same "
    "browsergym action API (fill/click/press/scroll/send_msg_to_user)."
)


def render_trajectory(episode, upto=None):
    """Render the episode as text for the corrector: goal + per-step obs/think/action."""
    lines = []
    for st in episode["steps"]:
        if upto is not None and st["step"] > upto:
            break
        sec = st["sections"]
        lines.append("=== step %d ===" % st["step"])
        ax = st["axtree"]
        lines.append(ax[:4000] + (" ...[truncated]" if len(ax) > 4000 else ""))
        if sec.get("think"):
            lines.append("<think> %s </think>" % sec["think"])
        lines.append("<action> %s </action>" % sec.get("action", ""))
    return "\n".join(lines)


def build_context(episode, task, privilege, upto=None):
    """Return (system, user) for the corrector; privilege is injected ONLY here."""
    goal = task.get("intent", "") or (episode["steps"][0]["goal"] if episode["steps"] else "")
    priv = ""
    if privilege == "gold":
        priv = "\n\n[REFERENCE PLAN]\n" + (task.get("additional_instructions", "") or "")
    elif privilege == "gold_noans":
        priv = "\n\n[REFERENCE PLAN (answer withheld)]\n" + strip_answer_line(
            task.get("additional_instructions", "") or ""
        )
    elif privilege == "self":
        plan0 = episode["steps"][0]["sections"].get("plan", "") if episode["steps"] else ""
        priv = ("\n\n[THE AGENT'S OWN INITIAL PLAN]\n" + plan0) if plan0 else ""
    elif privilege in ("gold_prefix25", "gold_prefix50"):
        priv = "\n\n[PARTIAL REFERENCE PLAN]\n" + plan_prefix(
            task, 0.25 if privilege.endswith("25") else 0.5
        )
    elif privilege == "plan_wrong":
        priv = "\n\n[REFERENCE PLAN]\n" + wrong_plan(task)
    elif privilege == "plan_shuffle":
        priv = "\n\n[REFERENCE PLAN]\n" + plan_shuffle(task)

    user = "## Goal:\n%s%s\n\n## Failed trajectory:\n%s" % (
        goal,
        priv,
        render_trajectory(episode, upto=upto),
    )
    return CORRECTOR_SYSTEM, user


def localize(episode, task, corrector, privilege):
    system, user = build_context(episode, task, privilege)
    out = corrector(system, user + LOCALIZE_INSTR, max_tokens=256)
    m = re.search(r"\d+", out or "")
    t = int(m.group(0)) if m else 0
    last = episode["steps"][-1]["step"] if episode["steps"] else 0
    return max(0, min(t, last))


def _split_rewritten_steps(text):
    """Parse the corrector's rewrite into a list of per-step section dicts."""
    chunks = re.split(r"\n\s*-{3,}\s*\n", text) if re.search(r"\n\s*-{3,}\s*\n", text) else [text]
    steps = []
    for ch in chunks:
        sec = T.parse_sections(ch)
        if any(sec.get(k) for k in ("think", "action", "plan", "memory")):
            steps.append(sec)
    return steps or [T.parse_sections(text)]


def edit(episode, t_star, task, corrector, privilege, locality):
    """Return the list of corrected section-dicts for steps t_star.. ."""
    system, user = build_context(episode, task, privilege, upto=t_star - 1 if t_star > 0 else None)
    instr = REWRITE_INSTR[locality].format(t=t_star)
    out = corrector(system, user + instr)
    new_steps = _split_rewritten_steps(out)
    if locality in ("sparse", "local"):
        new_steps = new_steps[:1]
        if locality == "sparse":
            orig = _section_at(episode, t_star)
            new_steps[0] = {
                "think": orig.get("think", ""),
                "plan": "",
                "step": "",
                "memory": "",
                "action": new_steps[0].get("action", ""),
            }
    return new_steps


def _section_at(episode, t):
    for st in episode["steps"]:
        if st["step"] == t:
            return st["sections"]
    return dict((k, "") for k in T.TAGS)


def _edit_distance_chars(a, b):
    """Cheap normalized char-level difference proxy (not Levenshtein) for the sidecar."""
    a, b = a or "", b or ""
    if not a and not b:
        return 0.0
    common = sum(1 for x, y in zip(a, b) if x == y)
    return 1.0 - common / float(max(len(a), len(b)))


def write_edited_episode(src_episode, out_dir, t_star, new_steps, locality, meta):
    """Write the edited episode (prefix copied, suffix rewritten) + properties sidecar."""
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    steps = src_episode["steps"]

    for st in steps:
        if st["step"] < t_star:
            T.copy_prompts(src_episode["dir"], out_dir, st["step"])
            T.write_agent_output(out_dir, st["step"], st["sections"])

    orig_before = "\n".join(
        _section_at(src_episode, s["step"]).get("action", "") for s in steps if s["step"] >= t_star
    )
    for i, sec in enumerate(new_steps):
        step_idx = t_star + i

        src_step = min(step_idx, steps[-1]["step"]) if steps else step_idx
        T.copy_prompts(src_episode["dir"], out_dir, src_step)
        T.write_agent_output(out_dir, step_idx, sec)
    n_steps = t_star + len(new_steps)

    submitted = new_steps[-1].get("action", "") if new_steps else ""
    with open(os.path.join(out_dir, "summary_info.json"), "w") as f:
        json.dump(
            {"n_steps": n_steps, "cum_reward": None, "err_msg": None, "edited": True}, f, indent=2
        )
    orig_after = "\n".join(s.get("action", "") for s in new_steps)
    props = dict(meta)
    props.update(
        {
            "t_star": t_star,
            "locality": locality,
            "n_new_steps": len(new_steps),
            "final_is_send_msg": T.is_send_msg(submitted),
            "edit_distance": round(_edit_distance_chars(orig_before, orig_after), 4),
            "validated": None,
        }
    )
    with open(os.path.join(out_dir, "properties.json"), "w") as f:
        json.dump(props, f, indent=2)
    return props


def iter_failed_episodes(root, limit=0):
    """Yield episode dirs (containing summary_info.json) with cum_reward < 1.0."""
    n = 0
    for summ in sorted(glob.glob(os.path.join(root, "**", "summary_info.json"), recursive=True)):
        d = os.path.dirname(summ)
        try:
            s = json.load(open(summ))
        except Exception:
            continue
        r = s.get("cum_reward", None)
        if r is not None and r < 1.0:
            yield d
            n += 1
            if limit and n >= limit:
                return


def iter_all_episodes(root, limit=0):
    """Yield EVERY judged episode dir, successes included (edit.py --mode on_policy_distillation)."""
    n = 0
    for summ in sorted(glob.glob(os.path.join(root, "**", "summary_info.json"), recursive=True)):
        d = os.path.dirname(summ)
        try:
            s = json.load(open(summ))
        except Exception:
            continue
        if s.get("cum_reward", None) is not None:
            yield d
            n += 1
            if limit and n >= limit:
                return


def main():
    ap = argparse.ArgumentParser(description="Edit failed TimeWarp trajectories (self-correction).")
    ap.add_argument("--episodes", required=True, help="root of failed-episode dirs")
    ap.add_argument("--out", required=True, help="output edited-episode tree")
    ap.add_argument("--corrector", default="teacher", choices=["self", "teacher"])
    ap.add_argument(
        "--localizer",
        default=None,
        choices=["self", "teacher"],
        help="E7: who localizes t* (default = --corrector)",
    )
    ap.add_argument(
        "--rewriter",
        default=None,
        choices=["self", "teacher"],
        help="E7: who rewrites the suffix (default = --corrector)",
    )
    ap.add_argument(
        "--oracle",
        action="store_true",
        help="E7: set t* from numbered-hint pseudo-gold instead of a localizer",
    )
    ap.add_argument("--privilege", default="none", choices=PRIVILEGES)
    ap.add_argument("--locality", default="dense", choices=["sparse", "local", "dense"])
    ap.add_argument("--version", default="6")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="parallel episodes (API calls)")
    ap.add_argument("--mock", action="store_true", help="use a scripted corrector (no LLM/network)")
    args = ap.parse_args()

    def build_corrector(role):
        if args.mock:
            return C.MockCorrector()
        return C.LiveCorrector(provider="local") if role == "self" else C.LiveCorrector()

    localizer = build_corrector(args.localizer or args.corrector)
    rewriter = build_corrector(args.rewriter or args.corrector)
    if args.oracle:
        from webkd import judge_reliability as J
    batch = os.path.join(args.out, "v%s" % args.version)

    def process(ep_dir):
        episode = T.load_episode(ep_dir)
        if episode is None or not episode["steps"]:
            return None
        task = task_config(episode["task_id"])
        if args.oracle:
            g = J.pseudo_gold_first_error(episode, task.get("additional_instructions", ""))
            t_star = g if g is not None else localize(episode, task, localizer, args.privilege)
        else:
            t_star = localize(episode, task, localizer, args.privilege)
        new_steps = edit(episode, t_star, task, rewriter, args.privilege, args.locality)
        out_dir = os.path.join(batch, os.path.basename(ep_dir))
        meta = {
            "corrector": args.corrector,
            "privilege": args.privilege,
            "localizer": args.localizer or args.corrector,
            "rewriter": args.rewriter or args.corrector,
            "oracle": args.oracle,
            "task_id": episode["task_id"],
            "src": ep_dir,
        }
        return write_edited_episode(episode, out_dir, t_star, new_steps, args.locality, meta)

    eps = list(iter_failed_episodes(args.episodes, args.limit))
    n = 0
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process, d): d for d in eps}
            for fut in as_completed(futs):
                try:
                    props = fut.result()
                except Exception as e:
                    print("  [skip] %s: %s" % (os.path.basename(futs[fut]), e))
                    continue
                if props is None:
                    continue
                n += 1
                print(
                    "  [%d] task %s t*=%d new=%d final_send=%s editdist=%.2f"
                    % (
                        n,
                        props["task_id"],
                        props["t_star"],
                        props["n_new_steps"],
                        props["final_is_send_msg"],
                        props["edit_distance"],
                    )
                )
    else:
        for d in eps:
            try:
                props = process(d)
            except Exception as e:
                print("  [skip] %s: %s" % (os.path.basename(d), e))
                continue
            if props is None:
                continue
            n += 1
            print(
                "  [%d] task %s t*=%d new=%d final_send=%s editdist=%.2f"
                % (
                    n,
                    props["task_id"],
                    props["t_star"],
                    props["n_new_steps"],
                    props["final_is_send_msg"],
                    props["edit_distance"],
                )
            )
    print("\nedited %d episodes -> %s" % (n, batch))


if __name__ == "__main__":
    main()
