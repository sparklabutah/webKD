"""Utilities for evaluating judgment and error-localization agreement."""

from __future__ import print_function

import argparse
import glob
import json
import os
import re

from webkd import trajectory as T
from webkd import corrector as C
from webkd import edit_offline as E


def plan_steps(additional_instructions):
    """Split a numbered additional_instructions walkthrough into its ordered steps."""
    if not additional_instructions:
        return []
    text = re.sub(
        r"(?is)^\s*perform these instructions[^\n:]*:?", "", additional_instructions.strip()
    )

    parts = re.split(r"\s*\b\d+\.\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _overlap(a, b):
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(ta & tb) / float(len(ta) + 1)


def pseudo_gold_first_error(episode, additional_instructions):
    """Align numbered plan steps to trajectory steps by token overlap; the pseudo-gold first"""
    steps = plan_steps(additional_instructions)
    if not steps or not episode["steps"]:
        return None
    mapped = []
    for st in episode["steps"]:
        txt = st["sections"].get("think", "") + " " + st["sections"].get("action", "")
        scores = [_overlap(txt, ps) for ps in steps]
        mapped.append(int(max(range(len(steps)), key=lambda i: scores[i])))
    best = -1
    for i, m in enumerate(mapped):
        if m < best:
            return episode["steps"][i]["step"]
        best = max(best, m)

    if len(episode["steps"]) >= 4 and max(mapped) == 0:
        return episode["steps"][1]["step"] if len(episode["steps"]) > 1 else 0
    return None


def corrector_verdict(episode, task, corrector, privilege="none"):
    """Ask the corrector whether the trajectory succeeded (1) or failed (0)."""
    system, user = E.build_context(episode, task, privilege)
    out = corrector(
        system,
        user + "\n\nDid the agent SUCCEED at the goal? Answer only YES or NO.",
        max_tokens=256,
    )
    return 1 if re.search(r"\byes\b", (out or "").lower()) else 0


def main():
    ap = argparse.ArgumentParser(description="Judge + localization reliability table.")
    ap.add_argument("--episodes", required=True, help="root of (failed) episode dirs")
    ap.add_argument("--corrector", default="teacher", choices=["self", "teacher"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="artifacts/judge_reliability.json")
    ap.add_argument(
        "--localization-only",
        action="store_true",
        help="skip the env judge; only measure localization vs pseudo-gold",
    )
    args = ap.parse_args()

    if args.mock:
        C.set_corrector(C.MockCorrector())
    elif args.corrector == "self":
        C.set_corrector(C.LiveCorrector(provider="local"))
    else:
        C.set_corrector(C.LiveCorrector())
    corrector = C.get_corrector()

    rows = []
    loc_have = loc_hit = 0
    for summ in sorted(
        glob.glob(os.path.join(args.episodes, "**", "summary_info.json"), recursive=True)
    )[: args.limit]:
        ep_dir = os.path.dirname(summ)
        ep = T.load_episode(ep_dir)
        if ep is None or not ep["steps"]:
            continue
        task = E.task_config(ep["task_id"])
        gold_t = pseudo_gold_first_error(ep, task.get("additional_instructions", ""))
        pred_t = None
        try:
            pred_t = E.localize(ep, task, corrector, "none")
        except Exception as e:
            print("  [loc skip] %s: %s" % (os.path.basename(ep_dir), e))
        if gold_t is not None and pred_t is not None:
            loc_have += 1
            if abs(pred_t - gold_t) <= 1:
                loc_hit += 1
        rows.append(
            {
                "task_id": ep["task_id"],
                "env_reward": ep["reward"],
                "gold_t": gold_t,
                "pred_t": pred_t,
            }
        )

    loc_acc = loc_hit / float(loc_have) if loc_have else None
    summary = {
        "n": len(rows),
        "localization_n": loc_have,
        "localization_accuracy": loc_acc,
        "corrector": args.corrector,
    }
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)
    print("\n== judge/localization reliability (corrector=%s) ==" % args.corrector)
    print(
        "  episodes=%d  localization_n=%d  localization_acc=%s"
        % (len(rows), loc_have, ("%.3f" % loc_acc) if loc_acc is not None else "n/a")
    )
    if loc_acc is not None and loc_acc < 0.40:
        print(
            "  ** GATE (idea 10.1): localization < 0.40 -> add the verifier-hybrid corrector cell."
        )
    print("  wrote", args.out)


if __name__ == "__main__":
    main()
