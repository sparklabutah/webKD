"""Create a copy of edited trajectories containing only state-consistent steps."""

from __future__ import print_function

import argparse
import json
import os
import shutil
import sys

from webkd import trajectory as T
from webkd import style as S
from webkd import grounding as F


def episode_t_star(ep_dir):
    """t* for a live-edit episode: properties.json sits on the TASK dir, one level above."""
    for cand in (
        os.path.join(ep_dir, "properties.json"),
        os.path.join(os.path.dirname(ep_dir), "properties.json"),
    ):
        if os.path.isfile(cand):
            try:
                return int(json.load(open(cand)).get("t_star") or 0)
            except Exception:
                return 0
    return 0


def score_steps(ep_dir, thresh):
    """[(step, kept, grounded_fraction, n_claims)] for one episode; replay steps are auto-kept."""
    ep = T.load_episode(ep_dir)
    if ep is None or not ep["steps"]:
        return []
    t_star = episode_t_star(ep_dir)
    goal = ep["steps"][0]["goal"]
    out = []
    for st in ep["steps"]:
        if st["step"] < t_star:
            out.append((st["step"], True, 1.0, 0, "replay"))
            continue
        reasoning = " ".join([st["sections"].get("think", ""), st["sections"].get("memory", "")])
        if not reasoning.strip():
            out.append((st["step"], True, 1.0, 0, "no-rationale"))
            continue
        corpus = T.axtree_upto(ep, st["step"]) + "\n" + goal
        res = F.scf_score(reasoning, corpus, thresh)
        out.append(
            (
                st["step"],
                bool(res["state_consistent"]),
                res["grounded_fraction"],
                res["n_claims"],
                "edited",
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Materialise the SCF-filtered copy of a live-edit cell."
    )
    ap.add_argument("--edits", required=True, help="source cell, e.g. .../style_matched_none/v6")
    ap.add_argument(
        "--out", required=True, help="destination cell, e.g. .../style_filtered_none/v6"
    )
    ap.add_argument("--thresh", type=float, default=F.THRESH_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.dry_run:
        raise SystemExit("Output already exists; choose a new directory")
    eps = sorted(S.iter_episode_dirs(a.edits))
    if not eps:
        print("FATAL: no episodes under %s" % a.edits)
        sys.exit(1)

    tot = kept = dropped = 0
    edited_tot = edited_kept = 0
    per_ep = []
    for ep_dir in eps:
        scored = score_steps(ep_dir, a.thresh)
        if not scored:
            continue
        keep_steps = set(s for s, k, _f, _n, _w in scored if k)
        drop_steps = [s for s, k, _f, _n, _w in scored if not k]
        e_tot = sum(1 for _s, _k, _f, _n, w in scored if w == "edited")
        e_kept = sum(1 for _s, k, _f, _n, w in scored if w == "edited" and k)
        tot += len(scored)
        kept += len(keep_steps)
        dropped += len(drop_steps)
        edited_tot += e_tot
        edited_kept += e_kept
        per_ep.append((os.path.basename(os.path.dirname(ep_dir)), e_kept, e_tot, drop_steps))

        if a.dry_run:
            continue

        rel = os.path.relpath(ep_dir, a.edits)
        dst = os.path.join(a.out, rel)
        shutil.copytree(ep_dir, dst)
        src_props = os.path.join(os.path.dirname(ep_dir), "properties.json")
        if os.path.isfile(src_props):
            props = json.load(open(src_props))
            props["scf_filtered"] = {
                "thresh": a.thresh,
                "kept_edited": e_kept,
                "total_edited": e_tot,
                "dropped_steps": drop_steps,
            }
            with open(os.path.join(os.path.dirname(dst), "properties.json"), "w") as f:
                json.dump(props, f, indent=2)
        td = os.path.join(dst, "training_data")
        for s in drop_steps:
            for pat in (
                "agent_output_step_%d.json",
                "agent_output_step_%d.txt",
                "user_prompt_step_%d.json",
                "user_prompt_step_%d.txt",
                "system_prompt_step_%d.json",
                "system_prompt_step_%d.txt",
            ):
                p = os.path.join(td, pat % s)
                if os.path.isfile(p):
                    os.remove(p)

    print("SCF-filter %s -> %s (thresh=%.2f)" % (a.edits, a.out, a.thresh))
    print("  episodes            : %d" % len(per_ep))
    print(
        "  EDITED steps        : %d kept / %d  (rejected %d, %.1f%%)"
        % (
            edited_kept,
            edited_tot,
            edited_tot - edited_kept,
            100.0 * (edited_tot - edited_kept) / edited_tot if edited_tot else 0.0,
        )
    )
    print("  all steps in corpus : %d kept / %d" % (kept, tot))
    if edited_tot and edited_kept == edited_tot:
        print(
            "  NOTE: SCF rejected NOTHING -- this filtered cell is identical to its parent, so a "
            "difference between them cannot be attributed to filtering. Lower --thresh to make "
            "the arm meaningful, or report it as a null manipulation."
        )
    for name, ek, et, ds in per_ep[:8]:
        print("    %-46s edited %d/%d kept, dropped steps %s" % (name[:46], ek, et, ds[:6]))


if __name__ == "__main__":
    main()
