"""Check rationale claims against the observations available at each step."""

from __future__ import print_function

import argparse
import glob
import json
import os
import re

from webkd import trajectory as T

THRESH_DEFAULT = 0.8

_STOP = set(
    "the a an of to in on and or is are was were be been for with as at by from this that "
    "it its into will can may click type page search box section list user message note "
    "step steps action think plan memory i you we they he she".split()
)


def extract_claims(text):
    """Candidate grounded claims: quoted strings, numbers (incl. commas), bids, capitalized entities."""
    claims = []
    claims += re.findall(r'"([^"]{2,})"', text)
    claims += re.findall(r"'([^']{2,})'", text)
    claims += re.findall(r"\b\d[\d,\.]*\b", text)
    claims += [
        m for m in re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,})*", text)
    ]

    seen, out = set(), []
    for c in claims:
        c = c.strip()
        low = c.lower()
        if not c or low in seen:
            continue
        if all(w in _STOP for w in low.split()):
            continue
        seen.add(low)
        out.append(c)
    return out


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def is_grounded(claim, corpus_norm):
    c = _norm(claim)
    if not c:
        return True
    if c in corpus_norm:
        return True

    if re.match(r"^[\d,\.]+$", c):
        return c.replace(",", "") in corpus_norm.replace(",", "")

    toks = [t for t in re.findall(r"[a-z0-9]+", c) if t not in _STOP and len(t) > 2]
    if toks and all(t in corpus_norm for t in toks):
        return True
    return False


def scf_score(edited_text, grounding_text, thresh=THRESH_DEFAULT):
    """Return {state_consistent, grounded_fraction, n_claims, ungrounded}."""
    claims = extract_claims(edited_text)
    if not claims:
        return {"state_consistent": True, "grounded_fraction": 1.0, "n_claims": 0, "ungrounded": []}
    corpus = _norm(grounding_text)
    grounded = [c for c in claims if is_grounded(c, corpus)]
    frac = len(grounded) / float(len(claims))
    ungrounded = [c for c in claims if c not in grounded]
    return {
        "state_consistent": frac >= thresh,
        "grounded_fraction": round(frac, 4),
        "n_claims": len(claims),
        "ungrounded": ungrounded[:20],
    }


def score_episode(edited_episode_dir, orig_episode_dir=None, thresh=THRESH_DEFAULT):
    ep = T.load_episode(edited_episode_dir)
    if ep is None:
        return None
    props_path = os.path.join(edited_episode_dir, "properties.json")
    props = {}
    if os.path.isfile(props_path):
        props = json.load(open(props_path))
    t_star = props.get("t_star", 0)
    goal = ep["steps"][0]["goal"] if ep["steps"] else ""
    per_step, fracs = {}, []
    consistent = True
    for st in ep["steps"]:
        if st["step"] < t_star:
            continue
        reasoning = " ".join([st["sections"].get("think", ""), st["sections"].get("memory", "")])
        if not reasoning.strip():
            continue
        corpus = T.axtree_upto(ep, st["step"]) + "\n" + goal
        res = scf_score(reasoning, corpus, thresh)
        per_step[str(st["step"])] = res
        fracs.append(res["grounded_fraction"])
        consistent = consistent and res["state_consistent"]
    props["scf"] = {
        "state_consistent": consistent,
        "grounded_fraction": round(sum(fracs) / len(fracs), 4) if fracs else 1.0,
        "per_step": per_step,
    }
    with open(props_path, "w") as f:
        json.dump(props, f, indent=2)
    return props["scf"]


def main():
    ap = argparse.ArgumentParser(description="State-Consistency Filtering of edited trajectories.")
    ap.add_argument("--edits", required=True, help="edited-episode tree (edit_trajectory --out)")
    ap.add_argument("--thresh", type=float, default=THRESH_DEFAULT)
    ap.add_argument(
        "--drop",
        action="store_true",
        help="move state-inconsistent episodes to <edits>_scf_rejected/",
    )
    args = ap.parse_args()

    eps = [
        os.path.dirname(p)
        for p in glob.glob(os.path.join(args.edits, "**", "properties.json"), recursive=True)
    ]
    kept = dropped = 0
    frac_sum = 0.0
    for ep in sorted(eps):
        res = score_episode(ep, thresh=args.thresh)
        if res is None:
            continue
        frac_sum += res["grounded_fraction"]
        if res["state_consistent"]:
            kept += 1
        else:
            dropped += 1
            if args.drop:
                rej = ep.replace(
                    os.path.normpath(args.edits), os.path.normpath(args.edits) + "_scf_rejected"
                )
                if not os.path.isdir(os.path.dirname(rej)):
                    os.makedirs(os.path.dirname(rej))
                os.rename(ep, rej)
    n = kept + dropped
    print(
        "SCF over %d edited episodes: kept=%d dropped=%d  reject_rate=%.3f  mean_grounded=%.3f"
        % (n, kept, dropped, dropped / float(n) if n else 0.0, frac_sum / n if n else 1.0)
    )


if __name__ == "__main__":
    main()
