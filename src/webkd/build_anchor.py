"""Build role-specific masks for the frozen-student anchor."""

import argparse
import json
import os
import sys


EPS_UNANCHORED = 1e-6

ROLES = {
    "rationale": {"think": 1.0, "plan": 1.0, "memory": 1.0},
    "action": {"action": 1.0},
}


def assistant_turns(path):
    out = []
    for rec in json.load(open(path)):
        for m in rec.get("messages", rec.get("conversations", [])):
            role = m.get("role", m.get("from", ""))
            if role in ("assistant", "gpt"):
                out.append(m.get("content", m.get("value", "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--roles", default="rationale")
    args = ap.parse_args()

    from webkd import weights as W

    ds = os.path.splitext(os.path.basename(args.corpus))[0]
    contents = assistant_turns(args.corpus)
    print(f"corpus {ds}: {len(contents)} assistant turns")

    for role in args.roles.split(","):
        role = role.strip()
        if role not in ROLES:
            sys.exit(f"unknown role {role}")
        spans, covered, empty = {}, 0, 0
        for c in contents:
            sp = W.weight_spans_from_tags(c, ROLES[role])
            spans[c] = sp
            if sp:
                covered += 1
            else:
                empty += 1
        out = os.path.join(args.out_dir, f"{ds}.klgamma_{role}.json")
        W.write_sidecar(
            out,
            spans,
            dataset=ds,
            default_weight=EPS_UNANCHORED,
            floor=0.0,
            meta={
                "kl_role_mask": role,
                "tags": sorted(ROLES[role]),
                "built_by": os.path.basename(__file__),
            },
        )

        tot = sum(len(c) for c in contents)
        anc = sum(sum(end - start for start, end, _ in spans[c]) for c in contents)
        print(f"  {role:10s} -> {out}")
        print(
            f"      turns with >=1 span: {covered}/{len(contents)} (empty {empty})"
            f"   anchored chars: {anc}/{tot} = {100.0 * anc / max(1, tot):.1f}%"
        )


if __name__ == "__main__":
    main()
