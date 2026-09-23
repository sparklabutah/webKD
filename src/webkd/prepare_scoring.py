"""Prepare leave-one-segment-out scoring requests from a ShareGPT corpus."""

from __future__ import print_function

import argparse
import json
import os

from webkd import weights as W
from webkd import action_informativeness as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-seg-chars", type=int, default=A.MIN_SEG_CHARS)
    ap.add_argument("--max-segs", type=int, default=A.MAX_SEGS_PER_STEP)
    args = ap.parse_args()

    with open(args.dataset) as f:
        blob = json.load(f)

    records, n_segs, no_action = [], 0, 0
    for i, sample in enumerate(blob):
        system = sample.get("system", "")
        human, target = "", ""
        for turn in sample.get("conversations") or []:
            if turn.get("from") in ("human", "user"):
                human = turn.get("value", "")
            elif turn.get("from") in ("gpt", "assistant"):
                target = turn.get("value", "")
        spans = W.tag_spans(target)
        if "action" not in spans:
            no_action += 1
            continue
        segs = A.step_segments(
            target, W.tag_spans, min_chars=args.min_seg_chars, max_segs=args.max_segs
        )
        n_segs += len(segs)
        records.append(
            {
                "index": i,
                "sha1": W.content_key(target),
                "system": system,
                "user": human,
                "target": target,
                "action_inner": list(spans["action"]["inner"]),
                "segments": [{"tag": s["tag"], "start": s["start"], "end": s["end"]} for s in segs],
            }
        )

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(args.out, "w") as f:
        json.dump(records, f)

    print(
        "records=%d  segments=%d  requests=%d  (dropped %d without <action>)"
        % (len(records), n_segs, len(records) + n_segs, no_action)
    )
    print("-> " + args.out)


if __name__ == "__main__":
    main()
