"""Build WebKD action-informativeness weights from scored rationale segments."""

from __future__ import print_function

import argparse
import json
import math
import os

from webkd import weights as W
from webkd import action_informativeness as A


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def describe(xs):
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    mean = sum(s) / float(len(s))
    return {
        "n": len(s),
        "mean": mean,
        "p5": pct(s, 0.05),
        "p25": pct(s, 0.25),
        "median": pct(s, 0.50),
        "p75": pct(s, 0.75),
        "p95": pct(s, 0.95),
        "min": s[0],
        "max": s[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--examples", default=None)
    ap.add_argument("--tokenizer", default=None, help="if set, report TOKEN-weighted stats")
    ap.add_argument("--floor", type=float, default=A.DEFAULT_FLOOR)
    ap.add_argument("--tau", type=float, default=A.DEFAULT_TAU)
    ap.add_argument(
        "--transform",
        default="rank",
        choices=["rank", "minmax"],
        help="rank = the shipped within-step rank map; minmax = magnitude map (magnitude weighting)",
    )
    ap.add_argument(
        "--mask-negative",
        dest="mask_negative",
        action="store_true",
        help="weight 0 (token dropped from the loss) on negative-contribution segments (hard masking)",
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Explicitly allow uniform weights for failed scoring requests",
    )
    args = ap.parse_args()
    if not 0 <= args.floor <= 1 or not math.isfinite(args.tau) or args.tau <= 0:
        ap.error("floor must be in [0, 1] and tau must be finite and positive")

    convs = W.load_conversations(args.dataset)
    if not convs:
        ap.error("Dataset has no assistant targets")

    scores = {}
    with open(args.scores) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                scores[r["sha1"]] = r

    spans_by_content, extra = {}, {}
    masks_by_content = {}
    stat = {
        "scored": 0,
        "unscored": 0,
        "base_failed": 0,
        "seg_errors": 0,
        "segments": 0,
        "negative": 0,
        "deadzone_steps": 0,
    }
    all_w, w_by_tag, contribs_by_tag, spreads = [], {}, {}, []
    examples = []

    for c in convs:
        rec = scores.get(c["sha1"])
        if rec is None or rec.get("error") or rec.get("base_nll") is None:
            stat["unscored"] += 1
            if rec is not None and rec.get("error"):
                stat["base_failed"] += 1
            spans_by_content[c["content"]] = []
            continue

        segs, contribs = [], []
        errs = []
        for s in rec["segments"]:
            if "contribution" not in s:
                errs.append(s)
                continue
            segs.append(
                {
                    "tag": s["tag"],
                    "start": s["start"],
                    "end": s["end"],
                    "text": c["content"][s["start"] : s["end"]],
                }
            )
            contribution = float(s["contribution"])
            if not math.isfinite(contribution):
                ap.error("Non-finite action contribution")
            if not 0 <= s["start"] < s["end"] <= len(c["content"]):
                ap.error("A scored segment is outside its target")
            contribs.append(contribution)
        stat["seg_errors"] += len(errs)
        if not segs:
            stat["unscored"] += 1
            spans_by_content[c["content"]] = []
            continue

        stat["scored"] += 1
        stat["segments"] += len(segs)
        stat["negative"] += sum(1 for x in contribs if x < 0)

        spans, info = A.weight_spans_for_step(
            c["content"],
            segs,
            contribs,
            floor=args.floor,
            tau=args.tau,
            transform=args.transform,
            mask_negative=False,
        )
        if args.mask_negative:
            neg = [
                [int(sg["start"]), int(sg["end"])]
                for sg, cb in zip(segs, contribs)
                if cb < 0 and sg["end"] > sg["start"]
            ]
            if neg:
                masks_by_content[c["content"]] = neg
                stat["masked_spans"] = stat.get("masked_spans", 0) + len(neg)
        spreads.append(info["spread"])
        if info["shrink"] < 1.0:
            stat["deadzone_steps"] += 1

        seg_weights = [s[2] for s in spans]

        if errs and spans:
            mean_w = sum(seg_weights) / float(len(seg_weights))
            for s in errs:
                spans.append([int(s["start"]), int(s["end"]), float(mean_w)])
            spans.sort()

        spans_by_content[c["content"]] = spans
        extra[c["content"]] = {
            "base_nll": rec["base_nll"],
            "action_tokens": rec["action_tokens"],
            "spread": info["spread"],
            "shrink": info["shrink"],
        }

        for seg, cb, w in zip(segs, contribs, seg_weights):
            all_w.append(w)
            w_by_tag.setdefault(seg["tag"], []).append(w)
            contribs_by_tag.setdefault(seg["tag"], []).append(cb)
            examples.append(
                {
                    "sha1": c["sha1"],
                    "index": c["index"],
                    "tag": seg["tag"],
                    "contribution": cb,
                    "weight": w,
                    "spread": info["spread"],
                    "text": seg["text"].strip(),
                    "action": _action_of(c["content"]),
                }
            )

    if not args.allow_incomplete and (stat["unscored"] or stat["seg_errors"]):
        ap.error(
            "Scoring is incomplete; fix failed or missing requests before building WebKD weights"
        )

    entries = W.write_sidecar(
        args.out,
        spans_by_content,
        dataset=os.path.basename(args.dataset).replace(".json", ""),
        default_weight=1.0,
        floor=args.floor,
        envelope_weight=1.0,
        mask_spans_by_content=(masks_by_content or None),
        extra_by_content=extra,
        meta={
            "loss": "action_informativeness",
            "built_by": "webkd/build_weights.py",
            "scores": os.path.abspath(args.scores),
            "granularity": "sentence/clause inside <think>/<plan>/<memory>",
            "measure": "contribution(s) = logP(action_call | full) - logP(action_call | rationale minus s)",
            "transform": (
                "within-step average-rank -> [floor, 1], dead-zone shrink below tau nats"
                if args.transform == "rank"
                else "within-step min-max magnitude -> [floor, 1], dead-zone shrink below tau nats"
            ),
            "transform_kind": args.transform,
            "mask_negative": bool(args.mask_negative),
            "tau_nats": args.tau,
            "weight_1_by_construction": [
                "step",
                "action",
                "section tags",
                "separators",
                "envelope",
            ],
            "negatives": "ranked, never masked; they land at/near the floor",
        },
    )
    W.validate(entries, convs)

    report = {
        "out": os.path.abspath(args.out),
        "floor": args.floor,
        "tau": args.tau,
        "transform": args.transform,
        "mask_negative": bool(args.mask_negative),
        "counts": stat,
        "weights_all": describe(all_w),
        "weights_by_tag": dict((k, describe(v)) for k, v in w_by_tag.items()),
        "contributions_by_tag": dict((k, describe(v)) for k, v in contribs_by_tag.items()),
        "within_step_spread_nats": describe(spreads),
        "sidecar": W.summarize(entries, convs),
    }

    if args.tokenizer:
        report["token_weighted"] = _token_report(args, convs, entries)

    print(json.dumps(report, indent=2))
    if args.report:
        d = os.path.dirname(os.path.abspath(args.report))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
    if args.examples:
        examples.sort(key=lambda e: -e["contribution"])
        with open(args.examples, "w") as f:
            json.dump(examples, f, indent=2)
        print("examples -> " + args.examples)


def _action_of(content):
    sp = W.tag_spans(content).get("action")
    return content[sp["inner"][0] : sp["inner"][1]] if sp else ""


def _token_report(args, convs, entries):
    """What the OPTIMIZER sees: per-token weights through the production span->token path."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    by_tag, allw, per_step_mean = {}, [], []
    for c in convs:
        entry = entries[c["sha1"]]
        enc = tok(c["content"], add_special_tokens=False, return_offsets_mapping=True)
        offs = list(enc["offset_mapping"])
        ws = W.token_weight_vector(
            offs,
            W.flatten_weight_spans(entry.get("weight_spans") or []),
            float(entry.get("default_weight", 1.0)),
            args.floor,
            entry.get("mask_spans") or [],
        )
        tags = W.tag_spans(c["content"])
        for (s, e), w in zip(offs, ws):
            if e <= s:
                continue
            allw.append(w)
            placed = "structure"
            for t, sp in tags.items():
                a, b = sp["inner"]
                if s >= a and s < b:
                    placed = t
                    break
            by_tag.setdefault(placed, []).append(w)
        if ws:
            per_step_mean.append(sum(ws) / float(len(ws)))
    return {
        "all_tokens": describe(allw),
        "by_tag": dict((k, describe(v)) for k, v in by_tag.items()),
        "per_step_mean_weight": describe(per_step_mean),
    }


if __name__ == "__main__":
    main()
