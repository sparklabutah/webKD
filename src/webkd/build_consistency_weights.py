"""Build the teacher consistency weighting baseline."""

from __future__ import print_function

import argparse
import collections
import json
import os

from webkd import consistency as CS
from webkd import weights as W

REPO = os.getcwd()
SCORED_TAGS = CS.SCORED_TAGS


def load_samples(path, k):
    """{(task_id, step): [sections, ...]} -- the first k good resamples of each step."""
    by_step = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error") or not rec.get("sections"):
                continue
            by_step.setdefault((rec["task_id"], rec["step"]), {})[rec["sample"]] = rec["sections"]
    out = {}
    for key, d in by_step.items():
        out[key] = [d[i] for i in sorted(d) if i < k]
    return out


def null_indices(i, n_total, tasks, n_null):
    """`n_null` deterministic OTHER-TASK step indices for step `i` (co-prime strides, no RNG)."""
    picked = []
    for stride in (97, 193, 271, 59, 149, 317):
        j = i
        for _ in range(n_total):
            j = (j + stride) % n_total
            if tasks[j] != tasks[i] and j not in picked:
                picked.append(j)
                break
        if len(picked) >= n_null:
            break
    return picked[:n_null]


def within_episode_indices(i, tasks, n_null):
    """Up to `n_null` other steps of the SAME task (diagnostic null)."""
    same = [j for j, t in enumerate(tasks) if t == tasks[i] and j != i]
    if not same:
        return []
    stride = max(1, len(same) // n_null)
    return same[::stride][:n_null]


def build(args):
    corpus = os.path.join(REPO, args.corpus)
    stats = corpus.replace(".json", ".stats.json")
    convs = W.load_conversations(corpus)
    prov = json.load(open(stats))["samples"]
    if len(convs) != len(prov):
        raise SystemExit("corpus/stats length mismatch: %d vs %d" % (len(convs), len(prov)))
    for c, p in zip(convs, prov):
        if c["sha1"] != p["sha1"]:
            raise SystemExit("corpus/stats are out of sync at sha1 " + c["sha1"])

    samples = load_samples(os.path.join(REPO, args.samples), args.k)
    tasks = [p["task_id"] for p in prov]
    keys = [(p["task_id"], p["step"]) for p in prov]

    docs = []
    tagspans = []
    for c in convs:
        ts = W.tag_spans(c["content"])
        tagspans.append(ts)
        for tag in SCORED_TAGS:
            if tag not in ts:
                continue
            lo, hi = ts[tag]["inner"]
            for s, e in CS.segment(c["content"][lo:hi], lo):
                docs.append(c["content"][s:e])
    table = CS.build_idf(docs)

    spans_by_content = {}
    audit = []
    n_no_samples = 0
    for i, c in enumerate(convs):
        mine = samples.get(keys[i]) or []
        if len(mine) < args.min_samples:
            n_no_samples += 1
            audit.append(
                {
                    "index": i,
                    "task_id": tasks[i],
                    "step": prov[i]["step"],
                    "sha1": c["sha1"],
                    "n_samples": len(mine),
                    "units": [],
                }
            )
            continue
        nulls = [
            samples.get(keys[j]) or [] for j in null_indices(i, len(convs), tasks, args.n_null)
        ]
        nulls = [g for g in nulls if g]
        rows = CS.score_units(table, c["content"], tagspans[i], mine, nulls, floor=args.floor)

        if args.within_diag:
            wgrp = [
                samples.get(keys[j]) or [] for j in within_episode_indices(i, tasks, args.n_null)
            ]
            wgrp = [g for g in wgrp if g]
            wrows = CS.score_units(table, c["content"], tagspans[i], mine, wgrp, floor=args.floor)
            for r, wr in zip(rows, wrows):
                r["c_null_within"] = wr["c_null"]
                r["c_adj_within"] = wr["c_adj"]

        spans_by_content[c["content"]] = [[r["start"], r["end"], r["weight"]] for r in rows]
        audit.append(
            {
                "index": i,
                "task_id": tasks[i],
                "step": prov[i]["step"],
                "sha1": c["sha1"],
                "n_samples": len(mine),
                "n_nulls": len(nulls),
                "changed": prov[i].get("changed"),
                "unchanged": prov[i].get("unchanged"),
                "units": rows,
            }
        )
    return convs, prov, spans_by_content, audit, table, n_no_samples


def token_report(convs, entries, tokenizer_path, floor):
    """Per-tag TOKEN-weighted distribution, computed through the production weight code path."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    per_tag = collections.defaultdict(list)
    for c in convs:
        entry = entries.get(c["sha1"])
        if entry is None:
            continue
        content = c["content"]
        enc = tok(content, add_special_tokens=False, return_offsets_mapping=True)
        offsets = list(enc["offset_mapping"])
        weights = W.token_weight_vector(
            offsets,
            W.flatten_weight_spans(entry.get("weight_spans") or []),
            float(entry.get("default_weight", 1.0)),
            floor,
            entry.get("mask_spans") or [],
        )
        ts = W.tag_spans(content)
        inner = dict((t, ts[t]["inner"]) for t in ts)
        for (s, e), w in zip(offsets, weights):
            if e <= s:
                continue
            tag = "envelope"
            for t, (lo, hi) in inner.items():
                if s >= lo and e <= hi:
                    tag = t
                    break
            per_tag[tag].append(float(w))
            per_tag["ALL"].append(float(w))
    return per_tag


def dist(vals):
    if not vals:
        return {}
    v = sorted(vals)
    n = len(v)

    def q(p):
        return v[min(n - 1, int(p * n))]

    return {
        "tokens": n,
        "mean": sum(v) / float(n),
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "min": v[0],
        "max": v[-1],
        "frac_below_0.9": sum(1 for x in v if x < 0.9) / float(n),
        "frac_below_0.5": sum(1 for x in v if x < 0.5) / float(n),
        "frac_at_floor": sum(1 for x in v if x < 0.1 + 1e-6) / float(n),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", default="data/corpus.json")
    ap.add_argument("--samples", default="artifacts/consistency/samples.jsonl")
    ap.add_argument("--out", default="artifacts/consistency.weights.json")
    ap.add_argument("--audit", default="artifacts/consistency/units.json")
    ap.add_argument("--report-out", default="artifacts/consistency/report.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="fewer good resamples than this -> leave the step uniform (no evidence)",
    )
    ap.add_argument("--n-null", type=int, default=3)
    ap.add_argument("--floor", type=float, default=CS.DEFAULT_FLOOR)
    ap.add_argument("--within-diag", action="store_true", default=True)
    ap.add_argument("--no-within-diag", dest="within_diag", action="store_false")
    ap.add_argument(
        "--surgical-merge",
        default="",
        help="also write this path with the surgical mask_spans merged in "
        "(e.g. .../corpus.surg.weights.json)",
    )
    ap.add_argument("--tokenizer", default="models/student")
    ap.add_argument(
        "--report", action="store_true", help="tokenize and print the weight distribution"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    convs, prov, spans_by_content, audit, table, n_no_samples = build(args)
    print(
        "corpus %d conversations; %d had too few resamples (left uniform)"
        % (len(convs), n_no_samples)
    )

    units = [u for a in audit for u in a["units"]]
    print(
        "scored units: %d  (idf over %d documents, %d types)"
        % (len(units), table["n"], len(table["idf"]))
    )

    if args.dry_run:
        print("--dry-run: nothing written")
        return

    entries = W.write_sidecar(
        args.out,
        spans_by_content,
        dataset=os.path.basename(args.corpus)[:-5],
        floor=args.floor,
        meta={
            "loss": "teacher_self_consistency",
            "built_by": "build_consistency_weights.py",
            "k": args.k,
            "n_null": args.n_null,
            "samples": args.samples,
            "scored_tags": list(SCORED_TAGS),
            "unweighted_tags": ["step", "action"],
        },
    )
    W.validate(entries, convs, require_all=False)
    print(
        "wrote %s  (%d/%d conversations weighted)"
        % (os.path.abspath(args.out), len(entries), len(convs))
    )

    if args.surgical_merge:
        mask = json.load(open(args.corpus.replace(".json", ".mask.json")))["entries"]
        merged = {}
        for c in convs:
            e = dict(entries.get(c["sha1"]) or {"len": len(c["content"])})
            ms = (mask.get(c["sha1"]) or {}).get("mask_spans") or []
            if ms:
                e["mask_spans"] = ms
            merged[c["sha1"]] = e
        W.write_sidecar(
            args.surgical_merge,
            None,
            entries=merged,
            dataset=os.path.basename(args.corpus)[:-5],
            floor=args.floor,
            meta={
                "loss": "teacher_self_consistency+surgical",
                "built_by": "build_consistency_weights.py",
                "k": args.k,
            },
        )
        print(
            "wrote %s  (consistency weights + surgical mask_spans)"
            % os.path.abspath(args.surgical_merge)
        )

    with open(args.audit, "w") as f:
        json.dump({"k": args.k, "floor": args.floor, "n_null": args.n_null, "steps": audit}, f)
    print("wrote %s" % os.path.abspath(args.audit))

    report = {
        "corpus": args.corpus,
        "out": os.path.abspath(args.out),
        "k": args.k,
        "floor": args.floor,
        "n_null": args.n_null,
        "units": len(units),
        "steps_uniform_no_samples": n_no_samples,
    }
    if args.report:
        per_tag = token_report(convs, entries, args.tokenizer, args.floor)
        report["token_weight_distribution"] = dict((t, dist(v)) for t, v in per_tag.items())
        print("\n==== token-weight distribution (through the production weight path) ====")
        for t in ("ALL", "think", "plan", "memory", "step", "action", "envelope"):
            d = report["token_weight_distribution"].get(t)
            if not d:
                continue
            print(
                "  %-9s n=%6d  mean %.3f  median %.3f  p10 %.3f p90 %.3f  <0.9: %5.1f%%  <0.5: %5.1f%%"
                % (
                    t,
                    d["tokens"],
                    d["mean"],
                    d["median"],
                    d["p10"],
                    d["p90"],
                    100 * d["frac_below_0.9"],
                    100 * d["frac_below_0.5"],
                )
            )
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=1)
    print("\nTRAIN WITH:  WEBKD_TOKEN_WEIGHTS=%s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
