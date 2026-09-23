"""Assign separate weights to changed and unchanged sections."""

import argparse
import json
import os


def _load(name):
    from importlib import import_module

    return import_module("webkd." + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.json")
    ap.add_argument(
        "--mask",
        default="data/surgical.mask.json",
        help="surgical side-car; its mask_spans ARE the unchanged sections",
    )
    ap.add_argument("--unchanged-weight", type=float, default=0.3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    W = _load("weights")
    out = a.out or "artifacts/edit.weights.json"

    if not os.path.exists(a.mask):
        raise SystemExit(f"{a.mask} not found -- run build_surgical.py first")
    surg = json.load(open(a.mask))
    entries = surg.get("entries", surg)

    spans, n_spanned = {}, 0
    convs = W.load_conversations(a.corpus)
    by_hash = {}
    for c in convs:
        import hashlib

        by_hash[hashlib.sha1(c["content"].encode()).hexdigest()] = c["content"]

    for h, e in entries.items():
        content = by_hash.get(h)
        if content is None:
            continue
        ms = e.get("mask_spans") or []
        if ms:
            n_spanned += 1

        spans[content] = [[s, t, a.unchanged_weight] for s, t in ms]

    W.write_sidecar(
        out,
        spans,
        dataset=os.path.splitext(os.path.basename(a.corpus))[0],
        default_weight=1.0,
        floor=min(0.05, a.unchanged_weight),
        meta={
            "loss": "edit_weighted",
            "unchanged_weight": a.unchanged_weight,
            "derived_from": a.mask,
            "built_by": os.path.basename(__file__),
        },
    )
    tot = sum(len(v) for v in spans.values())
    print(f"wrote {out}")
    print(f"  turns with >=1 unchanged span: {n_spanned}/{len(spans)}   unchanged spans: {tot}")
    print(
        f"  changed text w=1.0 | unchanged text w={a.unchanged_weight} | floor {min(0.05, a.unchanged_weight)}"
    )


if __name__ == "__main__":
    main()
