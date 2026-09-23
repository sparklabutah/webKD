"""Measure corrected-action likelihood under the frozen student via a scoring server."""

from __future__ import print_function

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ACTION_INNER = re.compile(r"<action>\s*\n(.*?)\n\s*</action>", re.DOTALL)


def action_inner_span(text):
    m = ACTION_INNER.search(text)
    if not m:
        return None
    i, j = m.span(1)
    body = m.group(1)
    lead = len(body) - len(body.lstrip())
    trail = len(body) - len(body.rstrip())
    return (i + lead, j - trail)


def _post(args, full):
    last = None
    for attempt in range(4):
        try:
            r = requests.post(
                "http://localhost:%s/v1/completions" % args.port,
                json={
                    "model": args.model,
                    "prompt": full,
                    "max_tokens": 1,
                    "echo": True,
                    "logprobs": 1,
                    "temperature": 0,
                },
                timeout=1200,
            )
            return r.json()["choices"][0]["logprobs"]["token_logprobs"]
        except Exception as e:
            last = e
            time.sleep(2 + 3 * attempt)
    sys.stderr.write("ERR post: %r\n" % (last,))
    return None


def score_variant(args, tok, ctx_ids, target):
    """(sum_nll_over_action_call, action_token_ids) or (None, None)."""
    span = action_inner_span(target)
    if span is None:
        return None, None
    a, b = span
    enc = tok(target, add_special_tokens=False, return_offsets_mapping=True)
    tgt_ids, offs = enc["input_ids"], enc["offset_mapping"]
    if not tgt_ids:
        return None, None
    full = list(ctx_ids) + list(tgt_ids)
    if len(tgt_ids) >= args.max_len:
        return None, None
    if len(full) > args.max_len:
        full = full[-args.max_len :]
    ctx_len = len(full) - len(tgt_ids)
    lp = _post(args, full)
    if lp is None:
        return None, None
    tgt_lp = lp[ctx_len : ctx_len + len(tgt_ids)]
    if len(tgt_lp) != len(tgt_ids) or any(x is None for x in tgt_lp):
        return None, None
    nll, ids = 0.0, []
    for o, logprob, t in zip(offs, tgt_lp, tgt_ids):
        if o[0] >= a and o[0] < b:
            nll += -logprob
            ids.append(t)
    if not ids:
        return None, None
    return nll, tuple(ids)


def context_ids(tok, system, user):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    try:
        ctx_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        ctx_text = re.sub(r"<think>\s*$", "", ctx_text)
        return tok(ctx_text, add_special_tokens=False)["input_ids"]
    except Exception:
        return tok((system + "\n" if system else "") + user, add_special_tokens=True)["input_ids"]


def score_record(args, tok, rec):
    ctx_ids = context_ids(tok, rec.get("system", ""), rec.get("user", ""))
    target = rec["target"]
    base_nll, base_ids = score_variant(args, tok, ctx_ids, target)
    if base_nll is None:
        return {"index": rec["index"], "sha1": rec["sha1"], "error": "base_failed"}

    segs = []
    for seg in rec["segments"]:
        ablated = target[: seg["start"]] + target[seg["end"] :]
        nll, ids = score_variant(args, tok, ctx_ids, ablated)
        row = {
            "tag": seg["tag"],
            "start": seg["start"],
            "end": seg["end"],
            "chars": seg["end"] - seg["start"],
        }
        if nll is None:
            row["error"] = "ablate_failed"
        elif ids != base_ids:
            row["error"] = "action_retokenized"
        else:
            row["ablate_nll"] = nll
            row["contribution"] = nll - base_nll
        segs.append(row)

    return {
        "index": rec["index"],
        "sha1": rec["sha1"],
        "base_nll": base_nll,
        "action_tokens": len(base_ids),
        "segments": segs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-len", dest="max_len", type=int, default=30000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    with open(args.corpus) as f:
        data = json.load(f)
    nreq = len(data) + sum(len(r["segments"]) for r in data)
    print("scoring %d records / %d requests" % (len(data), nreq), flush=True)

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)

    t0 = time.time()
    done = [0]
    failures = 0
    with open(args.out, "w") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(lambda r: score_record(args, tok, r), data):
                if res.get("error") or any(
                    segment.get("error") for segment in res.get("segments", [])
                ):
                    failures += 1
                f.write(json.dumps(res) + "\n")
                f.flush()
                done[0] += 1
                if done[0] % 10 == 0:
                    print(
                        "  %d/%d records  %.0fs" % (done[0], len(data), time.time() - t0),
                        flush=True,
                    )
    print("done %d records in %.0fs -> %s" % (done[0], time.time() - t0, args.out), flush=True)

    if failures:
        raise SystemExit(f"Scoring failed for {failures} records; inspect the output and retry")


if __name__ == "__main__":
    main()
