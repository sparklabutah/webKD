"""Student likelihood weighting baseline."""

import argparse
import json
import os

PINNED = ("step", "action")


def _load(name):
    from importlib import import_module

    return import_module("webkd." + name)


def rank_map(vals, floor):
    """Within-turn average rank -> [floor, 1]. Ties share their mean rank."""
    n = len(vals)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 / (n - 1)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return [floor + (1.0 - floor) * r for r in ranks]


def score(args):
    """Echo-logprob every assistant turn under the student, saving per-token (text, logprob)."""
    from openai import OpenAI

    W = _load("weights")
    convs = W.load_conversations(args.corpus)
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")
    os.makedirs(os.path.dirname(args.out_scores) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out_scores):
        for ln in open(args.out_scores):
            try:
                done.add(json.loads(ln)["sha1"])
            except Exception:
                pass
    import hashlib

    with open(args.out_scores, "a") as f:
        for k, c in enumerate(convs, 1):
            h = hashlib.sha1(c["content"].encode()).hexdigest()
            if h in done:
                continue
            r = client.completions.create(
                model=args.model,
                prompt=c["content"],
                max_tokens=0,
                echo=True,
                logprobs=0,
                temperature=0,
            )
            lp = r.choices[0].logprobs
            f.write(
                json.dumps(
                    {
                        "sha1": h,
                        "tokens": lp.tokens,
                        "logprobs": [x if x is not None else 0.0 for x in lp.token_logprobs],
                    }
                )
                + "\n"
            )
            f.flush()
            if k % 25 == 0:
                print(f"  scored {k}/{len(convs)}", flush=True)
    print(f"wrote {args.out_scores}")


def build(args):
    W = _load("weights")
    convs = {}
    import hashlib

    for c in W.load_conversations(args.corpus):
        convs[hashlib.sha1(c["content"].encode()).hexdigest()] = c["content"]

    spans, n = {}, 0
    for ln in open(args.scores):
        try:
            row = json.loads(ln)
        except Exception:
            continue
        content = convs.get(row["sha1"])
        if content is None:
            continue
        toks, lps = row["tokens"], row["logprobs"]

        offs, pos = [], 0
        for t in toks:
            j = content.find(t, pos)
            if j < 0:
                offs.append(None)
                continue
            offs.append((j, j + len(t)))
            pos = j + len(t)
        tags = W.tag_spans(content)
        pin = []
        for tg in PINNED:
            if tg in tags:
                pin.append(tags[tg]["inner"] if isinstance(tags[tg], dict) else tags[tg])
        idx = [
            i
            for i, o in enumerate(offs)
            if o is not None and not any(p and p[0] <= o[0] < p[1] for p in pin)
        ]
        if not idx:
            continue
        ws = rank_map([lps[i] for i in idx], args.floor)
        spans[content] = [[offs[i][0], offs[i][1], w] for i, w in zip(idx, ws)]
        n += 1

    out = args.out or "artifacts/likelihood.weights.json"
    W.write_sidecar(
        out,
        spans,
        dataset=os.path.splitext(os.path.basename(args.corpus))[0],
        default_weight=1.0,
        floor=args.floor,
        meta={
            "loss": "student_perplexity",
            "direction": "similar-to-student weighted UP",
            "pinned_tags": list(PINNED),
            "built_by": os.path.basename(__file__),
        },
    )
    allw = [w for v in spans.values() for *_, w in v]
    print(
        f"wrote {out}: {n} turns, {len(allw)} weighted tokens, mean w={sum(allw) / max(1, len(allw)):.3f}"
    )


def selftest():
    ok = fail = 0

    def ck(n, c):
        nonlocal ok, fail
        ok += 1 if c else 0
        fail += 0 if c else 1
        if not c:
            print("  FAIL", n)

    r = rank_map([-5.0, -1.0, -3.0], 0.1)
    ck("most-likely token gets w=1", abs(r[1] - 1.0) < 1e-9)
    ck("least-likely token gets floor", abs(r[0] - 0.1) < 1e-9)
    ck("middle is between", 0.1 < r[2] < 1.0)
    ck("ties share rank", rank_map([-2.0, -2.0], 0.1) == [0.55, 0.55])
    ck("single token", rank_map([-3.0], 0.1) == [1.0])
    ck("empty", rank_map([], 0.1) == [])
    print(f"{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.json")
    ap.add_argument("--model", default="models/student")
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--out-scores", default="artifacts/likelihood_scores.jsonl")
    ap.add_argument("--scores", default="artifacts/likelihood_scores.jsonl")
    ap.add_argument("--floor", type=float, default=0.1)
    ap.add_argument("--out", default="")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if a.score:
        score(a)
    if a.build:
        build(a)
