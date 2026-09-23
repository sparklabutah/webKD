"""Collect repeated teacher corrections for consistency weighting."""

from __future__ import print_function

import argparse
import glob
import json
import os
import sys
import threading
import time

from webkd import corrector as C
from webkd import style as S
from webkd import trajectory as T
from webkd import convert as C2S
from webkd import edit as EL


USAGE = {
    "calls": 0,
    "prompt_tokens": 0,
    "cached_tokens": 0,
    "completion_tokens": 0,
    "reasoning_tokens": 0,
    "errors": 0,
}
_USAGE_LOCK = threading.Lock()


def install_usage_meter():
    """Record token usage without touching corrector."""
    try:
        import openai
    except Exception:
        return False
    real = openai.OpenAI

    def factory(*a, **kw):
        client = real(*a, **kw)
        inner = client.chat.completions.create

        def create(**kwargs):
            resp = inner(**kwargs)
            u = getattr(resp, "usage", None)
            if u is not None:
                d = getattr(u, "completion_tokens_details", None)
                p = getattr(u, "prompt_tokens_details", None)
                with _USAGE_LOCK:
                    USAGE["calls"] += 1
                    USAGE["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
                    USAGE["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
                    USAGE["reasoning_tokens"] += int(getattr(d, "reasoning_tokens", 0) or 0)
                    USAGE["cached_tokens"] += int(getattr(p, "cached_tokens", 0) or 0)
            return resp

        client.chat.completions.create = create
        return client

    openai.OpenAI = factory
    return True


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def study_dir(edits_root, task_id):
    pat = os.path.join(edits_root, "task_%d" % task_id, "*", "training_data")
    hits = sorted(glob.glob(pat))
    return os.path.dirname(hits[0]) if hits else None


_STYLE_CACHE = {}
_STYLE_LOCK = threading.Lock()


def style_text_for(edits_root, task_id):
    with _STYLE_LOCK:
        if task_id in _STYLE_CACHE:
            return _STYLE_CACHE[task_id]
    props = load_json(os.path.join(edits_root, "task_%d" % task_id, "properties.json")) or {}
    st = props.get("style") or {}
    text = ""
    if props.get("mode") in EL.STYLE_MODES and st.get("bank"):
        bank = S.build_style_bank(st["bank"])
        k = int(st.get("k", 3)) - int(st.get("k_prefix", 0) or 0)
        text = S.style_block(S.pick_exemplars(bank, task_id, max(0, k)))
    with _STYLE_LOCK:
        _STYLE_CACHE[task_id] = text
    return text


def build_query(edits_root, task_id, step):
    """(system, user) exactly as TeacherPolicy sent them for this step, or (None, None)."""
    sd = study_dir(edits_root, task_id)
    if sd is None:
        return None, None
    td = os.path.join(sd, "training_data")
    sysd = load_json(os.path.join(td, "system_prompt_step_%d.json" % step))
    usrd = load_json(os.path.join(td, "user_prompt_step_%d.json" % step))
    if sysd is None or usrd is None:
        return None, None
    system = C2S.get_text_from_content(sysd.get("content", ""))
    user = C2S.get_text_from_content(usrd.get("content", ""))
    if not user:
        return None, None
    return system, user + EL.SECTION_CONTRACT + style_text_for(edits_root, task_id)


def read_cache(path):
    """{(task, step): {sample_index: record}} of everything already sampled."""
    have = {}
    if not os.path.isfile(path):
        return have
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("error"):
                continue
            have.setdefault((rec["task_id"], rec["step"]), {})[rec["sample"]] = rec
    return have


def run(args):
    prov = json.load(open(args.stats))["samples"]
    edits_root = os.path.abspath(args.edits)
    steps = [(p["task_id"], p["step"], p["sha1"]) for p in prov]
    if args.limit:
        steps = steps[: args.limit]

    have = read_cache(args.out)
    todo = []
    for task_id, step, sha1 in steps:
        missing = [i for i in range(args.k) if i not in have.get((task_id, step), {})]
        if missing:
            todo.append((task_id, step, sha1, missing))
    print("steps: %d total, %d need sampling (k=%d)" % (len(steps), len(todo), args.k))
    if not todo:
        return

    install_usage_meter()
    llm = C.LiveCorrector()
    print(
        "teacher: provider=%s model=%s effort=%s max_tokens=%d"
        % (llm.provider, llm.model, llm.reasoning_effort, llm.max_tokens)
    )

    out_lock = threading.Lock()
    fh = open(args.out, "a")
    done = [0]
    t0 = time.time()

    def work(item):
        task_id, step, sha1, missing = item
        system, user = build_query(edits_root, task_id, step)
        if user is None:
            recs = [
                {
                    "task_id": task_id,
                    "step": step,
                    "sha1": sha1,
                    "sample": i,
                    "error": "no prompt dump",
                }
                for i in missing
            ]
        else:
            recs = []

            for i in missing:
                try:
                    raw = llm(system, user)
                    sec = T.parse_sections(raw or "")
                    recs.append(
                        {
                            "task_id": task_id,
                            "step": step,
                            "sha1": sha1,
                            "sample": i,
                            "sections": dict((t, sec.get(t, "")) for t in EL.REQUIRED_TAGS),
                            "raw_chars": len(raw or ""),
                            "missing_tags": [t for t in EL.REQUIRED_TAGS if not sec.get(t)],
                        }
                    )
                except Exception as e:
                    with _USAGE_LOCK:
                        USAGE["errors"] += 1
                    recs.append(
                        {
                            "task_id": task_id,
                            "step": step,
                            "sha1": sha1,
                            "sample": i,
                            "error": "%s: %s" % (type(e).__name__, e),
                        }
                    )
        with out_lock:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(todo):
                el = time.time() - t0
                print(
                    "  %d/%d steps  %.0fs elapsed  eta %.0fs  calls=%d in=%d (cached %d) out=%d err=%d"
                    % (
                        done[0],
                        len(todo),
                        el,
                        el / done[0] * (len(todo) - done[0]),
                        USAGE["calls"],
                        USAGE["prompt_tokens"],
                        USAGE["cached_tokens"],
                        USAGE["completion_tokens"],
                        USAGE["errors"],
                    )
                )
                sys.stdout.flush()

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, todo))
    else:
        for item in todo:
            work(item)
    fh.close()

    print("\n==== usage ====")
    print(json.dumps(USAGE, indent=1))
    with open(args.out + ".usage.json", "a") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.time(),
                    "k": args.k,
                    "steps": len(todo),
                    "seconds": time.time() - t0,
                    "usage": USAGE,
                }
            )
            + "\n"
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stats",
        default="data/surgical.stats.json",
        help="corpus stats side-car: gives (task_id, step, sha1) per conversation",
    )
    ap.add_argument(
        "--edits",
        default="data/episodes",
        help="edit cell holding task_*/<study>/training_data (READ ONLY)",
    )
    ap.add_argument("--out", default="artifacts/consistency/samples.jsonl")
    ap.add_argument("--k", type=int, default=3, help="resamples per step")
    ap.add_argument(
        "--workers", type=int, default=8, help="parallel STEPS (k is sequential within)"
    )
    ap.add_argument("--limit", type=int, default=0, help="first N steps only (smoke test)")
    args = ap.parse_args()
    d = os.path.dirname(os.path.abspath(args.out))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    run(args)


if __name__ == "__main__":
    main()
