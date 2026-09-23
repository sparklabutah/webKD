"""Build a rationale groundedness weighting baseline."""

from __future__ import print_function

import argparse
import collections
import glob
import json
import os
import re

from webkd import surgical as SU
from webkd import weights as W
from webkd import trajectory as T
from webkd import grounding as F


SCORED_TAGS_DEFAULT = ("think", "plan", "memory")

CLAIM_PATTERNS = (
    (r'"([^"]{2,})"', 1, "quote"),
    (r"'([^']{2,})'", 1, "quote"),
    (r"\b\d[\d,\.]*\b", 0, "number"),
    (r"\b[A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,})*", 0, "entity"),
)

_SENTENCE_START = re.compile(r"(?:^|[.!?:;\n])[ \t]*(?:[-*•][ \t]*)?(?:\d+[.)][ \t]*)?$")


def claim_occurrences(text, policy="span"):
    """[(start, end, claim, kind)] -- where scf_filter's claims actually sit in `text`."""
    accepted = set(c.strip().lower() for c in F.extract_claims(text))
    if not accepted:
        return []
    out = []
    for pattern, group, kind in CLAIM_PATTERNS:
        for m in re.finditer(pattern, text):
            raw = m.group(group)
            if raw is None:
                continue
            start, end = m.span(group)
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            start, end = start + lead, end - trail
            claim = raw.strip()
            if not claim or end <= start or claim.lower() not in accepted:
                continue
            if policy != "scf":
                if kind == "quote" and "\n" in claim:
                    continue
                if kind == "number" and is_list_enumerator(text, start, end):
                    continue
                if kind == "entity":
                    trimmed = trim_orthographic_capital(text, start, end)
                    if trimmed is None:
                        continue
                    start, end = trimmed
                    claim = text[start:end]
            out.append((start, end, claim, kind))
    out.sort()
    return out


def trim_orthographic_capital(text, start, end):
    """(start, end) with a sentence-initial leading Capital dropped, or None if nothing is left."""
    if not _SENTENCE_START.search(text, 0, start):
        return (start, end)
    m = re.match(r"\S+\s+", text[start:end])
    if m is None:
        return None
    return (start + m.end(), end)


def is_list_enumerator(text, start, end):
    """True when text[start:end] is a bullet NUMBER ("3." in "3. Search for ..."), not a claim."""
    if not _SENTENCE_START.search(text, 0, start):
        return False
    return re.match(r"[.)]\s", text[end : end + 2]) is not None


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_bounds(text, start, end):
    """The clause/bullet/sentence of `text` that contains [start, end)."""
    lo = 0
    for m in _SENT_SPLIT.finditer(text, 0, start):
        lo = m.end()
    m = _SENT_SPLIT.search(text, end)
    hi = m.start() if m else len(text)
    return (lo, max(hi, end))


def _merge(spans):
    """Merge overlapping/adjacent [start, end] pairs."""
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def grounding_spans_for_target(
    content,
    sections,
    corpus_norm,
    scored_tags=SCORED_TAGS_DEFAULT,
    low=0.1,
    strict=True,
    policy="span",
    scope="claim",
):
    """([[start, end, low], ...], record) -- the ungrounded spans of one assistant turn."""
    found = W.tag_spans(content)
    spans, record = [], {"tags": {}, "claims": []}
    for tag in scored_tags:
        if tag not in found:
            continue
        lo, hi = found[tag]["inner"]
        text = content[lo:hi]
        expected = (sections or {}).get(tag)
        if expected is not None and text != expected:
            if strict:
                raise ValueError(
                    "span/dump mismatch on <%s>: the section found in the target is "
                    "not the section parsed from the dump" % tag
                )
            record.setdefault("mismatched_tags", []).append(tag)
            continue
        occ = claim_occurrences(text, policy=policy)
        n_grounded = n_ungrounded = 0
        bad = []
        for s, e, claim, kind in occ:
            if F.is_grounded(claim, corpus_norm):
                n_grounded += 1
                record["claims"].append(
                    {"tag": tag, "claim": claim, "kind": kind, "grounded": True}
                )
            else:
                n_ungrounded += 1
                ds, de = sentence_bounds(text, s, e) if scope == "sentence" else (s, e)
                bad.append([lo + ds, lo + de])
                record["claims"].append(
                    {
                        "tag": tag,
                        "claim": claim,
                        "kind": kind,
                        "grounded": False,
                        "span": [lo + ds, lo + de],
                        "claim_span": [lo + s, lo + e],
                    }
                )
        for s, e in _merge(bad):
            spans.append([s, min(e, hi), float(low)])
        record["tags"][tag] = {
            "inner": [lo, hi],
            "chars": hi - lo,
            "claims": len(occ),
            "grounded": n_grounded,
            "ungrounded": n_ungrounded,
            "downweighted_chars": sum(e - s for s, e in _merge(bad)),
        }
    spans.sort()
    return spans, record


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _step_num(path):
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def find_study_dir(task_dir):
    for d in sorted(glob.glob(os.path.join(task_dir, "*"))):
        if os.path.isdir(os.path.join(d, "training_data")):
            return d
    return None


def edited_sections(path):
    """Sections of an edited dump, parsed the way convert (and surgical) parse them."""
    data = _load_json(path)
    if data is None:
        return None
    if "content" in data:
        return SU.parse_edit_sections(T.get_text(data.get("content", "")))
    return dict((k, str(data.get(k) or "").strip()) for k in SU.TAG_ORDER)


def index_cell(edits_root):
    out = {}
    for task_dir in sorted(glob.glob(os.path.join(os.path.abspath(edits_root), "task_*"))):
        props = _load_json(os.path.join(task_dir, "properties.json")) or {}
        study = find_study_dir(task_dir)
        if study is None:
            continue
        ep = T.load_episode(study)
        if ep is None:
            continue
        goal = ep["steps"][0]["goal"] if ep["steps"] else ""
        td = os.path.join(study, "training_data")
        for path in sorted(glob.glob(os.path.join(td, "agent_output_step_*.json"))):
            n = _step_num(path)
            if n < 0:
                continue
            sec = edited_sections(path)
            if sec is None:
                continue
            value, _spans = SU.assemble_target(sec)
            if not value:
                continue
            key = W.content_key(value)
            if key in out:
                out[key]["duplicates"].append((props.get("task_id"), n))
                continue
            out[key] = {
                "content": value,
                "sections": sec,
                "corpus": T.axtree_upto(ep, n) + "\n" + goal,
                "task_id": props.get("task_id"),
                "step": n,
                "t_star": props.get("t_star"),
                "episode": study,
                "goal": goal,
                "duplicates": [],
            }
    return out


def token_stats(conversations, entries, tokenizer_path, scored_tags):
    """Per-tag token counts and the down-weighted fraction, through the PRODUCTION weight code."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    per_tag = collections.defaultdict(lambda: {"tokens": 0, "down": 0, "weight_sum": 0.0})
    tot = down = 0
    weight_sum = 0.0
    per_sample = []
    for c in conversations:
        entry = entries[c["sha1"]]
        content = c["content"]
        enc = tok(content, add_special_tokens=False, return_offsets_mapping=True)
        offsets = list(enc["offset_mapping"])
        weights = W.token_weight_vector(
            offsets,
            W.flatten_weight_spans(entry.get("weight_spans") or []),
            float(entry.get("default_weight", 1.0)),
            float(entry.get("floor", W.DEFAULT_FLOOR)),
            entry.get("mask_spans") or [],
        )
        outer = W.tag_spans(content)
        n_down = 0
        for (s, e), w in zip(offsets, weights):
            if e <= s:
                continue
            tag = "envelope"
            for t, sp in outer.items():
                if s >= sp["outer"][0] and s < sp["outer"][1]:
                    tag = t
                    break
            tot += 1
            weight_sum += w
            per_tag[tag]["tokens"] += 1
            per_tag[tag]["weight_sum"] += w
            if w < 1.0 - 1e-9:
                down += 1
                n_down += 1
                per_tag[tag]["down"] += 1
        per_sample.append(n_down / float(max(1, len(offsets))))
    per_sample.sort()
    return {
        "response_tokens": tot,
        "down_tokens": down,
        "down_frac": down / float(max(1, tot)),
        "mean_weight": weight_sum / float(max(1, tot)),
        "median_down_frac_per_sample": per_sample[len(per_sample) // 2] if per_sample else 0.0,
        "max_down_frac_per_sample": per_sample[-1] if per_sample else 0.0,
        "per_tag": dict((k, dict(v)) for k, v in per_tag.items()),
    }


def build(args):
    dataset_path = args.dataset
    convs = W.load_conversations(dataset_path)
    index = index_cell(args.edits)

    spans_by_content = {}
    records = []
    st = {
        "conversations": len(convs),
        "cell_targets": len(index),
        "matched": 0,
        "unmatched": 0,
        "steps_with_ungrounded": 0,
        "claims": 0,
        "claims_ungrounded": 0,
        "per_tag": collections.defaultdict(lambda: collections.Counter()),
        "scf_would_reject": 0,
        "grounded_fraction_sum": 0.0,
        "steps_with_claims": 0,
    }

    for c in convs:
        hit = index.get(c["sha1"])
        if hit is None:
            st["unmatched"] += 1
            spans_by_content[c["content"]] = []
            records.append({"sha1": c["sha1"], "matched": False})
            continue
        st["matched"] += 1
        corpus_norm = F._norm(hit["corpus"])
        spans, rec = grounding_spans_for_target(
            c["content"],
            hit["sections"],
            corpus_norm,
            scored_tags=args.tags,
            low=args.low_weight,
            strict=not args.lenient_spans,
            policy=args.claim_policy,
            scope=args.span_scope,
        )
        spans_by_content[c["content"]] = spans

        n_claims = len(rec["claims"])
        n_bad = sum(1 for cl in rec["claims"] if not cl["grounded"])
        st["claims"] += n_claims
        st["claims_ungrounded"] += n_bad
        if n_bad:
            st["steps_with_ungrounded"] += 1
        if n_claims:
            st["steps_with_claims"] += 1
            st["grounded_fraction_sum"] += (n_claims - n_bad) / float(n_claims)
        for tag, tv in rec["tags"].items():
            st["per_tag"][tag]["claims"] += tv["claims"]
            st["per_tag"][tag]["ungrounded"] += tv["ungrounded"]
            st["per_tag"][tag]["chars"] += tv["chars"]
            st["per_tag"][tag]["down_chars"] += tv["downweighted_chars"]
            st["per_tag"][tag]["steps"] += 1
            if tv["ungrounded"]:
                st["per_tag"][tag]["steps_with_ungrounded"] += 1

        if args.scf_parity:
            reasoning = " ".join(
                [hit["sections"].get("think", ""), hit["sections"].get("memory", "")]
            )
            res = F.scf_score(reasoning, hit["corpus"], args.scf_thresh)
            if not res["state_consistent"]:
                st["scf_would_reject"] += 1
            rec["scf"] = res

        rec.update(
            {
                "sha1": c["sha1"],
                "matched": True,
                "task_id": hit["task_id"],
                "step": hit["step"],
                "t_star": hit["t_star"],
                "n_spans": len(spans),
                "target_chars": len(c["content"]),
            }
        )
        records.append(rec)

    return convs, spans_by_content, records, st


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset", default="data/corpus.json", help="the ShareGPT corpus the weights are FOR"
    )
    ap.add_argument(
        "--edits",
        default="data/episodes",
        help="live-edit cell used for the AXTree grounding corpus. Use the UNFILTERED "
        "parent (style_matched_none), not style_filtered -- see index_cell.__doc__",
    )
    ap.add_argument("--out", default="artifacts/grounding.weights.json")
    ap.add_argument(
        "--low-weight",
        type=float,
        default=0.1,
        help="weight given to an UNGROUNDED claim's characters (default 0.1)",
    )
    ap.add_argument(
        "--floor",
        type=float,
        default=0.1,
        help="side-car floor; weights are clamped to [floor, 1] at train time",
    )
    ap.add_argument(
        "--tags",
        default="think,plan,memory",
        help="sections whose claims are scored (step/action are never scored)",
    )
    ap.add_argument(
        "--span-scope",
        choices=("claim", "sentence"),
        default="sentence",
        help="what gets the low weight: just the ungrounded claim's characters, or the "
        "whole sentence/plan bullet containing it (default; see sentence_bounds)",
    )
    ap.add_argument(
        "--claim-policy",
        choices=("span", "scf"),
        default="span",
        help="span (default): a sentence/bullet-initial capital is orthographic, so its "
        "first word is not part of the claim (see trim_orthographic_capital); "
        "scf: every extract_claims claim exactly as the per-step filter sees it",
    )
    ap.add_argument("--scf-thresh", type=float, default=F.THRESH_DEFAULT)
    ap.add_argument(
        "--scf-parity",
        action="store_true",
        default=True,
        help="also recompute the per-STEP SCF verdict, for comparison with the filter",
    )
    ap.add_argument("--no-scf-parity", dest="scf_parity", action="store_false")
    ap.add_argument(
        "--lenient-spans",
        action="store_true",
        help="skip a section instead of aborting when its span does not reproduce the "
        "text parsed from the dump (default: abort -- a silent offset error here "
        "would down-weight arbitrary characters)",
    )
    ap.add_argument(
        "--merge-surgical-mask",
        default="",
        help="fold a v1 *.mask.json's mask_spans into the side-car (they cannot both "
        "be active via env vars)",
    )
    ap.add_argument(
        "--token-stats",
        action="store_true",
        help="tokenize every target and report the down-weighted token fraction",
    )
    ap.add_argument("--tokenizer", default="models/student")
    ap.add_argument(
        "--examples",
        type=int,
        default=0,
        metavar="N",
        help="print N grounded and N ungrounded claims with their AXTree evidence",
    )
    ap.add_argument("--dry-run", action="store_true", help="compute + report, write nothing")
    args = ap.parse_args()
    args.tags = tuple(t.strip() for t in args.tags.split(",") if t.strip())

    convs, spans_by_content, records, st = build(args)

    mask_by_content = None
    if args.merge_surgical_mask:
        table = (_load_json(args.merge_surgical_mask) or {}).get("entries", {})
        mask_by_content = {}
        for c in convs:
            ent = table.get(c["sha1"])
            if ent and ent.get("mask_spans"):
                mask_by_content[c["content"]] = ent["mask_spans"]

    entries = W.build_entries(
        spans_by_content,
        default_weight=1.0,
        floor=args.floor,
        mask_spans_by_content=mask_by_content,
    )
    W.validate(entries, convs, require_all=True)

    print("\n==== grounding weights for %s ====" % os.path.basename(args.dataset))
    print(
        "conversations           : %d  (matched to the edit cell: %d, unmatched: %d)"
        % (st["conversations"], st["matched"], st["unmatched"])
    )
    print(
        "scored tags             : %s   ungrounded weight %.2f (floor %.2f)"
        % (",".join(args.tags), args.low_weight, args.floor)
    )
    print(
        "claims                  : %d  (%d ungrounded, %.1f%%)"
        % (
            st["claims"],
            st["claims_ungrounded"],
            100.0 * st["claims_ungrounded"] / max(1, st["claims"]),
        )
    )
    print(
        "steps with >=1 ungrounded claim: %d / %d  (%.1f%%)"
        % (
            st["steps_with_ungrounded"],
            st["matched"],
            100.0 * st["steps_with_ungrounded"] / max(1, st["matched"]),
        )
    )
    if st["steps_with_claims"]:
        print(
            "mean per-step grounded fraction: %.4f"
            % (st["grounded_fraction_sum"] / st["steps_with_claims"])
        )
    if args.scf_parity:
        print(
            "SCF (per-STEP, think++memory, thresh %.2f) would reject: %d / %d (%.1f%%)"
            % (
                args.scf_thresh,
                st["scf_would_reject"],
                st["matched"],
                100.0 * st["scf_would_reject"] / max(1, st["matched"]),
            )
        )
    print("\nper tag:")
    for tag in args.tags:
        v = st["per_tag"].get(tag)
        if not v:
            continue
        print(
            "  %-7s steps %3d (%3d with an ungrounded claim)  claims %4d (%3d ungrounded)  "
            "chars %6d (%5d down-weighted, %4.1f%%)"
            % (
                tag,
                v["steps"],
                v["steps_with_ungrounded"],
                v["claims"],
                v["ungrounded"],
                v["chars"],
                v["down_chars"],
                100.0 * v["down_chars"] / max(1, v["chars"]),
            )
        )

    summary = W.summarize(entries, convs)
    print(
        "\nside-car: %d entries, %d weight spans, %d weighted chars of %d target chars (%.2f%%)"
        % (
            summary["entries"],
            summary["weight_spans"],
            summary["weighted_chars"],
            summary["target_chars"],
            100.0 * summary["weighted_chars"] / max(1, summary["target_chars"]),
        )
    )

    if args.token_stats:
        ts = token_stats(convs, entries, args.tokenizer, args.tags)
        st["token_stats"] = ts
        print(
            "\ntokens: %d response tokens, %d down-weighted (%.2f%%), mean weight %.4f; "
            "per-sample median %.2f%%, max %.2f%%"
            % (
                ts["response_tokens"],
                ts["down_tokens"],
                100.0 * ts["down_frac"],
                ts["mean_weight"],
                100.0 * ts["median_down_frac_per_sample"],
                100.0 * ts["max_down_frac_per_sample"],
            )
        )
        for tag in list(SU.TAG_ORDER) + ["envelope"]:
            v = ts["per_tag"].get(tag)
            if not v:
                continue
            print(
                "  %-8s %6d tokens, %5d down-weighted (%5.2f%%), mean weight %.4f"
                % (
                    tag,
                    v["tokens"],
                    v["down"],
                    100.0 * v["down"] / max(1, v["tokens"]),
                    v["weight_sum"] / max(1, v["tokens"]),
                )
            )

    if args.examples:
        print_examples(args, convs, records, args.examples)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    W.write_sidecar(
        args.out,
        spans_by_content,
        dataset=os.path.basename(args.dataset)[:-5],
        floor=args.floor,
        mask_spans_by_content=mask_by_content,
        entries=entries,
        meta={
            "loss": "grounding",
            "built_by": "build_grounding_weights.py",
            "low_weight": args.low_weight,
            "scored_tags": list(args.tags),
            "span_scope": args.span_scope,
            "claim_policy": args.claim_policy,
            "grounding_cell": args.edits,
            "scf_thresh": args.scf_thresh,
        },
    )
    stats_path = (
        args.out[:-5] + ".stats.json" if args.out.endswith(".json") else args.out + ".stats"
    )
    st_out = dict(
        (
            k,
            (
                dict((kk, dict(vv)) for kk, vv in v.items())
                if isinstance(v, collections.defaultdict)
                else v
            ),
        )
        for k, v in st.items()
    )
    with open(stats_path, "w") as f:
        json.dump({"args": vars(args), "stats": st_out, "samples": records}, f, indent=1)
    print("\nwrote %s  (%d entries)" % (args.out, len(entries)))
    print("wrote %s" % stats_path)
    print("\nTRAIN WITH:  WEBKD_TOKEN_WEIGHTS=%s" % os.path.abspath(args.out))


def print_examples(args, convs, records, n):
    """Show N grounded and N ungrounded claims together with the AXTree line that grounds them."""
    index = index_cell(args.edits)
    by_sha = dict((c["sha1"], c) for c in convs)
    shown_g = shown_u = 0
    for rec in records:
        if not rec.get("matched"):
            continue
        hit = index.get(rec["sha1"])
        content = by_sha[rec["sha1"]]["content"]
        for cl in rec["claims"]:
            want_g = cl["grounded"] and shown_g < n
            want_u = (not cl["grounded"]) and shown_u < n
            if not (want_g or want_u):
                continue
            if cl["grounded"]:
                shown_g += 1
            else:
                shown_u += 1
            print("\n" + "=" * 78)
            print(
                "%s  task %s step %s  <%s>  claim %r"
                % (
                    "GROUNDED" if cl["grounded"] else "UNGROUNDED",
                    rec["task_id"],
                    rec["step"],
                    cl["tag"],
                    cl["claim"],
                )
            )
            if "span" in cl:
                s, e = cl["span"]
                print(
                    "  target chars [%d, %d) -> %r   (weight %.2f)"
                    % (s, e, content[s:e], args.low_weight)
                )
                print("  context: ...%s..." % content[max(0, s - 90) : e + 90].replace("\n", " "))
            else:
                print("  (grounded -> stays at weight 1.0)")
            ev = [ln for ln in hit["corpus"].splitlines() if F._norm(cl["claim"]) in F._norm(ln)][
                :3
            ]
            if ev:
                print("  AXTree evidence:")
                for ln in ev:
                    print("    | %s" % ln.strip()[:150])
            else:
                print(
                    "  AXTree evidence: NONE -- no line of the observations available at this "
                    "step contains it"
                )
                toks = [
                    t
                    for t in re.findall(r"[a-z0-9]+", F._norm(cl["claim"]))
                    if t not in F._STOP and len(t) > 2
                ]
                miss = [t for t in toks if t not in F._norm(hit["corpus"])]
                if miss:
                    print("  missing tokens: %s" % ", ".join(miss[:8]))


if __name__ == "__main__":
    main()
