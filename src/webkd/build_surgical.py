"""Build a ShareGPT corpus and unchanged-section masks for surgical training."""

from __future__ import print_function

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

from webkd import style as S
from webkd import surgical as SU
from webkd import trajectory as T
from webkd import convert as C2S


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def step_num(path):
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def find_study_dir(task_dir):
    """The single `<timestamp>_ReplayTeacher-.../` run directory inside a task_* dir."""
    for d in sorted(glob.glob(os.path.join(task_dir, "*"))):
        if os.path.isdir(os.path.join(d, "training_data")):
            return d
    return None


def edited_sections(path):
    """Sections of an EDITED dump, parsed exactly the way convert will parse them."""
    data = load_json(path)
    if data is None:
        return None
    if "content" in data:
        return C2S.parse_content_sections(C2S.get_text_from_content(data.get("content", "")))
    return dict((k, str(data.get(k) or "").strip()) for k in SU.TAG_ORDER)


def student_sections(path):
    """Sections of the STUDENT's own P-0 dump. LEGACY format: every tag is packed inside `think`,"""
    data = load_json(path)
    if data is None:
        return None
    if "content" in data:
        return T.parse_sections(T.get_text(data.get("content", "")))
    return S.recover_sections(dict((k, str(data.get(k) or "").strip()) for k in T.TAGS))


def student_step_sections(src_dir, orig_root, task_id):
    """{step: sections} for the P-0 episode this edit branched off."""
    ep = src_dir if (src_dir and os.path.isdir(os.path.join(src_dir, "training_data"))) else None
    if ep is None and orig_root:
        pat = os.path.join(orig_root, "*timewarp.%d_*" % task_id)
        cands = [
            d for d in sorted(glob.glob(pat)) if os.path.isdir(os.path.join(d, "training_data"))
        ]
        ep = cands[0] if cands else None
    if ep is None:
        return None, None
    out = {}
    for path in glob.glob(os.path.join(ep, "training_data", "agent_output_step_*.json")):
        n = step_num(path)
        if n < 0:
            continue
        sec = student_sections(path)
        if sec is not None:
            out[n] = sec
    return out, ep


def build(args):
    edits_root = os.path.abspath(args.edits)
    task_dirs = sorted(glob.glob(os.path.join(edits_root, "task_*")))
    if not task_dirs:
        raise SystemExit("no task_* dirs under " + edits_root)

    samples = []
    entries = {}
    provenance = []
    st = {
        "tasks_seen": len(task_dirs),
        "tasks_used": 0,
        "tasks_skipped_error": 0,
        "candidates": 0,
        "kept": 0,
        "dropped_all_unchanged": 0,
        "dropped_empty_target": 0,
        "steps_without_student_original": 0,
        "hash_collisions": 0,
        "changed_count_hist": collections.Counter(),
        "changed_count_hist_kept": collections.Counter(),
        "tag_changed": collections.Counter(),
        "tag_present": collections.Counter(),
        "tag_present_kept": collections.Counter(),
        "tag_masked_kept": collections.Counter(),
        "dropped_by_task": collections.Counter(),
        "t_star_by_task": {},
    }

    for task_dir in task_dirs:
        props = load_json(os.path.join(task_dir, "properties.json")) or {}
        task_id = props.get("task_id")
        t_star = props.get("t_star")
        study = find_study_dir(task_dir)
        if study is None:
            continue
        summary = load_json(os.path.join(study, "summary_info.json")) or {}
        if args.no_error and summary.get("err_msg") is not None:
            st["tasks_skipped_error"] += 1
            continue
        st["tasks_used"] += 1
        st["t_star_by_task"][str(task_id)] = t_star

        orig_by_step, orig_ep = student_step_sections(
            props.get("src"), args.orig_root, task_id or -1
        )
        if orig_by_step is None:
            print(
                "WARN task %s: no student P-0 episode found (src=%s)" % (task_id, props.get("src"))
            )
            orig_by_step = {}

        td = os.path.join(study, "training_data")
        system_text = ""
        steps = sorted(set(step_num(p) for p in glob.glob(os.path.join(td, "*.json"))) - set([-1]))
        if steps:
            sysd = load_json(os.path.join(td, "system_prompt_step_%d.json" % steps[0]))
            if sysd:
                system_text = C2S.get_text_from_content(sysd.get("content", ""))

        for n in steps:
            userd = load_json(os.path.join(td, "user_prompt_step_%d.json" % n))
            if userd is None:
                continue
            user_text = C2S.remove_perform_instructions(
                C2S.get_text_from_content(userd.get("content", ""))
            )
            if not user_text:
                continue
            edit = edited_sections(os.path.join(td, "agent_output_step_%d.json" % n))
            if edit is None:
                continue
            value, spans = SU.assemble_target(edit)
            if not value:
                continue
            st["candidates"] += 1

            present = [t for t in SU.TAG_ORDER if t in spans]
            for t in present:
                st["tag_present"][t] += 1
            orig = orig_by_step.get(n)
            if orig is None:
                st["steps_without_student_original"] += 1
            changed = SU.changed_tags(orig, edit, present, sim_threshold=args.sim_threshold)
            unchanged = [t for t in present if t not in changed]
            for t in changed:
                st["tag_changed"][t] += 1
            st["changed_count_hist"][len(changed)] += 1

            if not changed and not args.keep_unchanged:
                st["dropped_all_unchanged"] += 1
                st["dropped_by_task"][str(task_id)] += 1
                continue

            st["changed_count_hist_kept"][len(changed)] += 1
            for t in present:
                st["tag_present_kept"][t] += 1
            for t in unchanged:
                st["tag_masked_kept"][t] += 1

            spans_masked = SU.mask_spans(spans, unchanged, policy=args.span_policy)
            key = hashlib.sha1(value.encode("utf-8")).hexdigest()
            if key in entries:
                st["hash_collisions"] += 1
                prev = entries[key]["mask_spans"]

                spans_masked = _intersect(prev, spans_masked)
                entries[key]["mask_spans"] = spans_masked
            else:
                entries[key] = {
                    "mask_spans": spans_masked,
                    "len": len(value),
                    "changed": changed,
                    "unchanged": unchanged,
                }
            samples.append(
                {
                    "system": system_text,
                    "conversations": [
                        {"from": "human", "value": user_text},
                        {"from": "gpt", "value": value},
                    ],
                }
            )
            provenance.append(
                {
                    "task_id": task_id,
                    "step": n,
                    "t_star": t_star,
                    "phase": "replay" if (t_star is not None and n < t_star) else "edit",
                    "changed": changed,
                    "unchanged": unchanged,
                    "sha1": key,
                    "n_mask_spans": len(spans_masked),
                    "target_chars": len(value),
                }
            )
            st["kept"] += 1

    return samples, entries, provenance, st


def _intersect(a, b):
    """Character spans present in BOTH span lists (used only on a SHA1 collision)."""
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            lo, hi = max(s1, s2), min(e1, e2)
            if hi > lo:
                out.append([lo, hi])
    out.sort()
    return out


def verify_against_teacher_relabel(samples, teacher_relabel_path):
    teacher_relabel = json.load(open(teacher_relabel_path))
    seen = set()
    for e in teacher_relabel:
        seen.add(
            (e.get("system", ""), e["conversations"][0]["value"], e["conversations"][1]["value"])
        )
    missing = 0
    for e in samples:
        k = (e.get("system", ""), e["conversations"][0]["value"], e["conversations"][1]["value"])
        if k not in seen:
            missing += 1
    return len(teacher_relabel), missing


def decode_check(samples, entries, tokenizer_path, k=3, out=sys.stdout):
    """DECODE the surviving tokens for a few samples and show they are exactly the changed sections."""
    from transformers import AutoTokenizer

    tw = load_surgical_labels()
    tw.set_table(entries)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    picked = []
    for e in samples:
        content = e["conversations"][1]["value"]
        ent = entries[hashlib.sha1(content.encode("utf-8")).hexdigest()]
        if ent["unchanged"]:
            picked.append((content, ent))
        if len(picked) >= k:
            break
    if not picked:
        print("decode-check: no sample has an unchanged section", file=out)
        return
    for content, ent in picked:
        ids = tok(content, add_special_tokens=False)["input_ids"]
        labels = tw.mask_target_label(tok, content, ids, ids)
        kept_ids = [i for i in labels if i != tw.IGNORE_INDEX]
        print("\n" + "=" * 78, file=out)
        print(
            "changed=%s  unchanged=%s   tokens %d -> %d kept (%.1f%%)"
            % (
                ent["changed"],
                ent["unchanged"],
                len(ids),
                len(kept_ids),
                100.0 * len(kept_ids) / max(1, len(ids)),
            ),
            file=out,
        )
        print("---- tokens that SURVIVE the mask ----", file=out)
        print(tok.decode(kept_ids), file=out)
        print("---- tokens that are ZEROED ----", file=out)
        print(tok.decode([i for i, lab in zip(ids, labels) if lab == tw.IGNORE_INDEX]), file=out)


def token_stats(samples, entries, tokenizer_path):
    """How much of the response actually loses its loss, in TOKENS (the units the loss is in)."""
    from transformers import AutoTokenizer

    tw = load_surgical_labels()
    tw.set_table(entries)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tot, masked, per_sample = 0, 0, []
    for e in samples:
        content = e["conversations"][1]["value"]
        ids = tok(content, add_special_tokens=False)["input_ids"]
        labels = tw.mask_target_label(tok, content, ids, ids)
        m = sum(1 for x in labels if x == tw.IGNORE_INDEX)
        tot += len(ids)
        masked += m
        per_sample.append(m / float(max(1, len(ids))))
    per_sample.sort()
    return {
        "response_tokens": tot,
        "masked_tokens": masked,
        "masked_frac": masked / float(max(1, tot)),
        "median_masked_frac_per_sample": per_sample[len(per_sample) // 2] if per_sample else 0.0,
        "max_masked_frac_per_sample": per_sample[-1] if per_sample else 0.0,
    }


def load_surgical_labels():
    from webkd.training import surgical_labels

    return surgical_labels


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--edits", default="data/episodes", help="edited-trajectory root holding task_*/ dirs"
    )
    ap.add_argument(
        "--orig-root",
        default="data/episodes",
        help="fallback P-0 root if properties.json['src'] has moved",
    )
    ap.add_argument("--name", default="surgical", help="dataset name (and file stem)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument(
        "--span-policy",
        choices=("section", "inner"),
        default="section",
        help="section: mask <tag>..</tag> inclusive (default); inner: keep the tags",
    )
    ap.add_argument(
        "--sim-threshold",
        type=float,
        default=1.0,
        help="1.0 = a section is unchanged only if byte-equal after whitespace "
        "normalization; <1.0 = difflib ratio above which it counts as unchanged",
    )
    ap.add_argument(
        "--keep-unchanged",
        action="store_true",
        help="emit all-unchanged steps too (they would train on NOTHING -- diagnostic only)",
    )
    ap.add_argument(
        "--no-error",
        dest="no_error",
        action="store_true",
        default=True,
        help="drop episodes that ended in err_msg (exclude incomplete trajectories)",
    )
    ap.add_argument("--allow-error", dest="no_error", action="store_false")
    ap.add_argument(
        "--verify-against",
        default="",
        help="teacher_relabel corpus that every kept sample must match byte-for-byte ('' to skip)",
    )
    ap.add_argument("--register", action="store_true", help="add the dataset to dataset_info.json")
    ap.add_argument(
        "--token-stats",
        action="store_true",
        help="tokenize every kept target and report the masked-token fraction",
    )
    ap.add_argument(
        "--decode-check",
        type=int,
        default=0,
        metavar="N",
        help="decode N masked samples with the real tokenizer and print the surviving text",
    )
    ap.add_argument("--tokenizer", default="models/student")
    ap.add_argument("--dry-run", action="store_true", help="compute + report, write nothing")
    args = ap.parse_args()

    samples, entries, provenance, st = build(args)

    print("\n==== surgical corpus: %s ====" % args.name)
    print(
        "tasks: %d used, %d skipped (err_msg), of %d"
        % (st["tasks_used"], st["tasks_skipped_error"], st["tasks_seen"])
    )
    print("candidate conversations (== the teacher_relabel corpus): %d" % st["candidates"])
    print("kept: %d    dropped all-unchanged: %d" % (st["kept"], st["dropped_all_unchanged"]))
    print(
        "steps with no student original (all sections treated as changed): %d"
        % st["steps_without_student_original"]
    )
    if st["hash_collisions"]:
        print("identical assistant messages merged (masks intersected): %d" % st["hash_collisions"])
    print("\nsections changed per step:")
    total = float(sum(st["changed_count_hist"].values())) or 1.0
    for k in sorted(st["changed_count_hist"]):
        v = st["changed_count_hist"][k]
        print(
            "  %d changed : %4d steps  (%5.1f%%)%s"
            % (k, v, 100.0 * v / total, "   <- dropped" if k == 0 else "")
        )
    print("\nper-tag, over the %d KEPT steps:" % st["kept"])
    for t in SU.TAG_ORDER:
        pres = st["tag_present_kept"][t]
        msk = st["tag_masked_kept"][t]
        print(
            "  %-7s present %3d  trained %3d  MASKED %3d  (%4.1f%% of kept steps)"
            % (t, pres, pres - msk, msk, 100.0 * msk / max(1, st["kept"]))
        )

    if args.verify_against:
        n_teacher_relabel, missing = verify_against_teacher_relabel(samples, args.verify_against)
        print(
            "\nverify vs %s (n=%d): %d of %d kept samples NOT byte-identical"
            % (os.path.basename(args.verify_against), n_teacher_relabel, missing, len(samples))
        )
        if missing:
            raise SystemExit(
                "ABORT: surgical targets diverge from teacher_relabel -- the ablation would be confounded"
            )

    if args.token_stats:
        ts = token_stats(samples, entries, args.tokenizer)
        st["token_stats"] = ts
        print(
            "\ntokens: %d response tokens over the kept samples, %d masked (%.2f%%); "
            "per-sample median %.2f%%, max %.2f%%"
            % (
                ts["response_tokens"],
                ts["masked_tokens"],
                100.0 * ts["masked_frac"],
                100.0 * ts["median_masked_frac_per_sample"],
                100.0 * ts["max_masked_frac_per_sample"],
            )
        )

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    ds_path = os.path.join(data_dir, args.name + ".json")
    mask_path = os.path.join(data_dir, args.name + ".mask.json")
    stats_path = os.path.join(data_dir, args.name + ".stats.json")

    with open(ds_path, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    with open(mask_path, "w") as f:
        json.dump(
            {
                "version": 1,
                "dataset": args.name,
                "span_policy": args.span_policy,
                "sim_threshold": args.sim_threshold,
                "entries": entries,
            },
            f,
            indent=1,
        )
    st_out = dict(
        (k, (dict(v) if isinstance(v, collections.Counter) else v)) for k, v in st.items()
    )
    with open(stats_path, "w") as f:
        json.dump({"args": vars(args), "stats": st_out, "samples": provenance}, f, indent=1)
    print("\nwrote %s  (n=%d)" % (ds_path, len(samples)))
    print("wrote %s  (%d entries)" % (mask_path, len(entries)))
    print("wrote %s" % stats_path)

    if args.register:
        info_path = os.path.join(data_dir, "dataset_info.json")
        info = json.load(open(info_path))
        info[args.name] = {
            "file_name": args.name + ".json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
        }
        tmp = info_path + ".tmp%d" % os.getpid()
        with open(tmp, "w") as f:
            json.dump(info, f, indent=2)
        os.rename(tmp, info_path)
        print("registered %s in %s" % (args.name, info_path))

    print("\nTRAIN WITH:  WEBKD_SURGICAL_MASK=%s" % os.path.abspath(mask_path))

    if args.decode_check:
        decode_check(samples, entries, args.tokenizer, args.decode_check)


if __name__ == "__main__":
    main()
