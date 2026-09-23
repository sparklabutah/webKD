"Regression tests for action info behavior."

from __future__ import print_function

import re

from webkd import action_informativeness as A
from webkd import weights as W
from webkd import score as S


def check(name, cond):
    assert cond, name


SAMPLE = (
    "<think>\nFirst I look at the page. The search box is visible at [14].\n"
    "So I should type into it.\n</think>\n\n"
    "<plan>\n1. Open the search box in the Simple Wikipedia header.\n"
    "2. Type the query 'physics' and submit it.\n"
    "3. Read the first result and report it to the user.\n</plan>\n\n"
    "<step>2</step>\n\n"
    "<memory>\nSearch box is [14]. Query not yet typed.\n</memory>\n\n"
    "<action>\nfill('14', 'physics')\n</action>"
)


def test_segmentation():
    segs = A.step_segments(SAMPLE, W.tag_spans)
    spans = W.tag_spans(SAMPLE)
    check(
        "a1 every rationale tag produced segments",
        set(s["tag"] for s in segs) == set(["think", "plan", "memory"]),
    )

    for tag in A.RATIONALE_TAGS:
        a, b = spans[tag]["inner"]
        mine = [s for s in segs if s["tag"] == tag]
        check("a2 %s segments start at inner start" % tag, mine[0]["start"] == a)
        check("a3 %s segments end at inner end" % tag, mine[-1]["end"] == b)
        contiguous = all(mine[i]["end"] == mine[i + 1]["start"] for i in range(len(mine) - 1))
        check("a4 %s segments are contiguous (no gap, no overlap)" % tag, contiguous)
        joined = "".join(SAMPLE[s["start"] : s["end"]] for s in mine)
        check("a5 %s segments reconstruct the inner text" % tag, joined == SAMPLE[a:b])

    step_s, step_e = spans["step"]["outer"]
    act_s, act_e = spans["action"]["outer"]
    check(
        "teacher_relabel no segment overlaps <step>",
        not any(s["start"] < step_e and s["end"] > step_s for s in segs),
    )
    check(
        "expert_completion no segment overlaps <action>",
        not any(s["start"] < act_e and s["end"] > act_s for s in segs),
    )
    check(
        "style_matched segments are in document order",
        all(segs[i]["end"] <= segs[i + 1]["start"] for i in range(len(segs) - 1)),
    )
    check(
        "student_restyle a numbered plan splits one segment per line",
        len([s for s in segs if s["tag"] == "plan"]) == 3,
    )
    check(
        "b5 short fragments merged, none below the floor length",
        all(A._nonspace(s["text"]) >= A.MIN_SEG_CHARS or len(segs) == 1 for s in segs),
    )
    short_plan = "<plan>\n1. Go.\n2. Look.\n3. Stop.\n</plan>\n\n<action>\nnoop()\n</action>"
    check(
        "b6 a plan of tiny lines collapses instead of making 1-token ablations",
        len(A.step_segments(short_plan, W.tag_spans)) == 1,
    )


def test_segment_edge_cases():
    check("a6 empty text -> no segments", A.segment_inner("") == [])
    one = A.segment_inner("short")
    check("a7 a single short line stays one segment", one == [(0, 5)])
    txt = "aaaaaaaaaaaaaaaaaaaaaaaaaaaa.\nb.\ncccccccccccccccccccccccccccccc."
    segs = A.segment_inner(txt)
    check("a8 tiny middle fragment merged into its neighbour", len(segs) == 2)
    check("a9 merged segments still partition", "".join(txt[s:e] for s, e in segs) == txt)
    many = "\n".join("line number %d has enough characters here" % i for i in range(40))
    capped = A.segment_inner(many, max_segs=8)
    check("a10 max_segs caps the count", len(capped) == 8)
    check("a11 capping still partitions", "".join(many[s:e] for s, e in capped) == many)


def test_ablation():
    segs = A.step_segments(SAMPLE, W.tag_spans)
    base_call = S.action_inner_span(SAMPLE)
    check("c1 action call found", SAMPLE[base_call[0] : base_call[1]] == "fill('14', 'physics')")
    for s in segs:
        ab = A.ablate(SAMPLE, s)
        check(
            "c2 ablation removes exactly the segment (%s %d)" % (s["tag"], s["start"]),
            len(ab) == len(SAMPLE) - (s["end"] - s["start"]),
        )
        sp = S.action_inner_span(ab)
        check(
            "c3 action call survives ablation (%s %d)" % (s["tag"], s["start"]),
            sp is not None and ab[sp[0] : sp[1]] == "fill('14', 'physics')",
        )
        check(
            "c4 ablation keeps the other four tags parseable (%s %d)" % (s["tag"], s["start"]),
            "action" in W.tag_spans(ab),
        )


def test_transform():
    c = [0.0, 1.0, 2.0, 3.0]
    w, info = A.weights_from_contributions(c, floor=0.1, tau=0.05)
    check("d1 monotone in contribution", w == sorted(w))
    check("d2 spans exactly [floor, 1]", abs(w[0] - 0.1) < 1e-9 and abs(w[-1] - 1.0) < 1e-9)
    check("d3 evenly spaced ranks", abs(w[1] - (0.1 + 0.9 / 3.0)) < 1e-9)
    check("d4 shrink is 1 when the spread is large", abs(info["shrink"] - 1.0) < 1e-9)

    cn = [-2.0, -1.0, 0.5, 4.0]
    wn, _ = A.weights_from_contributions(cn, floor=0.1, tau=0.05)
    check(
        "d5 negative contributions get the floor, not a mask",
        abs(wn[0] - 0.1) < 1e-9 and wn[0] > 0.0,
    )
    check(
        "d6 a purely negative step is still ranked, not uniform",
        A.weights_from_contributions([-3.0, -2.0, -1.0], floor=0.1, tau=0.05)[0]
        == [0.1, 0.55, 1.0],
    )
    check(
        "d7 adding a constant to every contribution leaves weights unchanged",
        A.weights_from_contributions([x + 7.0 for x in cn], floor=0.1, tau=0.05)[0] == wn,
    )

    ties, _ = A.weights_from_contributions([1.0, 1.0, 5.0], floor=0.1, tau=0.05)
    check("d8 ties share a weight", abs(ties[0] - ties[1]) < 1e-9 and ties[2] > ties[0])

    flat, finfo = A.weights_from_contributions([0.0, 0.0001, 0.0002], floor=0.1, tau=0.05)
    check(
        "d9 dead zone collapses a noise-level step to uniform",
        max(flat) - min(flat) < 0.02 and finfo["shrink"] < 0.01,
    )
    check(
        "d10 dead-zone weights sit at the midpoint (== no reweighting under weight-sum norm)",
        all(abs(x - 0.55) < 0.02 for x in flat),
    )

    check(
        "d11 a single segment keeps weight 1.0",
        A.weights_from_contributions([3.3], floor=0.1)[0] == [1.0],
    )
    check("d12 no segments -> no weights", A.weights_from_contributions([])[0] == [])

    part, pinfo = A.weights_from_contributions([0.0, 0.025], floor=0.1, tau=0.05)
    check("d13 partial shrink at half the dead-zone width", abs(pinfo["shrink"] - 0.5) < 1e-9)
    check("d14 partial shrink keeps the midpoint", abs((part[0] + part[1]) / 2.0 - 0.55) < 1e-9)

    check(
        "d15 average_ranks handles a tie block",
        A.average_ranks([5.0, 1.0, 1.0, 1.0]) == [3.0, 1.0, 1.0, 1.0],
    )

    spans, _ = A.weight_spans_for_step(
        SAMPLE,
        A.step_segments(SAMPLE, W.tag_spans),
        [0.0] * len(A.step_segments(SAMPLE, W.tag_spans)),
    )
    check(
        "d16 weight_spans_for_step emits one span per segment",
        len(spans) == len(A.step_segments(SAMPLE, W.tag_spans)),
    )
    check("d17 all weights inside [floor, 1]", all(0.1 <= s[2] <= 1.0 for s in spans))


def test_through_production_path():
    segs = A.step_segments(SAMPLE, W.tag_spans)
    contribs = [float(i) for i in range(len(segs))]
    spans, _ = A.weight_spans_for_step(SAMPLE, segs, contribs, floor=0.1)
    entry = W.build_entry(SAMPLE, weight_spans=spans, default_weight=1.0, floor=0.1)
    flat = W.flatten_weight_spans(entry["weight_spans"])
    check("e1 flattening does not change the span count", len(flat) == len(spans))

    offsets, _i = [], 0
    for m in re.finditer(r"\S+", SAMPLE):
        offsets.append((m.start(), m.end()))
    weights = W.token_weight_vector(offsets, flat, 1.0, 0.1, [])
    check("e2 one weight per token", len(weights) == len(offsets))
    check("e3 every weight in [floor, 1]", all(0.1 <= w <= 1.0 for w in weights))

    act_s, act_e = W.tag_spans(SAMPLE)["action"]["outer"]
    act_w = [w for (s, e), w in zip(offsets, weights) if s >= act_s and e <= act_e]
    check("e4 <action> tokens keep weight 1.0", act_w and all(abs(w - 1.0) < 1e-9 for w in act_w))
    st_s, st_e = W.tag_spans(SAMPLE)["step"]["outer"]
    st_w = [w for (s, e), w in zip(offsets, weights) if s >= st_s and e <= st_e]
    check("e5 <step> keeps weight 1.0", st_w and all(abs(w - 1.0) < 1e-9 for w in st_w))

    lowest = min(segs, key=lambda s: contribs[segs.index(s)])
    low_w = [
        w for (s, e), w in zip(offsets, weights) if s >= lowest["start"] and e <= lowest["end"]
    ]
    check(
        "e6 the least informative segment's tokens carry the floor",
        low_w and all(abs(w - 0.1) < 1e-9 for w in low_w),
    )
    tag_open = SAMPLE.index("<think>")
    open_w = [w for (s, e), w in zip(offsets, weights) if s == tag_open]
    check("e7 the <think> tag itself keeps weight 1.0", open_w and abs(open_w[0] - 1.0) < 1e-9)
