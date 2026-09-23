"Regression tests for takeover behavior."

from __future__ import print_function


from webkd import trajectory as T
from webkd import takeover as TK
from webkd import edit as EL


def check(name, cond):
    assert cond, name


STUDENT = {
    "think": "The search box already holds Biophysics, so pressing Enter opens the page.",
    "plan": "1. search Biophysics\n2. read Related Pages",
    "step": "1",
    "memory": "search box holds Biophysics",
    "action": "press('14', 'Enter')",
}
TEACHER_FIX = {
    "think": "Enter will not submit here; the search button at bid 21 is the control that works.",
    "plan": "1. click the search button\n2. read Related Pages",
    "step": "1",
    "memory": "Enter is inert on this page; use bid 21",
    "action": "click('21')",
}


def test_judge_query():
    print("\n[a] judge query")
    q = TK.judge_query(STUDENT, EL.SECTION_CONTRACT)
    check("carries the student's action", "press('14', 'Enter')" in q)
    check("carries the student's rationale", "pressing Enter opens the page" in q)
    check("asks for a verdict", "<verdict>OK</verdict>" in q and "<verdict>FIX</verdict>" in q)
    check("reuses teacher_relabel's five-tag contract verbatim", EL.SECTION_CONTRACT in q)
    check(
        "biases toward OK (guards against becoming expert_completion)",
        "If you are unsure, answer OK." in q,
    )
    check("says the fix will really execute", "EXECUTED in the live browser" in q)
    broken = TK.judge_query({}, EL.SECTION_CONTRACT, student_broken=True)
    check("broken-student query demands a fix", "You MUST answer <verdict>FIX</verdict>" in broken)
    check("broken-student query says so plainly", "no parseable step" in broken)
    check("normal query does NOT demand a fix", "You MUST answer" not in q)


def test_parse_verdict():
    print("\n[b] verdict parsing is three-valued")
    v, s, n = TK.parse_verdict("<verdict>OK</verdict>")
    check("OK parses", v == TK.OK and not s)
    v, s, n = TK.parse_verdict("blah\n<verdict>ok</verdict>\ntrailing")
    check("OK parses case/whitespace-insensitively", v == TK.OK)

    v, s, n = TK.parse_verdict("<verdict>OK</verdict>\n" + T.sections_to_content(TEACHER_FIX))
    check("OK + a repeated step is still OK", v == TK.OK and not s and n == "ok_with_step")

    v, s, n = TK.parse_verdict("<verdict>FIX</verdict>\n" + T.sections_to_content(TEACHER_FIX))
    check("FIX parses with sections", v == TK.FIX and s.get("action") == "click('21')")
    check("FIX keeps all five sections", all(s.get(t) for t in T.TAGS))

    v, s, n = TK.parse_verdict("<verdict>FIX</verdict>\nI would do something else entirely.")
    check(
        "FIX without an action is FIX_UNUSABLE", v == TK.FIX_UNUSABLE and n == "fix_without_action"
    )

    v, s, n = TK.parse_verdict(T.sections_to_content(TEACHER_FIX))
    check("a bare step is inferred as FIX", v == TK.FIX and n.startswith("verdict_inferred"))

    v, s, n = TK.parse_verdict("The step looks broadly fine to me.")
    check("prose with nothing usable is UNPARSED", v == TK.UNPARSED and not s)
    v, s, n = TK.parse_verdict("")
    check("empty reply is UNPARSED", v == TK.UNPARSED)
    v, s, n = TK.parse_verdict("Verdict: FIX\n" + T.sections_to_content(TEACHER_FIX))
    check("untagged 'Verdict: FIX' still parses", v == TK.FIX)


def test_merge_fix():
    print("\n[c] the fix is authoritative on the action, complete on the tags")
    final, info = TK.merge_fix(STUDENT, TEACHER_FIX)
    check("teacher's action wins", final["action"] == "click('21')")
    check("teacher's rationale wins", "bid 21" in final["memory"])
    check("action_changed is reported", info["action_changed"] is True)
    check("nothing refilled when the fix is complete", info["filled_from_student"] == [])

    partial = dict(TEACHER_FIX)
    partial["plan"] = ""
    partial["memory"] = ""
    final, info = TK.merge_fix(STUDENT, partial)
    check("dropped plan is refilled from the student", final["plan"] == STUDENT["plan"])
    check("dropped memory is refilled from the student", final["memory"] == STUDENT["memory"])
    check("refills are reported", set(info["filled_from_student"]) == {"plan", "memory"})
    check("all five tags present after refill", all(final.get(t) for t in T.TAGS))

    same = dict(TEACHER_FIX)
    same["action"] = 'press("14", "Enter")'
    final, info = TK.merge_fix(STUDENT, same)
    check(
        "re-issuing the student's action is NOT an action change", info["action_changed"] is False
    )
    check(
        "normalize_action ignores quotes/space",
        TK.normalize_action("press('14', 'Enter')") == TK.normalize_action('press("14",  "Enter")'),
    )
    check(
        "normalize_action still separates different actions",
        TK.normalize_action("click('21')") != TK.normalize_action("click('22')"),
    )


def test_mode_wiring():
    print("\n[e] mode wiring")
    check("takeover is a mode", "takeover" in EL.MODES)
    check("takeover is NOT teacher-forced", "takeover" not in EL.TEACHER_FORCED_MODES)
    check("takeover skips the localizer", "takeover" in EL.NO_LOCALIZE_MODES)
    check(
        "takeover uses teacher_relabel's failure substrate, not on_policy_distillation's all-episodes",
        "takeover" not in EL.ALL_EPISODE_MODES,
    )
    check("takeover needs a served student", "takeover" in EL.STUDENT_MODES)
    check("student_restyle's restyle path is untouched", EL.RESTYLE_MODES == ("student_restyle",))
    check(
        "teacher_relabel/style_matched/student_restyle/on_policy_distillation remain teacher-forced",
        EL.TEACHER_FORCED_MODES
        == (
            "webkd",
            "teacher_relabel",
            "on_policy_distillation",
            "style_matched",
            "student_restyle",
        ),
    )

    try:
        EL.StudentPolicy(base_url="")
        ok = False
    except AssertionError:
        ok = True
    except Exception:
        ok = True
    check("StudentPolicy without an endpoint raises", ok)


def test_judge_voice():
    print("\n[g] the fix must be a STEP, not a review of a step")
    q = TK.judge_query(STUDENT, EL.SECTION_CONTRACT)
    check("the query demands agent voice", "WRITE THE REPLACEMENT AS THE AGENT" in q)
    check("the query names the banned phrasing", '"the proposed"' in q and '"the student"' in q)
    reviewer = {
        "think": "The proposed action repeats a scroll that already failed, so it would loop.",
        "plan": "1. jump to the section anchor",
        "step": "1",
        "memory": "scrolling is not working",
        "action": "click('21')",
    }
    agent = {
        "think": "The Atoms section is linked in the contents, so I will jump straight to it.",
        "plan": "1. jump to the section anchor",
        "step": "1",
        "memory": "Atoms is reachable from the contents list",
        "action": "click('21')",
    }
    check("reviewer voice detected", TK.has_judge_voice(reviewer) is True)
    check("agent voice passes", TK.has_judge_voice(agent) is False)
    check(
        "detector ignores the action field",
        TK.has_judge_voice({"action": "click('the student')"}) is False,
    )
    check("empty sections are safe", TK.has_judge_voice({}) is False)

    recs = [
        {
            "phase": "takeover",
            "takeover": {"verdict": TK.FIX, "intervened": True, "judge_voice": True},
        },
        {
            "phase": "takeover",
            "takeover": {"verdict": TK.FIX, "intervened": True, "judge_voice": False},
        },
        {"phase": "takeover", "takeover": {"verdict": TK.OK, "intervened": False}},
    ]
    s = TK.summarize(recs)
    check("judge-voice counted", s["n_judge_voice"] == 1)
    check("rate is per intervention", abs(s["judge_voice_rate"] - 0.5) < 1e-9)
    check(
        "no interventions -> rate 0, not a division error",
        TK.summarize([{"phase": "takeover", "takeover": {"verdict": TK.OK}}])["judge_voice_rate"]
        == 0.0,
    )


def test_summarize():
    print("\n[f] cell diagnostics")
    recs = [
        {"phase": "takeover", "takeover": {"verdict": TK.OK, "intervened": False}},
        {
            "phase": "takeover",
            "takeover": {"verdict": TK.FIX, "intervened": True, "action_changed": True},
        },
        {
            "phase": "takeover",
            "takeover": {
                "verdict": TK.FIX,
                "intervened": True,
                "action_changed": False,
                "note": "verdict_inferred_from_action",
            },
        },
        {"phase": "takeover", "takeover": {"verdict": TK.UNPARSED, "intervened": False}},
        {"phase": "teacher_relabel", "takeover": {"verdict": TK.FIX, "intervened": True}},
    ]
    s = TK.summarize(recs)
    check("counts only takeover steps", s["n_steps"] == 4)
    check("intervention rate", abs(s["intervention_rate"] - 0.5) < 1e-9)
    check(
        "action-change rate is lower than intervention rate",
        abs(s["action_change_rate"] - 0.25) < 1e-9,
    )
    check("inferred verdicts counted", s["n_inferred"] == 1)
    check("unparsed counted", s["n_unparsed"] == 1)
    check("empty log is safe", TK.summarize([])["n_steps"] == 0)
