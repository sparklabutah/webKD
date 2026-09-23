"Regression tests for style behavior."

from __future__ import print_function

import json
import os
import subprocess
import sys

from webkd import trajectory as T
from webkd import style as S
from webkd import edit as EL


def check(name, cond):
    assert cond, name


LEGACY_THINK = (
    'The search box already contains "Rio". I should press Enter.\n'
    "</think>\n\n<plan>\n1. Press Enter to search\n2. Note the year\n</plan>\n\n"
    "<step>1</step>\n\n<memory>\nAbout to press Enter in the search box.\n</memory>\n\n"
    "<action>\npress('14', 'Enter')\n</action>"
)


def make_episode(root, task_id=104, n=3, reward=0.0):
    ep = os.path.join(root, "GenericAgentWithTraining_on_timewarp.%d_7" % task_id)
    td = os.path.join(ep, "training_data")
    os.makedirs(td)
    for i in range(n):
        user = (
            "# Instructions\n## Goal:\nFind the city.\n\n## AXTree:\n[14] textbox 'Search'\n"
            "## Extra instructions:\nx"
        )
        json.dump(
            {"role": "system", "content": "You are an agent."},
            open(os.path.join(td, "system_prompt_step_%d.json" % i), "w"),
        )
        json.dump(
            {"role": "user", "content": user},
            open(os.path.join(td, "user_prompt_step_%d.json" % i), "w"),
        )
        out = {"think": LEGACY_THINK, "action": "press('14', 'Enter')"}
        json.dump(out, open(os.path.join(td, "agent_output_step_%d.json" % i), "w"))
        open(os.path.join(td, "agent_output_step_%d.txt" % i), "w").write(LEGACY_THINK)
    json.dump(
        {"n_steps": n, "cum_reward": reward}, open(os.path.join(ep, "summary_info.json"), "w")
    )
    return ep


def test_bank_recovers_legacy_format(tmp):
    root = os.path.join(tmp, "p0")
    os.makedirs(root)
    make_episode(root, task_id=104, reward=0.0)
    make_episode(root, task_id=105, reward=1.0)
    make_episode(root, task_id=106, reward=1.0)

    sec = S.recover_sections(
        {
            "think": LEGACY_THINK,
            "action": "press('14', 'Enter')",
            "plan": "",
            "step": "",
            "memory": "",
        }
    )
    check("recover: plan pulled out of the packed think field", sec["plan"].startswith("1. Press"))
    check("recover: memory pulled out", "press Enter" in sec["memory"])
    check("recover: think no longer carries the other tags", "</think>" not in sec["think"])
    check("recover: executed action wins", sec["action"] == "press('14', 'Enter')")
    untouched = {"think": "plain reasoning", "plan": "p", "step": "1", "memory": "m", "action": "a"}
    check(
        "recover: modern sections pass through unchanged",
        S.recover_sections(dict(untouched)) == untouched,
    )

    bank = S.build_style_bank(root)
    check("bank: mines exemplars from the legacy dumps", len(bank) > 0)
    check("bank: prefers SOLVED episodes only", set(r["task_id"] for r in bank) == set([105, 106]))
    check(
        "bank: skips step 0 (its plan is the 'No plan yet' template)",
        all(r["step"] > 0 for r in bank),
    )

    picked = S.pick_exemplars(bank, 105, 2)
    check("bank: never shows the task being edited", all(r["task_id"] != 105 for r in picked))
    again = S.pick_exemplars(bank, 105, 2)
    check(
        "bank: deterministic for a given task",
        [(r["task_id"], r["step"]) for r in picked] == [(r["task_id"], r["step"]) for r in again],
    )

    block = S.style_block(picked)
    check("style block: renders the student's rationale", "<think>" in block and "<plan>" in block)
    check(
        "style block: carries NO <action> (an action is a decision, not a voice)",
        "<action>" not in block,
    )
    check(
        "style block: tells the teacher to keep its own decision",
        "only the wording changes" in block,
    )
    check("style block: empty when there are no exemplars", S.style_block([]) == "")


def test_prefix_exemplars(tmp):
    root = os.path.join(tmp, "p0_prefix")
    os.makedirs(root)
    ep = make_episode(root, task_id=110, n=4, reward=0.0)
    episode = T.load_episode(ep)

    got = S.prefix_exemplars(episode, t_star=3, k=2)
    check("prefix: rejects steps whose sections are not fully populated", got == [])
    for st in episode["steps"]:
        st["sections"] = S.recover_sections(st["sections"])
    got = S.prefix_exemplars(episode, t_star=3, k=2)
    check(
        "prefix: takes the 2 steps nearest t* from the replay prefix",
        [r["step"] for r in got] == [1, 2],
    )
    check("prefix: never crosses t*", all(r["step"] < 3 for r in got))


def test_style_block_reaches_the_teacher():
    seen = {}

    class FakeLLM(object):
        def __call__(self, system, user, max_tokens=None):
            seen["user"] = user
            return "<think>t</think><plan>p</plan><step>1</step><memory>m</memory><action>click('1')</action>"

    tp = EL.TeacherPolicy(privilege="none", style_text="\n[STYLE -- ...]\nexample")
    tp.llm = FakeLLM()
    tp("sys", "USER-PROMPT")
    check(
        "style_matched: the style block is appended to the teacher's query",
        "[STYLE" in seen["user"] and "USER-PROMPT" in seen["user"],
    )
    check(
        "style_matched: the format contract still precedes the style block (recency: tags first)",
        seen["user"].index("ALL FIVE tags") < seen["user"].index("[STYLE"),
    )

    tp2 = EL.TeacherPolicy(privilege="none")
    tp2.llm = FakeLLM()
    tp2("sys", "USER-PROMPT")
    check(
        "teacher_relabel unchanged: no style text when the arm does not ask for it",
        seen["user"] == "USER-PROMPT" + EL.SECTION_CONTRACT,
    )

    tp3 = EL.TeacherPolicy(privilege="oracle", gold_plan="GOLD-WALKTHROUGH", style_text="\n[STYLE]")
    tp3.llm = FakeLLM()
    tp3("sys", "USER-PROMPT")
    check(
        "style_matched+oracle: privilege and style compose",
        "GOLD-WALKTHROUGH" in seen["user"] and "[STYLE]" in seen["user"],
    )


def test_restyle_merge():
    teacher = {
        "think": "The article is missing, so I will search with autocomplete to find the "
        "real title instead of substituting another entity.",
        "plan": "1. search\n2. open\n3. report",
        "step": "1",
        "memory": "/wiki/Biophysics shows Article not found.",
        "action": "fill('14', 'Biophys', True)",
    }

    restyled = {
        "think": "Page not found. I will search with autocomplete.",
        "plan": "1. search\n2. open\n3. report",
        "step": "1",
        "memory": "Biophysics article missing.",
        "action": "fill('14', 'Biophys', True)",
    }
    final, info = S.merge_restyled(teacher, restyled)
    check(
        "student_restyle: the student's rationale is what gets trained on",
        final["think"] == restyled["think"],
    )
    check("student_restyle: an unchanged action is not flagged", not info["action_changed"])
    check("student_restyle: no override needed", not info["action_overridden"])
    check(
        "student_restyle: similarity reported below 1 when the text really changed",
        info["sim_think"] < 0.95,
    )
    check("student_restyle: length ratio tracks the shortening", 0 < info["len_ratio"] < 1)

    hijack = dict(restyled)
    hijack["action"] = "send_msg_to_user('Physics')"
    final, info = S.merge_restyled(teacher, hijack)
    check("student_restyle: a restyle that changes the action is detected", info["action_changed"])
    check(
        "student_restyle: the TEACHER's action is what survives (student_restyle must not become the self-edit arm)",
        final["action"] == teacher["action"] and info["action_overridden"],
    )

    spaced = dict(restyled)
    spaced["action"] = "fill('14',  'Biophys',  True)"
    _, info = S.merge_restyled(teacher, spaced)
    check(
        "student_restyle: whitespace-only action differences are not counted as changes",
        not info["action_changed"],
    )

    partial = {"think": "Page not found. I will search.", "action": teacher["action"]}
    final, info = S.merge_restyled(teacher, partial)
    check(
        "student_restyle: dropped <plan>/<memory>/<step> refilled from the teacher",
        final["plan"] == teacher["plan"]
        and final["memory"] == teacher["memory"]
        and final["step"] == teacher["step"],
    )
    check(
        "student_restyle: refills are reported",
        set(["plan", "memory", "step"]) <= set(info["filled_tags"]),
    )

    final, info = S.merge_restyled(teacher, dict(teacher))
    check(
        "student_restyle: an echoing restyle reports similarity ~1.0 (cell is really teacher_relabel)",
        info["sim_rationale"] > 0.99,
    )

    final, info = S.merge_restyled(teacher, {})
    check(
        "student_restyle: an unparseable restyle falls back to the teacher's sections",
        final == dict((k, teacher.get(k, "")) for k in T.TAGS),
    )
    check("student_restyle: the fallback is not silent", not info["restyle_parsed"])


def test_restyle_query_shape():
    teacher = {"think": "t", "plan": "p", "step": "1", "memory": "m", "action": "click('1')"}
    q = S.restyle_query(teacher)
    check("student_restyle prompt: shows the student the step it must re-voice", "click('1')" in q)
    check("student_restyle prompt: forbids changing the action", "Do not change the action" in q)
    check("student_restyle prompt: demands all five tags", "ALL FIVE tags" in q)


def test_restyle_policy_needs_an_endpoint():
    try:
        EL.RestylePolicy(base_url="", model="m")
        ok = False
    except AssertionError:
        ok = True
    check("student_restyle: refuses to run without a served frozen student", ok)


def test_modes_are_wired():
    check(
        "wiring: style_matched/student_restyle are selectable modes",
        "style_matched" in EL.MODES and "student_restyle" in EL.MODES,
    )
    check(
        "wiring: both are TEACHER-FORCED (original action executes, corrections never chain)",
        "style_matched" in EL.TEACHER_FORCED_MODES and "student_restyle" in EL.TEACHER_FORCED_MODES,
    )
    check(
        "wiring: both localize (they are teacher_relabel variants, not on_policy_distillation)",
        "style_matched" not in EL.NO_LOCALIZE_MODES
        and "student_restyle" not in EL.NO_LOCALIZE_MODES,
    )
    check(
        "wiring: only student_restyle runs a restyle pass",
        EL.RESTYLE_MODES == ("student_restyle",) and EL.STYLE_MODES == ("webkd", "style_matched"),
    )


def test_t_star_reuse(tmp):
    root = os.path.join(tmp, "p0_ts")
    os.makedirs(root)
    make_episode(root, task_id=104, n=5, reward=0.0)
    make_episode(root, task_id=107, n=5, reward=0.0)

    ref = os.path.join(tmp, "ref", "teacher_relabel_none", "v6")
    os.makedirs(os.path.join(ref, "task_104"))
    json.dump(
        {"task_id": 104, "t_star": 3, "mode": "teacher_relabel"},
        open(os.path.join(ref, "task_104", "properties.json"), "w"),
    )
    json.dump(
        {"mode": "teacher_relabel", "plan": [{"task_id": 107, "t_star": 2}]},
        open(os.path.join(ref, "edit_plan_shard0_of1.json"), "w"),
    )
    m = EL.load_t_star_map(ref)
    check("t*: read from both properties.json and the plan file", m == {104: 3, 107: 2})

    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    out_dir = os.path.join(tmp, "style_matched_out")
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "webkd.edit",
            "--episodes",
            root,
            "--out",
            out_dir,
            "--mode",
            "style_matched",
            "--t-star-from",
            ref,
            "--dry-run",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    p.communicate()
    plan_f = os.path.join(out_dir, "style_matched_none", "v6", "edit_plan_shard0_of1.json")
    check(
        "style_matched: dry-run plans the cell offline when t* is reused",
        p.returncode == 0 and os.path.isfile(plan_f),
    )
    if os.path.isfile(plan_f):
        plan = json.load(open(plan_f))["plan"]
        got = dict((it["task_id"], it["t_star"]) for it in plan)
        check(
            "style_matched: branch points match the reference cell exactly", got == {104: 3, 107: 2}
        )
        check(
            "style_matched: provenance of t* is recorded",
            all(it["t_star_src"] == "reused" for it in plan),
        )

    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "webkd.edit",
            "--episodes",
            root,
            "--out",
            os.path.join(tmp, "student_restyle_out"),
            "--mode",
            "student_restyle",
            "--t-star-from",
            ref,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out, _ = p.communicate()
    check(
        "student_restyle: aborts when no restyler endpoint is configured",
        p.returncode != 0 and b"restyle-base-url" in out,
    )
