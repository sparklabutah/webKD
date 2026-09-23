"Regression tests for core behavior."

from __future__ import print_function

import json
import os
import subprocess
import sys

from webkd import trajectory as T
from webkd import corrector as C
from webkd import grounding as S
from webkd import realizability as R
from webkd import edit_offline as E
from webkd import judge_reliability as J


def check(name, cond):
    assert cond, name


def make_fake_episode(root, task_id=104, n=3, answer="Toronto", reward=0.0):
    "Build a minimal on-disk episode with the real dump schema (old agent_output format)."
    ep = os.path.join(root, "GenericAgentWithTraining_on_timewarp.%d_7" % task_id)
    td = os.path.join(ep, "training_data")
    os.makedirs(td)
    for i in range(n):
        ax = "## AXTree:\n[26] searchbox 'Search'\n[14] link 'Canada'\npopulation 3 million Toronto\n"
        user = (
            "# Instructions\n## Goal:\nFind the largest city.\n\n%s## Extra instructions:\nx" % ax
        )
        json.dump(
            {"role": "system", "content": "You are an agent."},
            open(os.path.join(td, "system_prompt_step_%d.json" % i), "w"),
        )
        json.dump(
            {"role": "user", "content": user},
            open(os.path.join(td, "user_prompt_step_%d.json" % i), "w"),
        )
        if i < n - 1:
            out = {"think": "Search for it.", "action": "fill('26', 'Canada')"}
        else:
            out = {"think": "Answer now.", "action": "send_msg_to_user('%s')" % answer}
        json.dump(out, open(os.path.join(td, "agent_output_step_%d.json" % i), "w"))
        open(os.path.join(td, "agent_output_step_%d.txt" % i), "w").write(
            "Thinking:\n%s\n\nAction:\n%s" % (out["think"], out["action"])
        )
    json.dump(
        {"n_steps": n, "cum_reward": reward, "err_msg": None},
        open(os.path.join(ep, "summary_info.json"), "w"),
    )
    return ep


def test_traj_roundtrip(tmp):
    ep_dir = make_fake_episode(tmp)
    ep = T.load_episode(ep_dir)
    check("traj: loads episode", ep is not None and len(ep["steps"]) == 3)
    check("traj: task_id parsed", ep["task_id"] == 104)
    check("traj: axtree extracted", "population 3 million" in ep["steps"][0]["axtree"])
    check("traj: submitted answer", T.submitted_answer(ep) == "Toronto")

    sec = {"think": "abc", "plan": "", "step": "", "memory": "", "action": "click('1')"}
    rt = T.parse_sections(T.sections_to_content(sec))
    check("traj: tag render round-trip", rt["think"] == "abc" and rt["action"] == "click('1')")


def test_editor_pipeline_flows_through_real_tools(tmp):
    "Editor output must be consumed by the REAL convert.py and filter_episodes.py."
    src_root = os.path.join(tmp, "src")
    os.makedirs(src_root)
    ep_dir = make_fake_episode(src_root)
    ep = T.load_episode(ep_dir)
    C.set_corrector(C.MockCorrector(answer="Toronto"))
    corr = C.get_corrector()
    task = {
        "intent": "Find the largest city.",
        "additional_instructions": "1. Search. 2. Send: Toronto.",
    }
    t_star = E.localize(ep, task, corr, "none")
    new_steps = E.edit(ep, t_star, task, corr, "none", "dense")
    out_ep = os.path.join(tmp, "edits", "v6", os.path.basename(ep_dir))
    props = E.write_edited_episode(
        ep,
        out_ep,
        t_star,
        new_steps,
        "dense",
        {"corrector": "teacher", "privilege": "none", "task_id": 104},
    )
    check("editor: final action is send_msg", props["final_is_send_msg"] is True)
    check(
        "editor: wrote agent_output json+txt",
        os.path.isfile(os.path.join(out_ep, "training_data", "agent_output_step_%d.json" % t_star)),
    )

    sgpt = os.path.join(tmp, "out.json")
    edits_root = os.path.join(tmp, "edits")
    rc = subprocess.call(
        [
            sys.executable,
            "-m",
            "webkd.convert",
            edits_root,
            "--output",
            sgpt,
            "--think",
            "--plan",
            "--memory",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    ok = rc == 0 and os.path.isfile(sgpt)
    data = json.load(open(sgpt)) if ok else []
    check("convert2sgpt: runs & emits samples", ok and len(data) >= 1)
    if data:
        s = data[0]
        good = (
            "system" in s
            and isinstance(s["conversations"], list)
            and s["conversations"][0]["from"] == "human"
            and s["conversations"][1]["from"] == "gpt"
            and "<action>" in s["conversations"][1]["value"]
        )
        check("convert2sgpt: sharegpt schema + action tag", good)
        check(
            "convert2sgpt: send_msg survived",
            any(
                "send_msg_to_user" in c["value"]
                for s in data
                for c in s["conversations"]
                if c["from"] == "gpt"
            ),
        )

    filt = os.path.join(tmp, "edits_filtered")
    rc2 = subprocess.call(
        [
            sys.executable,
            "-m",
            "webkd.filter_episodes",
            edits_root,
            "--final-answer",
            "--no-error",
            "-o",
            filt,
            "--copy",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    kept = os.path.isdir(filt) and any(
        os.path.isfile(os.path.join(dp, "summary_info.json")) for dp, _, _ in os.walk(filt)
    )
    check("filter: --final-answer keeps the edited episode", rc2 == 0 and kept)


def test_scf_flags_hindsight():
    ax = "## AXTree:\n[26] link 'Toronto' population 3 million\n"
    grounded = "The page shows Toronto has population 3 million."
    hindsight = "The answer is Ottawa with 1 million; the next page will confirm Vancouver 2.5m."
    g = S.scf_score(grounded, ax)
    h = S.scf_score(hindsight, ax)
    check("scf: grounded reasoning is consistent", g["state_consistent"] is True)
    check("scf: hindsight reasoning is flagged", h["state_consistent"] is False)
    check("scf: grounded fraction ordering", g["grounded_fraction"] > h["grounded_fraction"])
    check("scf: number grounding", S.is_grounded("3 million", S._norm(ax)))
    check("scf: ungrounded entity caught", not S.is_grounded("Ottawa", S._norm(ax)))


def test_realizability_math():
    check("nll: empty -> None", R.mean_nll([]) is None)
    high_conf = R.mean_nll([-0.1, -0.05, -0.2])
    low_conf = R.mean_nll([-4.0, -3.5, -5.0])
    check("nll: finite", high_conf is not None and low_conf is not None)
    check("nll: monotone (off-policy scores higher NLL)", low_conf > high_conf)
    lp = R.extract_token_logprobs({"content": [{"logprob": -0.5}, {"logprob": -1.5}]})
    check("nll: extract from dict logprobs", abs(R.mean_nll(lp) - 1.0) < 1e-9)


def test_privilege_discipline():
    ep = {
        "steps": [
            {
                "step": 0,
                "goal": "Find the largest city.",
                "sections": {"plan": "step A then B", "think": "", "memory": "", "action": ""},
                "axtree": "## AXTree:\nx",
                "user_text": "## Goal:\nFind the largest city.",
            }
        ]
    }
    task = {
        "intent": "Find the largest city.",
        "additional_instructions": "1. Type Canada. 2. Send a message to the user: 'Toronto'.",
    }
    _, u_none = E.build_context(ep, task, "none")
    _, u_gold = E.build_context(ep, task, "gold")
    _, u_self = E.build_context(ep, task, "self")
    _, u_na = E.build_context(ep, task, "gold_noans")
    check("privilege: none has NO plan leak", "Type Canada" not in u_none)
    check("privilege: gold injects the plan", "Type Canada" in u_gold)
    check("privilege: self injects own plan", "step A then B" in u_self)
    check(
        "privilege: gold_noans strips the answer line", "Toronto" not in u_na and "Canada" in u_na
    )


def test_pseudo_gold_localization():
    steps = J.plan_steps("1. Type Physics. 2. Read the section. 3. Send answer.")
    check("localize: plan split", len(steps) == 3)


def test_on_policy_distillation_mode(tmp):
    "Baseline 3 (on-policy distillation): relabel EVERY episode from t*=0, no localizer call."
    root = os.path.join(tmp, "on_policy_distillation_src")
    os.makedirs(root)
    make_fake_episode(root, task_id=104, reward=0.0)
    make_fake_episode(root, task_id=105, reward=1.0)

    check(
        "on_policy_distillation: iter_failed_episodes keeps only the failure",
        len(list(E.iter_failed_episodes(root))) == 1,
    )
    check(
        "on_policy_distillation: iter_all_episodes keeps the success too",
        len(list(E.iter_all_episodes(root))) == 2,
    )
    check(
        "on_policy_distillation: iter_all_episodes honours limit",
        len(list(E.iter_all_episodes(root, 1))) == 1,
    )

    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    out_dir = os.path.join(tmp, "on_policy_distillation_out")
    rc = subprocess.call(
        [
            sys.executable,
            "-m",
            "webkd.edit",
            "--episodes",
            root,
            "--out",
            out_dir,
            "--mode",
            "on_policy_distillation",
            "--dry-run",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    plan_f = os.path.join(out_dir, "on_policy_distillation_none", "v6", "edit_plan_shard0_of1.json")
    check(
        "on_policy_distillation: dry-run plans the cell offline", rc == 0 and os.path.isfile(plan_f)
    )
    if os.path.isfile(plan_f):
        plan = json.load(open(plan_f))["plan"]
        check("on_policy_distillation: plans every episode", len(plan) == 2)
        check(
            "on_policy_distillation: sidecar carries the source reward",
            sorted(p["src_reward"] for p in plan) == [0.0, 1.0],
        )
        fail = [p for p in plan if p["src_reward"] == 0.0][0]
        solved = [p for p in plan if p["src_reward"] == 1.0][0]

        check(
            "on_policy_distillation: unsolved episode relabeled from step 0",
            fail["t_star"] == 0 and fail["relabeled"] is True,
        )
        check(
            "on_policy_distillation: solved episode kept as-is (t* past the end, not relabeled)",
            solved["t_star"] == solved["n_steps"] and solved["relabeled"] is False,
        )
