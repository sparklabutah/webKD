"""Generate state-aligned teacher targets with optional student-style conditioning."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from dataclasses import dataclass
import sys
from pathlib import Path

from webkd import trajectory as T
from webkd import corrector as C
from webkd import style as S
from webkd import takeover as TK
from webkd import edit_offline as ET

logger = logging.getLogger(__name__)

MODES = (
    "webkd",
    "teacher_relabel",
    "expert_completion",
    "on_policy_distillation",
    "style_matched",
    "student_restyle",
    "takeover",
)
PRIVILEGES = ("none", "oracle")

TEACHER_FORCED_MODES = (
    "webkd",
    "teacher_relabel",
    "on_policy_distillation",
    "style_matched",
    "student_restyle",
)

NO_LOCALIZE_MODES = ("on_policy_distillation", "takeover")

ALL_EPISODE_MODES = ("on_policy_distillation",)

STUDENT_MODES = ("takeover",)

STYLE_MODES = ("webkd", "style_matched")
RESTYLE_MODES = ("student_restyle",)

SECTION_CONTRACT = (
    "\n\nRespond for THIS STEP ONLY, in EXACTLY this format, with ALL FIVE tags present:\n"
    "<think>\n...your reasoning, grounded ONLY in what is visible in the observation above...\n</think>\n"
    "<plan>\n...the running multi-step plan for finishing the task...\n</plan>\n"
    "<step>\n...a single INTEGER: which plan step the action below belongs to...\n</step>\n"
    "<memory>\n...facts worth carrying to later steps...\n</memory>\n"
    "<action>\n...exactly ONE browsergym action...\n</action>\n"
    "Emit exactly one <action>. Do not put an <action> inside <think>."
)

REQUIRED_TAGS = ("think", "plan", "step", "memory", "action")

ORACLE_PREFIX = "\n\n[REFERENCE PLAN -- available to you only; the student never sees this]\n"


def gold_plan_for(task_id):
    """The task's numbered gold walkthrough (includes the answer). Train tasks only."""
    return (ET.task_config(task_id) or {}).get("additional_instructions", "") or ""


class TeacherPolicy(object):
    """GPT-5.5 acting as the agent, prompted with the SAME rendered prompt the student sees."""

    def __init__(self, privilege="none", gold_plan="", max_tokens=None, style_text=""):
        self.privilege = privilege
        self.gold_plan = gold_plan or ""
        self.style_text = style_text or ""
        self.llm = C.LiveCorrector(max_tokens=max_tokens)

    def __call__(self, system_text, user_text):
        user = user_text
        if self.privilege == "oracle" and self.gold_plan:
            user = user + ORACLE_PREFIX + self.gold_plan

        out = self.llm(system_text, user + SECTION_CONTRACT + self.style_text)
        sections = T.parse_sections(out or "")
        return sections, (out or "")


class RestylePolicy(object):
    MAX_CONSECUTIVE_FAILURES = int(os.environ.get("WEBKD_RESTYLER_MAX_CONSEC_FAIL", "3"))

    def __init__(
        self, base_url="", model="", max_tokens=None, temperature=None, keep_teacher_action=True
    ):
        self._consecutive_failures = 0
        self._think_refills = 0
        self.base_url = base_url or os.environ.get("WEBKD_RESTYLER_BASE_URL", "")
        self.model = (
            model
            or os.environ.get("WEBKD_RESTYLER_MODEL", "")
            or os.environ.get("STUDENT_MODEL", "")
        )
        self.max_tokens = int(max_tokens or os.environ.get("WEBKD_RESTYLER_MAX_TOKENS", "2048"))
        self.temperature = float(
            temperature
            if temperature is not None
            else os.environ.get("WEBKD_RESTYLER_TEMPERATURE", "0.2")
        )
        self.keep_teacher_action = keep_teacher_action
        assert self.base_url, (
            "student_restyle needs WEBKD_RESTYLER_BASE_URL (the frozen student's vLLM endpoint)"
        )

        self.llm = C.LiveCorrector(
            provider="local",
            model=self.model,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            enable_thinking=True,
        )

    def __call__(self, system_text, user_text, teacher_sections):
        try:
            out = self.llm(system_text, user_text + S.restyle_query(teacher_sections))
            self._consecutive_failures = 0
        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    "student_restyle restyler unreachable at %s -- %d consecutive failures (last: %s). Aborting "
                    "rather than writing teacher text as if it were restyled: that silently turns "
                    "this cell into teacher_relabel. Check the served frozen student (a stranded VLLM::EngineCore "
                    "holding the GPU is the usual cause) and re-run the cell."
                    % (self.base_url, self._consecutive_failures, str(e)[:200])
                )
            logger.warning(
                "restyle call failed (%s) [%d/%d consecutive] -- keeping the teacher's "
                "sections for this step",
                e,
                self._consecutive_failures,
                self.MAX_CONSECUTIVE_FAILURES,
            )
            info = {
                "restyle_parsed": False,
                "restyle_error": str(e)[:200],
                "action_changed": False,
                "action_overridden": False,
                "filled_tags": list(T.TAGS),
                "sim_think": 1.0,
                "sim_rationale": 1.0,
                "len_ratio": 1.0,
            }
            return dict(teacher_sections), info, ""

        restyled = T.parse_sections(S.recover_open_think(out or ""))
        final, info = S.merge_restyled(
            teacher_sections, restyled, keep_teacher_action=self.keep_teacher_action
        )

        if "think" in (info.get("filled_tags") or []):
            self._think_refills += 1
            if self._think_refills in (1, 10, 50) or self._think_refills % 100 == 0:
                logger.warning(
                    "student_restyle: <think> refilled from the teacher on %d steps so far -- the "
                    "restyler is not producing a parseable think section",
                    self._think_refills,
                )
        return final, info, (out or "")


class StudentPolicy(object):
    """takeover: the SERVED student proposing its own step, from its own rendered prompt."""

    MAX_CONSECUTIVE_FAILURES = int(os.environ.get("WEBKD_STUDENT_MAX_CONSEC_FAIL", "3"))

    def __init__(self, base_url="", model="", max_tokens=None, temperature=None):
        self._consecutive_failures = 0
        self.base_url = base_url or os.environ.get("WEBKD_STUDENT_BASE_URL", "")
        self.model = (
            model
            or os.environ.get("WEBKD_STUDENT_MODEL", "")
            or os.environ.get("STUDENT_MODEL", "")
        )
        self.max_tokens = int(
            max_tokens
            or os.environ.get("WEBKD_STUDENT_MAX_TOKENS", os.environ.get("MAX_NEW_TOKENS", "2048"))
        )
        self.temperature = float(
            temperature
            if temperature is not None
            else os.environ.get("WEBKD_STUDENT_TEMPERATURE", "0.01")
        )
        assert self.base_url, (
            "takeover needs WEBKD_STUDENT_BASE_URL (the served student's vLLM endpoint)"
        )
        self.llm = C.LiveCorrector(
            provider="local",
            model=self.model,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            enable_thinking=True,
        )

    def __call__(self, system_text, user_text):
        """-> (sections, raw). sections may lack an action; the caller treats that as a broken step."""
        try:
            out = self.llm(system_text, user_text)
            self._consecutive_failures = 0
        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    "takeover student unreachable at %s -- %d consecutive failures (last: %s). "
                    "Aborting rather than letting the teacher write every step, which would turn "
                    "this cell into expert_completion. Check the served seed (a stranded VLLM::EngineCore holding "
                    "the GPU is the usual cause)."
                    % (self.base_url, self._consecutive_failures, str(e)[:200])
                )
            logger.warning(
                "student call failed (%s) [%d/%d consecutive]",
                e,
                self._consecutive_failures,
                self.MAX_CONSECUTIVE_FAILURES,
            )
            return dict((k, "") for k in T.TAGS), ""

        raw = S.recover_open_think(out or "")
        return T.parse_sections(raw), (out or "")


class TakeoverJudge(object):
    """takeover: the teacher judging the student's proposed step, and replacing it when it is wrong."""

    def __init__(self, privilege="none", gold_plan="", max_tokens=None):
        self.privilege = privilege
        self.gold_plan = gold_plan or ""
        self.llm = C.LiveCorrector(max_tokens=max_tokens)

    def __call__(self, system_text, user_text, student_sections, student_broken=False):
        """-> (verdict, sections, info, raw)."""
        user = user_text
        if self.privilege == "oracle" and self.gold_plan:
            user = user + ORACLE_PREFIX + self.gold_plan
        query = user + TK.judge_query(
            student_sections, SECTION_CONTRACT, student_broken=student_broken
        )
        try:
            out = self.llm(system_text, query)
        except Exception as e:
            logger.warning("judge call failed (%s) -- keeping the student's step", e)
            return TK.UNPARSED, {}, {"judge_error": str(e)[:200]}, ""
        verdict, sections, note = TK.parse_verdict(out or "")
        return verdict, sections, {"note": note}, (out or "")


def _sections_at(episode, step):
    for st in episode["steps"]:
        if st["step"] == step:
            return dict(st["sections"])
    return dict((k, "") for k in T.TAGS)


def _original_actions(episode):
    return [_sections_at(episode, st["step"]).get("action", "") for st in episode["steps"]]


try:
    from browsergym.experiments.agent import AgentInfo
    from agentlab.agents import dynamic_prompting as dp
    from agentlab.llm.llm_utils import SystemMessage
    from webkd.recording_agent import GenericAgentWithTraining, GenericAgentWithTrainingArgs
    from agentlab.agents.generic_agent.generic_agent_prompt import MainPrompt

    _AGENTLAB_OK = True
except Exception as _e:
    _AGENTLAB_OK, _AGENTLAB_ERR = False, _e

if _AGENTLAB_OK:

    class ReplayTeacherAgent(GenericAgentWithTraining):
        """Replays the original trajectory, then hands control to (or consults) the teacher at t*."""

        def __init__(
            self,
            chat_model_args,
            flags,
            max_retry=4,
            episode_dir="",
            t_star=0,
            mode="teacher_relabel",
            privilege="none",
            max_steps=30,
            episodes_root="",
            style_k=3,
            style_k_prefix=0,
            restyle_base_url="",
            restyle_model="",
            student_base_url="",
            student_model="",
            student_temperature=None,
        ):
            super().__init__(chat_model_args, flags, max_retry)
            self.episode = T.load_episode(episode_dir)
            self.t_star = int(t_star)
            self.mode = mode
            self.max_steps = int(max_steps)
            task_id = self.episode["task_id"] if self.episode else None

            self.style_exemplars = []
            style_text = ""
            if mode in STYLE_MODES and episodes_root:
                prefix = S.prefix_exemplars(self.episode, self.t_star, int(style_k_prefix))
                bank = S.build_style_bank(episodes_root)
                self.style_exemplars = prefix + S.pick_exemplars(
                    bank, task_id, max(0, int(style_k) - len(prefix))
                )
                style_text = S.style_block(self.style_exemplars)
                if not style_text:
                    raise ValueError(
                        "No student-style exemplars for task %s in %s" % (task_id, episodes_root)
                    )

            self.teacher = TeacherPolicy(
                privilege=privilege, gold_plan=gold_plan_for(task_id), style_text=style_text
            )
            self.restyler = (
                RestylePolicy(base_url=restyle_base_url, model=restyle_model)
                if mode in RESTYLE_MODES
                else None
            )

            self.student = (
                StudentPolicy(
                    base_url=student_base_url, model=student_model, temperature=student_temperature
                )
                if mode in STUDENT_MODES
                else None
            )
            self.judge = (
                TakeoverJudge(privilege=privilege, gold_plan=gold_plan_for(task_id))
                if mode in STUDENT_MODES
                else None
            )
            self.orig_actions = _original_actions(self.episode) if self.episode else []
            self._step = 0
            self.edit_log = []

        def _append_edit_log(self, rec):
            self.edit_log.append(rec)
            if self._training_data_dir is None:
                return
            try:
                path = Path(self._training_data_dir).parent / "edit_log.jsonl"
                with open(str(path), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                logger.warning("edit_log write failed at step %s: %s", rec.get("step"), e)

        def _save_training_output(self, ans_dict, step):
            sections = {k: (ans_dict.get(k) or "") for k in T.TAGS}
            try:
                T.write_agent_output(str(Path(self._training_data_dir).parent), step, sections)
            except Exception as e:
                logger.warning("dump failed at step %s: %s", step, e)

        def _render(self, obs):
            """Build the same system/user prompt the student would see at this step."""
            self.obs_history.append(obs)
            main_prompt = MainPrompt(
                action_set=self.action_set,
                obs_history=self.obs_history,
                actions=self.actions,
                memories=self.memories,
                thoughts=self.thoughts,
                previous_plan=self.plan,
                step=self.plan_step,
                flags=self.flags,
            )
            max_prompt_tokens, max_trunc_itr = self._get_maxes()
            system_prompt = SystemMessage(dp.SystemPrompt().prompt)
            human_prompt = dp.fit_tokens(
                shrinkable=main_prompt,
                max_prompt_tokens=max_prompt_tokens,
                model_name=self.chat_model_args.model_name,
                max_iterations=max_trunc_itr,
                additional_prompts=system_prompt,
            )
            return system_prompt, human_prompt

        def _commit(self, sections):
            """Push a step's sections into the agent history (what the NEXT step is conditioned on)."""
            self.actions.append(sections.get("action", ""))
            self.memories.append(sections.get("memory", "") or None)
            self.thoughts.append(sections.get("think", "") or None)
            if sections.get("plan"):
                self.plan = sections["plan"]
            if sections.get("step"):
                raw_step = str(sections["step"]).strip()
                try:
                    self.plan_step = int(raw_step)
                except (TypeError, ValueError):
                    self.plan_step = raw_step

        def get_action(self, obs):
            k = self._step
            self._step += 1
            system_prompt, human_prompt = self._render(obs)
            if self._training_data_dir is not None:
                self._save_training_prompts(system_prompt, human_prompt, k)

            orig = _sections_at(self.episode, k) if self.episode else {}
            in_replay = k < self.t_star
            teacher_out = None
            restyle_info = None
            teacher_sections = None
            takeover_info = None

            if self.mode in STUDENT_MODES:
                student_sections, student_raw = self.student(str(system_prompt), str(human_prompt))
                student_broken = not (student_sections.get("action") or "").strip()
                verdict, fix, jinfo, judge_raw = self.judge(
                    str(system_prompt),
                    str(human_prompt),
                    student_sections,
                    student_broken=student_broken,
                )
                teacher_out = judge_raw
                minfo = {}
                if verdict == TK.FIX:
                    dumped, minfo = TK.merge_fix(student_sections, fix)
                elif student_broken:
                    self._append_edit_log(
                        {
                            "step": k,
                            "phase": self.mode,
                            "abort": True,
                            "takeover": {
                                "verdict": verdict,
                                "intervened": False,
                                "student_broken": True,
                                "note": (jinfo or {}).get("note", ""),
                            },
                        }
                    )
                    return None, AgentInfo(
                        think="takeover: student produced no action and the teacher did not fix it",
                        chat_messages=[],
                        stats={},
                        extra_info={"edit_abort": True},
                    )
                else:
                    dumped = dict(student_sections)
                executed = dumped.get("action", "")
                self._commit(dumped)
                takeover_info = {
                    "verdict": verdict,
                    "intervened": verdict == TK.FIX,
                    "action_changed": bool(minfo.get("action_changed")),
                    "student_broken": student_broken,
                    "student_action": (student_sections.get("action") or "").strip()[:300],
                    "student_missing_tags": [
                        t for t in REQUIRED_TAGS if not (student_sections.get(t) or "").strip()
                    ],
                    "filled_from_student": minfo.get("filled_from_student", []),
                    "judge_voice": TK.has_judge_voice(dumped) if verdict == TK.FIX else False,
                    "sim_think": minfo.get("sim_think"),
                    "note": (jinfo or {}).get("note", ""),
                    "judge_error": (jinfo or {}).get("judge_error", ""),
                    "diverged_from_p0": TK.normalize_action(dumped.get("action"))
                    != TK.normalize_action(orig.get("action")),
                    "student_raw_chars": len(student_raw or ""),
                }
            elif in_replay:
                dumped = orig
                executed = orig.get("action", "")
                self._commit(orig)
            else:
                sections, raw = self.teacher(str(system_prompt), str(human_prompt))
                teacher_out = raw
                if not sections.get("action"):
                    if self.mode in TEACHER_FORCED_MODES:
                        sections = dict(orig)
                    else:
                        return None, AgentInfo(
                            think="teacher emitted no action",
                            chat_messages=[],
                            stats={},
                            extra_info={"edit_abort": True},
                        )
                elif self.restyler is not None:
                    teacher_sections = dict(sections)
                    sections, restyle_info, restyle_raw = self.restyler(
                        str(system_prompt), str(human_prompt), teacher_sections
                    )
                    teacher_out = (teacher_out or "") + "\n\n[RESTYLED]\n" + (restyle_raw or "")
                if self.mode in TEACHER_FORCED_MODES:
                    dumped = sections
                    executed = orig.get("action", "")
                    self._commit(orig)
                else:
                    dumped = sections
                    executed = sections.get("action", "")
                    self._commit(sections)

            if self._training_data_dir is not None:
                self._save_training_output(dumped, k)

            rec = {
                "step": k,
                "phase": "replay" if in_replay else self.mode,
                "orig_action": orig.get("action", ""),
                "written_action": dumped.get("action", ""),
                "executed_action": executed,
                "changed": bool(
                    not in_replay and dumped.get("action", "") != orig.get("action", "")
                ),
                "has_all_sections": all(dumped.get(t) for t in REQUIRED_TAGS),
                "missing_tags": [t for t in REQUIRED_TAGS if not dumped.get(t)],
            }
            if self.mode in STYLE_MODES and not in_replay:
                rec["n_style_exemplars"] = len(self.style_exemplars)
            if restyle_info is not None:
                rec["restyle"] = restyle_info
                rec["teacher_action"] = (teacher_sections or {}).get("action", "")
            if takeover_info is not None:
                rec["takeover"] = takeover_info
            self._append_edit_log(rec)

            if self.mode == "expert_completion" and self._step > self.max_steps:
                executed = executed or ""
            return executed, AgentInfo(
                think=dumped.get("think", ""),
                chat_messages=[],
                stats={},
                extra_info={"teacher_raw": teacher_out},
            )

    @dataclass
    class ReplayTeacherAgentArgs(GenericAgentWithTrainingArgs):
        episode_dir: str = ""
        t_star: int = 0
        mode: str = "teacher_relabel"
        privilege: str = "none"
        max_steps: int = 30

        episodes_root: str = ""
        style_k: int = 3
        style_k_prefix: int = 0
        restyle_base_url: str = ""
        restyle_model: str = ""

        student_base_url: str = ""
        student_model: str = ""
        student_temperature: float = 0.01

        def __post_init__(self):
            try:
                super().__post_init__()
            except Exception:
                pass
            self.agent_name = f"ReplayTeacher-{self.mode}-{self.privilege}"

        def make_agent(self):
            return ReplayTeacherAgent(
                chat_model_args=self.chat_model_args,
                flags=self.flags,
                max_retry=self.max_retry,
                episode_dir=self.episode_dir,
                t_star=self.t_star,
                mode=self.mode,
                privilege=self.privilege,
                max_steps=self.max_steps,
                episodes_root=self.episodes_root,
                style_k=self.style_k,
                style_k_prefix=self.style_k_prefix,
                restyle_base_url=self.restyle_base_url,
                restyle_model=self.restyle_model,
                student_base_url=self.student_base_url,
                student_model=self.student_model,
                student_temperature=self.student_temperature,
            )


def localize_t_star(episode, task, privilege):
    """Reuse the Tier-1 localizer (GPT-5.5 names the first erroneous step)."""
    teacher = C.LiveCorrector()
    priv = "gold" if privilege == "oracle" else "none"
    return ET.localize(episode, task, teacher, priv)


def load_t_star_map(cell_dir):
    """{task_id: t*} recorded by a previously-built cell, for --t-star-from."""
    out = {}
    cell = Path(cell_dir)
    for pf in sorted(cell.glob("edit_plan_shard*.json")):
        try:
            data = json.load(open(str(pf)))
        except Exception:
            continue
        for item in data.get("plan", []):
            if item.get("task_id") is not None and item.get("t_star") is not None:
                out[int(item["task_id"])] = int(item["t_star"])
    for pf in sorted(cell.glob("task_*/properties.json")):
        try:
            data = json.load(open(str(pf)))
        except Exception:
            continue
        if data.get("task_id") is not None and data.get("t_star") is not None:
            out[int(data["task_id"])] = int(data["t_star"])
    return out


def _takeover_props(study_dir, args):
    """Roll this episode's edit_log.jsonl up into the takeover diagnostics + its LIVE outcome."""
    recs, out = [], {}
    for lf in sorted(Path(study_dir).glob("*/edit_log.jsonl")):
        for line in open(str(lf)):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    out.update(TK.summarize(recs))

    for sf in sorted(Path(study_dir).glob("*/summary_info.json")):
        try:
            s = json.load(open(str(sf)))
        except Exception:
            continue
        out["live_reward"] = s.get("cum_reward")
        out["live_n_steps"] = s.get("n_steps")
        out["live_err"] = s.get("err_msg") or ""
    out["student"] = {
        "base_url": args.student_base_url,
        "model": args.student_model or os.environ.get("STUDENT_MODEL", ""),
        "temperature": args.student_temperature,
    }
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Live-environment trajectory editing (teacher_relabel/expert_completion/on_policy_distillation, the style_matched/student_restyle style arms, takeover)."
    )
    ap.add_argument(
        "--episodes",
        required=True,
        help="root of P-0 episode dirs (teacher_relabel/expert_completion use the FAILURES; on_policy_distillation uses ALL episodes)",
    )
    ap.add_argument("--out", required=True, help="output edited-episode tree")
    ap.add_argument("--mode", default="webkd", choices=MODES)
    ap.add_argument("--privilege", default="none", choices=PRIVILEGES)
    ap.add_argument("--version", default="6")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=30, help="expert_completion completion cap")
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="AgentLab n_jobs per study (keep 1; parallelize via --num-shards)",
    )

    ap.add_argument(
        "--shard", type=int, default=0, help="this process's shard index [0, num-shards)"
    )
    ap.add_argument(
        "--num-shards", type=int, default=1, help="total shards (parallel edit processes)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="localize + plan the cells without importing agentlab or touching a browser",
    )

    ap.add_argument(
        "--style-k",
        type=int,
        default=int(os.environ.get("WEBKD_STYLE_K", "3")),
        help="style_matched: in-context exemplars of the student's own writing per episode",
    )
    ap.add_argument(
        "--style-k-prefix",
        type=int,
        default=int(os.environ.get("WEBKD_STYLE_K_PREFIX", "0")),
        help="style_matched: how many of those come from THIS episode's replay prefix (steps < t*) "
        "rather than the bank; the prefix is already in the teacher's history, so 0",
    )
    ap.add_argument(
        "--style-bank",
        default=os.environ.get("WEBKD_STYLE_BANK", ""),
        help="style_matched: root to mine exemplars from (default: --episodes, i.e. the student's "
        "own P-0 rollout)",
    )
    ap.add_argument(
        "--restyle-base-url",
        default=os.environ.get("WEBKD_RESTYLER_BASE_URL", ""),
        help="student_restyle: OpenAI-compatible endpoint of the FROZEN student, e.g. "
        "http://localhost:8300/v1",
    )
    ap.add_argument(
        "--restyle-model",
        default=os.environ.get("WEBKD_RESTYLER_MODEL", ""),
        help="student_restyle: served-model-name at --restyle-base-url (default: $STUDENT_MODEL)",
    )

    ap.add_argument(
        "--student-base-url",
        default=os.environ.get("WEBKD_STUDENT_BASE_URL", ""),
        help="takeover: OpenAI-compatible endpoint of the SERVED student that acts, e.g. "
        "http://localhost:8400/v1",
    )
    ap.add_argument(
        "--student-model",
        default=os.environ.get("WEBKD_STUDENT_MODEL", ""),
        help="takeover: served-model-name at --student-base-url (default: $STUDENT_MODEL)",
    )
    ap.add_argument(
        "--student-temperature",
        type=float,
        default=float(os.environ.get("WEBKD_STUDENT_TEMPERATURE", "0.01")),
        help="takeover: sampling temperature for the student's proposals; 0.01 matches "
        "benchmarkGeneralNoPlan.py, i.e. the P-0 rollout this arm is read against",
    )
    ap.add_argument(
        "--substrate",
        default=os.environ.get("WEBKD_SUBSTRATE", ""),
        choices=["", "failed", "all"],
        help="which P-0 episodes to run: 'failed' (default for teacher_relabel/expert_completion/style_matched/student_restyle/takeover) or "
        "'all' (default for on_policy_distillation). takeover defaults to 'failed' so it shares teacher_relabel's "
        "task set and the arms differ only in the control loop",
    )
    ap.add_argument(
        "--t-star-from",
        default=os.environ.get("WEBKD_T_STAR_FROM", ""),
        help="reuse the branch points of an already-built cell dir (e.g. "
        "data/edits/teacher_relabel_none/v6) instead of re-localizing; keeps a "
        "style arm step-for-step comparable with its teacher_relabel reference",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=os.environ.get("WEBKD_RESUME", "0") not in ("", "0", "false", "False"),
        help="skip tasks this cell has ALREADY edited (task_<id>/properties.json "
        "exists). Off by default so an intentional rebuild still rebuilds. "
        "Without it a re-run after an interruption re-pays the teacher for every "
        "finished episode AND leaves two agentlab study dirs under the same "
        "task_<id>, which convert.py would emit as duplicate "
        "conversations. properties.json is written only after study.run() "
        "returns, so a half-finished episode has none and is correctly redone.",
    )
    args = ap.parse_args()
    assert 0 <= args.shard < args.num_shards, "shard must be in [0, num-shards)"
    if args.mode in RESTYLE_MODES and not args.dry_run:
        assert args.restyle_base_url, (
            "--mode %s needs --restyle-base-url / WEBKD_RESTYLER_BASE_URL: the frozen student must be "
            "served for the restyle pass" % args.mode
        )
    if args.mode in STUDENT_MODES and not args.dry_run:
        assert args.student_base_url, (
            "--mode %s needs --student-base-url / WEBKD_STUDENT_BASE_URL: unlike every other mode the "
            "acting policy is a LIVE forward pass, not a recorded action" % args.mode
        )

    substrate = args.substrate or ("all" if args.mode in ALL_EPISODE_MODES else "failed")
    iter_eps = ET.iter_all_episodes if substrate == "all" else ET.iter_failed_episodes
    all_eps = list(iter_eps(args.episodes, args.limit))

    eps = [e for i, e in enumerate(all_eps) if i % args.num_shards == args.shard]
    print(
        "found %d episodes (%s); shard %d/%d handles %d"
        % (len(all_eps), substrate, args.shard, args.num_shards, len(eps))
    )
    out_root = Path(args.out) / ("%s_%s" % (args.mode, args.privilege)) / ("v%s" % args.version)
    out_root.mkdir(parents=True, exist_ok=True)

    t_star_map = load_t_star_map(args.t_star_from) if args.t_star_from else {}
    if args.t_star_from:
        print("reusing t* from %s (%d tasks)" % (args.t_star_from, len(t_star_map)))

    plan = []
    for ep_dir in eps:
        episode = T.load_episode(ep_dir)
        if episode is None or not episode["steps"]:
            continue
        task = ET.task_config(episode["task_id"])
        solved = episode["reward"] is not None and episode["reward"] >= 1.0
        if episode["task_id"] in t_star_map and args.mode not in NO_LOCALIZE_MODES:
            t_star = t_star_map[episode["task_id"]]
            t_star_src = "reused"
        elif args.mode in STUDENT_MODES:
            t_star = 0
        elif args.mode in NO_LOCALIZE_MODES:
            t_star = len(episode["steps"]) if solved else 0
        else:
            t_star = localize_t_star(episode, task, args.privilege)
            t_star_src = "localized"
        if args.mode in STUDENT_MODES:
            t_star_src = "live-from-step0"
        elif args.mode in NO_LOCALIZE_MODES:
            t_star_src = "step0" if not solved else "solved-kept"
        kept_as_is = args.mode in ALL_EPISODE_MODES and args.mode in NO_LOCALIZE_MODES and solved
        plan.append(
            {
                "episode_dir": ep_dir,
                "task_id": episode["task_id"],
                "t_star": t_star,
                "n_steps": len(episode["steps"]),
                "src_reward": episode["reward"],
                "t_star_src": t_star_src,
                "relabeled": not kept_as_is,
            }
        )
        print(
            "  task %s  T=%d  t*=%d (%s)  src_reward=%s%s"
            % (
                episode["task_id"],
                len(episode["steps"]),
                t_star,
                t_star_src,
                episode["reward"],
                "  [solved -> kept as-is, no teacher]" if kept_as_is else "",
            )
        )

    plan_file = out_root / ("edit_plan_shard%d_of%d.json" % (args.shard, args.num_shards))
    with open(plan_file, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "privilege": args.privilege,
                "shard": args.shard,
                "num_shards": args.num_shards,
                "plan": plan,
            },
            f,
            indent=2,
        )
    print("wrote edit plan -> %s" % plan_file)

    if args.dry_run:
        print("--dry-run: stopping before the live rollouts")
        return

    from agentlab.experiments.study import make_study
    from browsergym.experiments.benchmark.configs import DEFAULT_BENCHMARKS
    from agentlab.agents.generic_agent import AGENT_LLAMA3_70B
    from agentlab.llm.chat_api import SelfHostedModelArgs

    assert _AGENTLAB_OK, "agentlab import failed at module load: %r" % (_AGENTLAB_ERR,)

    flags = AGENT_LLAMA3_70B.flags.copy()
    flags.obs.use_error_logs = True
    flags.obs.use_history = True
    flags.obs.use_think_history = True
    flags.obs.use_past_error_logs = True
    flags.use_thinking = True
    flags.use_memory = True
    flags.use_plan = True
    flags.use_criticise = False
    flags.extra_instructions = (
        "IMPORTANT: only navigate within the TimeWarp environment (WIKI %s, NEWS %s, SHOP %s).\n"
        "Output EXACTLY ONE action per response, in an <action> tag at the END, outside <think>."
        % (
            os.environ.get("TW_WIKI", ""),
            os.environ.get("TW_NEWS", ""),
            os.environ.get("TW_WEBSHOP", ""),
        )
    )

    model_args = SelfHostedModelArgs(
        model_name=os.environ.get("WEBKD_STUDENT_MODEL") or args.student_model or "student-model",
        max_total_tokens=int(os.environ.get("MAX_MODEL_LEN", "40960")),
        max_input_tokens=int(os.environ.get("MAX_MODEL_LEN", "40960")) - 2048,
        max_new_tokens=2048,
        backend="vllm",
        temperature=0.01,
        n_retry_server=1,
    )

    bench = DEFAULT_BENCHMARKS["timewarp"](n_repeats=1).subset_from_split("train")
    by_task = {}
    for env_args in bench.env_args_list:
        if env_args.task_kwargs is None:
            env_args.task_kwargs = {}

        env_args.task_kwargs["with_additional_instructions"] = False
        by_task[env_args.task_name] = env_args

    n_done = n_resumed = 0
    for item in plan:
        tname = "timewarp.%s" % item["task_id"]
        env_args = by_task.get(tname)
        if env_args is None:
            print("  [skip] %s not in train split" % tname)
            continue
        if args.resume and (out_root / ("task_%s" % item["task_id"]) / "properties.json").is_file():
            n_resumed += 1
            print("  [resume] task %s already edited -- skipping" % item["task_id"])
            continue
        agent_args = ReplayTeacherAgentArgs(
            chat_model_args=model_args,
            flags=flags,
            episode_dir=item["episode_dir"],
            t_star=item["t_star"],
            mode=args.mode,
            privilege=args.privilege,
            max_steps=args.max_steps,
            episodes_root=(args.style_bank or args.episodes),
            style_k=args.style_k,
            style_k_prefix=args.style_k_prefix,
            restyle_base_url=args.restyle_base_url,
            restyle_model=args.restyle_model,
            student_base_url=args.student_base_url,
            student_model=args.student_model,
            student_temperature=args.student_temperature,
        )

        single = copy.copy(bench)
        single.env_args_list = [env_args]
        study = make_study(
            benchmark=single,
            agent_args=[agent_args],
            comment="edit-%s/%s task %s" % (args.mode, args.privilege, item["task_id"]),
        )
        study.dir = out_root / ("task_%s" % item["task_id"])
        try:
            study.run(n_jobs=1, parallel_backend="sequential")
            n_done += 1
        except Exception as e:
            print("  [fail] task %s: %s" % (item["task_id"], e))
            continue
        with open(study.dir / "properties.json", "w") as f:
            props = {
                "mode": args.mode,
                "privilege": args.privilege,
                "t_star": item["t_star"],
                "task_id": item["task_id"],
                "src": item["episode_dir"],
                "src_reward": item.get("src_reward"),
                "t_star_src": item.get("t_star_src"),
                "relabeled": item.get("relabeled", True),
                "grounded": True,
                "live_env": True,
            }
            if args.mode in STYLE_MODES:
                props["style"] = {
                    "k": args.style_k,
                    "k_prefix": args.style_k_prefix,
                    "bank": args.style_bank or args.episodes,
                }
            if args.mode in RESTYLE_MODES:
                props["restyle"] = {
                    "base_url": args.restyle_base_url,
                    "model": args.restyle_model or os.environ.get("STUDENT_MODEL", ""),
                    "temperature": os.environ.get("WEBKD_RESTYLER_TEMPERATURE", "0.2"),
                    "keep_teacher_action": True,
                }
            if args.mode in STUDENT_MODES:
                props["takeover"] = _takeover_props(study.dir, args)
            json.dump(props, f, indent=2)

    print(
        "\n%s/%s: completed %d/%d episodes (%d skipped by --resume) -> %s"
        % (args.mode, args.privilege, n_done, len(plan), n_resumed, out_root)
    )

    if n_done + n_resumed != len(plan):
        print("ERROR: one or more episodes failed -- failing shard")
        sys.exit(1)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
