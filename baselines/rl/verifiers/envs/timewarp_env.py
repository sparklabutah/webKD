"""Multi-turn policy-gradient baseline for web agents."""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.prompts.timewarp_prompts import (
    build_initial_prompt,
    build_observation_text,
    get_timewarp_flags,
    is_answer_action,
    make_obs_preprocessor,
    parse_action,
    parse_memory,
    parse_plan,
)
from verifiers.utils.timewarp_data import build_timewarp_dataset

logger = logging.getLogger("verifiers.envs.TimeWarpEnv")


class TimeWarpEnv(MultiStepEnv):
    def __init__(
        self,
        max_steps: int = 10,
        seed: int = 42,
        mask_thinking: bool = False,
        max_obs_tokens: int = 4096,
        max_tokens_per_turn: int = 1024,
        env_timeout_ms: int = 15000,
        step_timeout_s: float = 180.0,
        tw_wiki: str | None = None,
        tw_news: str | None = None,
        tw_webshop: str | None = None,
        tw_home: str | None = None,
        mask_env_response: bool = True,
        sites_filter: list[str] | None = None,
        task_limit: int = -1,
        rollout_mode: str | None = None,
        **kwargs,
    ):

        super().__init__(mask_env_response=mask_env_response, **kwargs)

        self.tw_wiki = tw_wiki or os.environ["TW_WIKI"]
        self.tw_news = tw_news or os.environ["TW_NEWS"]
        self.tw_webshop = tw_webshop or os.environ["TW_WEBSHOP"]
        self.tw_home = tw_home or os.environ["TW_HOME"]

        self.max_steps = max_steps
        self.seed = seed
        self.mask_thinking = mask_thinking
        self.max_obs_tokens = max_obs_tokens
        self.max_tokens_per_turn = max_tokens_per_turn
        self.env_timeout_ms = env_timeout_ms
        self.step_timeout_s = step_timeout_s

        self.rollout_mode = (
            (rollout_mode or os.environ.get("TW_ROLLOUT_MODE", "accumulated")).strip().lower()
        )
        if self.rollout_mode not in ("accumulated", "fresh"):
            raise ValueError(
                f"TW_ROLLOUT_MODE must be 'accumulated' or 'fresh', got {self.rollout_mode!r}"
            )

        self.flags = get_timewarp_flags(self.tw_wiki, self.tw_news, self.tw_webshop, self.tw_home)
        self.obs_preprocessor = make_obs_preprocessor(self.flags)

        self._frag = None

        self.sites_filter = sites_filter
        self.task_limit = task_limit

        self.dataset_name = "timewarp"
        self.dataset = build_timewarp_dataset(
            split="train",
            seed=self.seed,
            flags=self.flags,
            tw_wiki=self.tw_wiki,
            tw_news=self.tw_news,
            tw_webshop=self.tw_webshop,
            tw_home=self.tw_home,
            sites=self.sites_filter,
            limit=self.task_limit,
        )
        self.eval_dataset = None

        self.reward_funcs = [self.outcome_reward_func]

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset

    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset | None:
        if self.eval_dataset is None:
            self.eval_dataset = build_timewarp_dataset(
                split="test",
                seed=self.seed,
                flags=self.flags,
                tw_wiki=self.tw_wiki,
                tw_news=self.tw_news,
                tw_webshop=self.tw_webshop,
                tw_home=self.tw_home,
                sites=self.sites_filter,
                limit=self.task_limit,
            )
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n))
        return self.eval_dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.reward_funcs

    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:

        last = messages[-1] if messages else {}
        if last.get("role") != "assistant":
            return False
        return is_answer_action(parse_action(last.get("content", "")))

    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:

        raise NotImplementedError("TimeWarpEnv drives the environment inside generate().")

    def outcome_reward_func(self, llm_as_judge_reward, **kwargs) -> List[float]:
        """Return the per-trajectory outcome reward from task.validate()."""
        return llm_as_judge_reward

    @staticmethod
    def _ct_to_ids(tokenizer, messages, add_generation_prompt: bool) -> list[int]:

        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        return tokenizer.encode(text, add_special_tokens=False)

    def _ensure_fragments(self, tokenizer) -> dict:
        if self._frag is not None:
            return self._frag

        def enc(s: str) -> list[int]:
            return tokenizer.encode(s, add_special_tokens=False)

        unk = tokenizer.unk_token_id

        def _pair(open_tok, close_tok):
            a = tokenizer.convert_tokens_to_ids(open_tok)
            b = tokenizer.convert_tokens_to_ids(close_tok)
            bad = a is None or b is None or a == unk or b == unk
            return None if bad else (a, b)

        _FAMILIES = [
            ("chatml", "<|im_start|>", "<|im_end|>"),
            ("gemma4", "<|turn>", "<turn|>"),
        ]
        im_start = im_end = None
        for _name, _o, _c in _FAMILIES:
            got = _pair(_o, _c)
            if got:
                im_start, im_end = got
                self._turn_family = _name
                break
        if im_start is None:
            raise RuntimeError(
                "TimeWarpEnv needs a known turn-delimiter pair. Tried: "
                + ", ".join(f"{n} ({o}/{c})" for n, o, c in _FAMILIES)
                + ". Got unknown ids for all of them. Add the model's pair to _FAMILIES."
            )

        nl = enc("\n")
        frag = {
            "im_start": im_start,
            "im_end": im_end,
            "assistant_term": [im_end] + nl,
            "user_open": [im_start] + enc("user") + nl,
            "user_close": [im_end] + nl,
            "assistant_open": [im_start] + enc("assistant") + nl,
        }

        sentinel = "TWSENTINELOBS"
        rendered = self._ct_to_ids(tokenizer, [{"role": "user", "content": sentinel}], True)

        marker = frag["user_open"] + enc(sentinel) + frag["user_close"]
        pos = -1
        for i in range(len(rendered) - len(marker), -1, -1):
            if rendered[i : i + len(marker)] == marker:
                pos = i
                break
        if pos == -1:
            raise RuntimeError(
                f"TimeWarpEnv fragment verification failed for turn family "
                f"'{getattr(self, '_turn_family', '?')}'; could not locate the "
                "`<open>user\\n{content}<close>\\n` block in the rendered chat "
                "template. Inspect verifiers/envs/timewarp_env.py _ensure_fragments().\n"
                f"marker={marker}\nrendered={rendered}"
            )
        frag["assistant_open"] = rendered[pos + len(marker) :]

        self._frag = frag
        return frag

    def _user_turn_ids(self, tokenizer, frag: dict, obs_text: str) -> list[int]:
        """Token ids for: '\\n<|im_start|>user\\n{obs}<|im_end|>\\n<|im_start|>assistant\\n'."""
        obs_ids = tokenizer.encode(obs_text, add_special_tokens=False)
        if len(obs_ids) > self.max_obs_tokens:
            obs_ids = obs_ids[: self.max_obs_tokens]

        nl = frag["assistant_term"][1:]
        return nl + frag["user_open"] + obs_ids + frag["user_close"] + frag["assistant_open"]

    def _reset_env(self, state: dict) -> None:
        import gymnasium as gym

        env = gym.make(
            f"browsergym/timewarp.{state['task_id']}",
            headless=True,
            slow_mo=0,
            timeout=self.env_timeout_ms,
        )
        obs, _info = env.reset(seed=state["seed"])
        state["env"] = env

        if self.rollout_mode == "fresh":
            try:
                state["cur_obs"] = build_observation_text(self.obs_preprocessor(obs), self.flags)
            except Exception as e:
                logger.warning("initial obs render failed for task %s: %s", state["task_id"], e)
                state["cur_obs"] = "(observation unavailable)"

    def _step_env(self, state: dict, action: str) -> tuple[str, float, bool]:
        obs, reward, terminated, truncated, info = state["env"].step(action)
        processed = self.obs_preprocessor(obs)
        obs_text = build_observation_text(processed, self.flags)
        return obs_text, float(reward), bool(terminated or truncated)

    def _close_env(self, state: dict) -> None:
        env = state.get("env")
        if env is not None:
            try:
                env.close()
            except Exception as e:
                logger.warning("env.close failed for task %s: %s", state.get("task_id"), e)

    def _init_states(self, prompts, task_configs, tokenizer, frag) -> List[Dict[str, Any]]:
        states: List[Dict[str, Any]] = []
        for prompt, cfg in zip(prompts, task_configs):
            messages = (
                prompt["prompt"] if isinstance(prompt, dict) and "prompt" in prompt else prompt
            )
            seq_ids = self._ct_to_ids(tokenizer, messages, True)
            states.append(
                {
                    "messages": list(messages),
                    "prompt_messages": len(messages),
                    "task_id": cfg["task_id"],
                    "seed": cfg.get("seed", self.seed),
                    "goal": (cfg.get("goal") if isinstance(cfg, dict) else None)
                    or "Complete the task",
                    "additional_instructions": (
                        cfg.get("additional_instructions") if isinstance(cfg, dict) else ""
                    )
                    or None,
                    "carried_plan": None,
                    "carried_memory": None,
                    "plan_step": 0,
                    "cur_obs": None,
                    "fresh_rows": [],
                    "seq_ids": list(seq_ids),
                    "prompt_len": len(seq_ids),
                    "completion_ids": [],
                    "completion_mask": [],
                    "turn_ids": [],
                    "role_ids": [],
                    "reward": 0.0,
                    "last_reward": 0.0,
                    "n_steps": 0,
                    "n_valid_actions": 0,
                    "ended_by_answer": False,
                    "completed": False,
                    "last_action": None,
                    "pending_obs": None,
                    "executor": ThreadPoolExecutor(max_workers=1, thread_name_prefix="tw-env"),
                    "env": None,
                }
            )

        futures = {i: s["executor"].submit(self._reset_env, s) for i, s in enumerate(states)}
        failed: List[int] = []
        for i, fut in futures.items():
            try:
                fut.result(timeout=self.step_timeout_s)
            except Exception as e:
                logger.warning("env.reset failed for task %s: %s", states[i]["task_id"], e)
                failed.append(i)

        for i in failed:
            s = states[i]
            ok = False
            for _attempt in range(2):
                time.sleep(3.0)
                try:
                    s["executor"].submit(self._close_env, s).result(timeout=self.step_timeout_s)
                except Exception:
                    pass
                try:
                    s["executor"].submit(self._reset_env, s).result(timeout=self.step_timeout_s)
                    ok = True
                    break
                except Exception as e:
                    logger.warning(
                        "env.reset retry %d failed for task %s: %s", _attempt + 1, s["task_id"], e
                    )
            if ok:
                logger.info("env.reset retry succeeded for task %s", s["task_id"])
            else:
                self._mark_failed(s, frag)
        return states

    def _mark_failed(self, state: dict, frag: dict) -> None:
        """A rollout that could not run: emit a single fully-masked token so the"""
        state["completed"] = True
        state["reward"] = 0.0
        if not state["completion_ids"]:
            state["seq_ids"].append(frag["im_end"])
            state["completion_ids"].append(frag["im_end"])
            state["completion_mask"].append(self.env_mask)
            state["turn_ids"].append(0)
            state["role_ids"].append(0)
            state["messages"].append({"role": "assistant", "content": ""})

    def _apply_assistant_turn(self, state, out_ids, text, tokenizer, frag) -> None:
        state["messages"].append({"role": "assistant", "content": text})
        state["n_steps"] += 1
        turn = state["n_steps"]

        if "<action>" in text:
            think_text = text.split("<action>")[0]
            role_think_len = len(tokenizer.encode(think_text, add_special_tokens=False))
            role_think_len = max(0, min(role_think_len, len(out_ids)))
        else:
            role_think_len = len(out_ids)
        len(out_ids) - role_think_len

        mask_think_len = role_think_len if (self.mask_thinking and "<action>" in text) else 0
        mask_action_len = len(out_ids) - mask_think_len

        state["seq_ids"].extend(out_ids)
        state["completion_ids"].extend(out_ids)
        state["completion_mask"].extend([self.env_mask] * mask_think_len + [1] * mask_action_len)
        state["turn_ids"].extend([turn] * len(out_ids))

        state["role_ids"].extend(self._role_split(out_ids, text, tokenizer))

        if not out_ids or out_ids[-1] != frag["im_end"]:
            state["seq_ids"].append(frag["im_end"])
            state["completion_ids"].append(frag["im_end"])
            state["completion_mask"].append(1)
            state["turn_ids"].append(turn)
            state["role_ids"].append(2)

        assert (
            len(state["completion_ids"])
            == len(state["completion_mask"])
            == len(state["turn_ids"])
            == len(state["role_ids"])
        )

    _PLAN_SPAN_RE = re.compile(r"<plan>.*?</plan>", re.DOTALL)
    _MEMORY_SPAN_RE = re.compile(r"<memory>.*?</memory>", re.DOTALL)

    def _role_split(self, out_ids, text, tokenizer) -> List[int]:
        n = len(out_ids)
        if "<action>" in text:
            think_text = text.split("<action>")[0]
            think_len = len(tokenizer.encode(think_text, add_special_tokens=False))
            think_len = max(0, min(think_len, n))
        else:
            think_text = text
            think_len = n

        roles = [1] * think_len + [2] * (n - think_len)
        if think_len == 0:
            return roles

        def _tok(pos: int) -> int:
            """char offset in think_text -> approximate token offset, clamped to the region."""
            if pos <= 0:
                return 0
            return max(
                0, min(len(tokenizer.encode(think_text[:pos], add_special_tokens=False)), think_len)
            )

        for rx, role in ((self._PLAN_SPAN_RE, 3), (self._MEMORY_SPAN_RE, 4)):
            for mt in rx.finditer(think_text):
                a, b = _tok(mt.start()), _tok(mt.end())
                for i in range(a, b):
                    roles[i] = role
        return roles

    def _append_observation(self, state, obs_text, tokenizer, frag) -> None:
        delta = self._user_turn_ids(tokenizer, frag, obs_text)
        state["seq_ids"].extend(delta)
        state["completion_ids"].extend(delta)
        state["completion_mask"].extend([self.env_mask] * len(delta))
        state["turn_ids"].extend([0] * len(delta))
        state["role_ids"].extend([0] * len(delta))
        state["messages"].append({"role": "user", "content": obs_text})
        assert (
            len(state["completion_ids"])
            == len(state["completion_mask"])
            == len(state["turn_ids"])
            == len(state["role_ids"])
        )

    def _make_fresh_row(self, out_ids, text, tokenizer, frag) -> dict:
        """Build ONE independent single-step training row from an assistant turn."""
        if "<action>" in text:
            think_text = text.split("<action>")[0]
            role_think_len = len(tokenizer.encode(think_text, add_special_tokens=False))
            role_think_len = max(0, min(role_think_len, len(out_ids)))
        else:
            role_think_len = len(out_ids)
        len(out_ids) - role_think_len
        mask_think_len = role_think_len if (self.mask_thinking and "<action>" in text) else 0
        mask_action_len = len(out_ids) - mask_think_len

        comp = list(out_ids)
        mask = [self.env_mask] * mask_think_len + [1] * mask_action_len
        turn = [1] * len(out_ids)
        role = self._role_split(out_ids, text, tokenizer)

        if not out_ids or out_ids[-1] != frag["im_end"]:
            comp.append(frag["im_end"])
            mask.append(1)
            turn.append(1)
            role.append(2)
        assert len(comp) == len(mask) == len(turn) == len(role)
        return {"completion_ids": comp, "mask": mask, "turn_ids": turn, "role_ids": role}

    def _generate_fresh(
        self,
        prompts,
        vllm_client,
        tokenizer,
        sampling_kwargs,
        max_seq_len,
        task_configs,
        frag,
    ) -> Dict[str, Any]:
        """Single-turn-like-SFT rollout: a FRESH prompt per browser step (state carried"""
        _sk = dict(sampling_kwargs or {})
        _sk.update(self.sampling_args or {})
        states = self._init_states(prompts, task_configs, tokenizer, frag)

        loop_guard = self.max_steps + 2
        step_idx = 0
        while not all(s["completed"] for s in states):
            live = [i for i, s in enumerate(states) if not s["completed"]]
            if not live:
                break

            prompt_ids_batch = []
            for i in live:
                s = states[i]
                msgs = build_initial_prompt(
                    self.flags,
                    s["goal"],
                    s["additional_instructions"],
                    carried_plan=s["carried_plan"],
                    carried_plan_step=s["plan_step"],
                    carried_memory=s["carried_memory"],
                    observation_text=s["cur_obs"],
                )
                pid = self._ct_to_ids(tokenizer, msgs, True)

                _mpl = int(os.environ.get("TW_MAX_PROMPT_LENGTH", "10240"))
                _cap = min(int(max_seq_len), _mpl) - self.max_tokens_per_turn - 64
                if len(pid) > _cap:
                    pid = pid[-_cap:]
                s["_cur_prompt_ids"] = pid
                s["_cur_prompt_msgs"] = msgs
                prompt_ids_batch.append(list(pid))

            resp = vllm_client.generate(
                prompt_ids=prompt_ids_batch,
                n=1,
                max_tokens=self.max_tokens_per_turn,
                temperature=_sk.get("temperature", 1.0),
                top_p=_sk.get("top_p", 1.0),
                top_k=_sk.get("top_k", -1),
                min_p=_sk.get("min_p", 0.0),
                repetition_penalty=_sk.get("repetition_penalty", 1.0),
            )
            comp_ids = resp["completion_ids"]

            for li, i in enumerate(live):
                s = states[i]
                out_ids = list(comp_ids[li])
                text = tokenizer.decode(out_ids, skip_special_tokens=False)
                row = self._make_fresh_row(out_ids, text, tokenizer, frag)
                row["prompt_ids"] = list(s["_cur_prompt_ids"])
                row["messages"] = list(s["_cur_prompt_msgs"]) + [
                    {"role": "assistant", "content": text}
                ]
                s["fresh_rows"].append(row)
                s["n_steps"] += 1
                s["last_action"] = parse_action(text)
                if s["last_action"]:
                    s["n_valid_actions"] += 1
                p, m = parse_plan(text), parse_memory(text)
                if p is not None:
                    s["carried_plan"] = p
                if m is not None:
                    s["carried_memory"] = m
                s["plan_step"] += 1

            futures = {}
            for i in live:
                s = states[i]
                if s["last_action"] is not None:
                    futures[i] = s["executor"].submit(self._step_env, s, s["last_action"])

            for i in live:
                s = states[i]
                if i in futures:
                    try:
                        obs_text, reward, done = futures[i].result(timeout=self.step_timeout_s)
                        s["last_reward"] = reward
                        if done:
                            s["reward"] = reward
                            s["ended_by_answer"] = is_answer_action(s["last_action"])
                            s["completed"] = True
                            continue
                        s["cur_obs"] = obs_text
                    except Exception as e:
                        logger.warning("env.step failed for task %s: %s", s["task_id"], e)
                        s["reward"] = s["last_reward"]
                        s["completed"] = True
                        continue
                else:
                    s["cur_obs"] = (
                        "Error: no valid action found in your previous response. "
                        "Output exactly one action inside <action>...</action> tags."
                    )
                if s["n_steps"] >= self.max_steps:
                    s["reward"] = s["last_reward"]
                    s["completed"] = True
                    continue

            step_idx += 1
            if step_idx > loop_guard:
                for i in live:
                    states[i]["reward"] = states[i]["last_reward"]
                    states[i]["completed"] = True
                break

        for s in states:
            try:
                s["executor"].submit(self._close_env, s).result(timeout=self.step_timeout_s)
            except Exception:
                pass
            s["executor"].shutdown(wait=False)

        shaped_on = os.environ.get("TW_SHAPED_REWARD", "0").lower() in ("1", "true", "yes", "on")
        masktrunc_on = os.environ.get("TW_MASK_TRUNCATED", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        for s in states:
            if shaped_on:
                shaped = 0.0
                if s["n_steps"] > 0:
                    shaped += 0.05 * (s["n_valid_actions"] / s["n_steps"])
                if s["ended_by_answer"]:
                    shaped += 0.1
                s["reward"] = float(s["reward"]) + shaped
            if masktrunc_on and not s["ended_by_answer"]:
                for r in s["fresh_rows"]:
                    r["mask"] = [0] * len(r["mask"])

        out = {
            k: []
            for k in (
                "ids",
                "messages",
                "mask",
                "reward",
                "steps",
                "turn_ids",
                "role_ids",
                "prompt_ids",
                "episode_id",
                "task_index",
            )
        }
        for ep_idx, s in enumerate(states):
            if not s["fresh_rows"]:
                s["fresh_rows"].append(
                    {
                        "prompt_ids": list(s["seq_ids"]),
                        "completion_ids": [frag["im_end"]],
                        "mask": [self.env_mask],
                        "turn_ids": [0],
                        "role_ids": [0],
                        "messages": [{"role": "assistant", "content": ""}],
                    }
                )
            R = float(s["reward"])
            for r in s["fresh_rows"]:
                out["ids"].append(r["completion_ids"])
                out["messages"].append(r["messages"])
                out["mask"].append(r["mask"])
                out["turn_ids"].append(r["turn_ids"])
                out["role_ids"].append(r["role_ids"])
                out["prompt_ids"].append(r["prompt_ids"])
                out["reward"].append(R)
                out["steps"].append(s["n_steps"])
                out["episode_id"].append(ep_idx)
                out["task_index"].append(int(s["task_id"]))
        out["fresh"] = True
        return out

    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        vllm_client,
        tokenizer,
        sampling_kwargs: Dict[str, Any],
        max_seq_len: int,
        task_configs=None,
        **kwargs: Any,
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        assert task_configs is not None, "TimeWarpEnv.generate requires task_configs"
        frag = self._ensure_fragments(tokenizer)

        if self.rollout_mode == "fresh":
            return self._generate_fresh(
                prompts,
                vllm_client,
                tokenizer,
                sampling_kwargs,
                max_seq_len,
                task_configs,
                frag,
            )

        _sk = dict(sampling_kwargs or {})
        _sk.update(self.sampling_args or {})

        states = self._init_states(prompts, task_configs, tokenizer, frag)

        _max_model_len = int(max_seq_len)

        _max_prompt = int(os.environ.get("TW_MAX_PROMPT_LENGTH", "16384"))

        seq_budget = (
            min(_max_model_len, _max_prompt) - self.max_tokens_per_turn - self.max_obs_tokens - 256
        )

        loop_guard = self.max_steps + 2
        step_idx = 0
        while not all(s["completed"] for s in states):
            live = [i for i, s in enumerate(states) if not s["completed"]]

            for i in list(live):
                s = states[i]
                if len(s["seq_ids"]) > seq_budget:
                    s["reward"] = s["last_reward"]
                    s["completed"] = True
                    if not s["completion_ids"]:
                        self._mark_failed(s, frag)
                    logger.debug(
                        "rollout task %s ended early: context budget reached (%d > %d)",
                        s["task_id"],
                        len(s["seq_ids"]),
                        seq_budget,
                    )
            live = [i for i in live if not states[i]["completed"]]
            if not live:
                continue

            prompt_ids_batch = [list(states[i]["seq_ids"]) for i in live]
            resp = vllm_client.generate(
                prompt_ids=prompt_ids_batch,
                n=1,
                max_tokens=self.max_tokens_per_turn,
                temperature=_sk.get("temperature", 1.0),
                top_p=_sk.get("top_p", 1.0),
                top_k=_sk.get("top_k", -1),
                min_p=_sk.get("min_p", 0.0),
                repetition_penalty=_sk.get("repetition_penalty", 1.0),
            )
            comp_ids = resp["completion_ids"]

            for li, i in enumerate(live):
                state = states[i]
                out_ids = list(comp_ids[li])

                text = tokenizer.decode(out_ids, skip_special_tokens=False)
                self._apply_assistant_turn(state, out_ids, text, tokenizer, frag)
                state["last_action"] = parse_action(text)
                if state["last_action"]:
                    state["n_valid_actions"] += 1
                state["pending_obs"] = None

            futures = {}
            for i in live:
                state = states[i]
                if state["last_action"] is not None:
                    futures[i] = state["executor"].submit(
                        self._step_env, state, state["last_action"]
                    )

            for i in live:
                state = states[i]
                if i in futures:
                    try:
                        obs_text, reward, done = futures[i].result(timeout=self.step_timeout_s)
                        state["last_reward"] = reward
                        if done:
                            state["reward"] = reward
                            state["ended_by_answer"] = is_answer_action(state["last_action"])
                            state["completed"] = True
                            continue
                        state["pending_obs"] = obs_text
                    except Exception as e:
                        logger.warning("env.step failed for task %s: %s", state["task_id"], e)
                        state["reward"] = state["last_reward"]
                        state["completed"] = True
                        continue
                else:
                    state["pending_obs"] = (
                        "Error: no valid action found in your previous response. "
                        "Output exactly one action inside <action>...</action> tags."
                    )

                if state["n_steps"] >= self.max_steps:
                    state["reward"] = state["last_reward"]
                    state["completed"] = True
                    continue

                self._append_observation(state, state["pending_obs"], tokenizer, frag)

            step_idx += 1
            if step_idx > loop_guard:
                for i in live:
                    states[i]["reward"] = states[i]["last_reward"]
                    states[i]["completed"] = True
                break

        close_futures = []
        for s in states:
            try:
                close_futures.append(s["executor"].submit(self._close_env, s))
            except Exception:
                pass
        for f in close_futures:
            try:
                f.result(timeout=self.step_timeout_s)
            except Exception:
                pass
        for s in states:
            s["executor"].shutdown(wait=False)

        if os.environ.get("TW_SHAPED_REWARD", "0").lower() in ("1", "true", "yes", "on"):
            for s in states:
                shaped = 0.0
                if s["n_steps"] > 0:
                    shaped += 0.05 * (s["n_valid_actions"] / s["n_steps"])
                if s["ended_by_answer"]:
                    shaped += 0.1
                s["reward"] = float(s["reward"]) + shaped

        if os.environ.get("TW_MASK_TRUNCATED", "0").lower() in ("1", "true", "yes", "on"):
            for s in states:
                if not s["ended_by_answer"]:
                    s["completion_mask"] = [0] * len(s["completion_mask"])

        completion_messages = [s["messages"][s["prompt_messages"] :] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        completion_reward = [s["reward"] for s in states]
        completion_steps = [s["n_steps"] for s in states]

        return {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
            "reward": completion_reward,
            "steps": completion_steps,
            "turn_ids": [s["turn_ids"] for s in states],
            "role_ids": [s["role_ids"] for s in states],
        }
