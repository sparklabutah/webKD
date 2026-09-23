"""Multi-turn policy-gradient baseline for web agents."""

from __future__ import annotations

import os
import re
from typing import Optional


def _envflag(name: str, default: str) -> bool:
    return os.environ.get(name, default) in ("1", "true", "True")


def get_timewarp_flags(tw_wiki: str, tw_news: str, tw_webshop: str, tw_home: Optional[str] = None):
    """Return the GenericPromptFlags used for TimeWarp (AGENT_LLAMA3_70B base)."""
    from agentlab.agents.generic_agent import AGENT_LLAMA3_70B
    from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS

    flags = AGENT_LLAMA3_70B.flags.copy()
    flags.obs.use_error_logs = True
    flags.obs.use_tabs = True
    flags.action.action_set = DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["timewarp"]

    flags.use_thinking = _envflag("TW_USE_THINKING", "1")
    flags.use_memory = _envflag("TW_USE_MEMORY", "0")
    flags.use_plan = _envflag("TW_USE_PLAN", "0")

    flags.extra_instructions = (
        f"IMPORTANT: You must only navigate to URLs within the TimeWarp environment. "
        f"Do NOT navigate to external websites.\n"
        f"Only use the URLs provided in the TimeWarp environment.\n"
        f"WIKI URL: {tw_wiki}\n"
        f"NEWS URL: {tw_news}\n"
        f"SHOP URL: {tw_webshop}\n"
        f'For instance, goto("{tw_wiki}/") to navigate to WIKI. '
        f"If you need to access Wikipedia content, use the local TimeWarp wiki instance instead.\n"
        f"Strictly follow the instructions provided in the task description.\n"
        f"You MUST output EXACTLY ONE action per response. Do NOT attempt multiple actions at once.\n"
        f"CRITICAL: Output the <action> tag ONLY ONCE, at the end of your response, "
        f"OUTSIDE of any <think> tags. Do NOT include <action> tags inside your reasoning section."
    )
    return flags


def make_obs_preprocessor(flags):
    from agentlab.agents.dynamic_prompting import make_obs_preprocessor as _mk

    return _mk(flags.obs)


def build_observation_text(obs: dict, flags) -> str:
    """Render a processed BrowserGym obs into the same text GenericAgent shows."""
    from agentlab.agents.dynamic_prompting import Observation

    observation = Observation(obs, flags.obs)
    return observation.prompt


def build_initial_prompt(
    flags,
    goal_text: str,
    additional_instructions: Optional[str] = None,
    *,
    carried_plan: Optional[str] = None,
    carried_plan_step: int = -1,
    carried_memory: Optional[str] = None,
    observation_text: Optional[str] = None,
) -> list[dict]:
    """Build the [system, user] messages that start the conversation — the same"""
    from agentlab.agents import dynamic_prompting as dp

    goal_object = [{"type": "text", "text": goal_text}]
    if additional_instructions:
        goal_object.append({"type": "text", "text": f"\n\n{additional_instructions}"})

    action_set = flags.action.action_set.make_action_set()
    action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)

    system_text = dp.SystemPrompt().prompt

    user_parts: list[str] = []

    instructions = dp.GoalInstructions(goal_object, extra_instructions=flags.extra_instructions)
    instr_prompt = instructions.prompt
    if isinstance(instr_prompt, list):
        user_parts.append("".join(p.get("text", "") for p in instr_prompt))
    else:
        user_parts.append(str(instr_prompt))

    user_parts.append(
        "\n# Observation\n"
        "After your first action the environment returns the current page "
        "observation (AXTree, open tabs, errors). Begin by navigating to the "
        "page relevant to your goal.\n"
    )

    user_parts.append(action_prompt.prompt)

    if flags.use_hints:
        user_parts.append(dp.Hints().prompt)

    from agentlab.agents.generic_agent.generic_agent_prompt import Memory, Plan

    think = dp.Think(visible=lambda: flags.use_thinking)

    plan = Plan(
        carried_plan if carried_plan else "No plan yet",
        carried_plan_step,
        visible=lambda: flags.use_plan,
    )
    memory = Memory(visible=lambda: flags.use_memory)
    user_parts.append(think.prompt)
    user_parts.append(plan.prompt)
    user_parts.append(memory.prompt)

    if flags.use_memory and carried_memory:
        user_parts.append(f"\n# Previous memories\n{carried_memory}\n")

    if flags.use_abstract_example:
        user_parts.append(
            "\n# Abstract Example\n\n"
            "Here is an abstract version of the answer with description of the content of\n"
            "each tag. Make sure you follow this structure, but replace the content with your\n"
            "answer:\n"
            f"{think.abstract_ex}"
            f"{plan.abstract_ex}"
            f"{memory.abstract_ex}"
            f"{action_prompt.abstract_ex}\n"
        )

    if flags.use_concrete_example:
        user_parts.append(
            "\n# Concrete Example\n\n"
            "Here is a concrete example of how to format your answer.\n"
            "Make sure to follow the template with proper tags:\n"
            f"{think.concrete_ex}"
            f"{plan.concrete_ex}"
            f"{memory.concrete_ex}"
            f"{action_prompt.concrete_ex}\n"
        )

    if observation_text is not None:
        user_parts.append(f"\n# Current Observation\n{observation_text}\n")

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
_MEMORY_RE = re.compile(r"<memory>(.*?)</memory>", re.DOTALL)


def parse_plan(text: str) -> Optional[str]:
    """Last <plan>...</plan> block from an assistant turn, or None."""
    m = _PLAN_RE.findall(text or "")
    return m[-1].strip() if m else None


def parse_memory(text: str) -> Optional[str]:
    """Last <memory>...</memory> block from an assistant turn, or None."""
    m = _MEMORY_RE.findall(text or "")
    return m[-1].strip() if m else None


_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


def parse_action(text: str) -> Optional[str]:
    """Extract the last <action>...</action> block, stripping think/memory/plan."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<memory>.*?</memory>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<plan>.*?</plan>", "", cleaned, flags=re.DOTALL)
    matches = _ACTION_RE.findall(cleaned)
    if not matches:
        return None
    return matches[-1].strip()


def is_answer_action(action: str) -> bool:
    """True if the action is the terminal send_msg_to_user(...) answer."""
    if action is None:
        return False
    return action.strip().startswith("send_msg_to_user")
