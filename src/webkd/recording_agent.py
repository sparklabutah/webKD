# Copyright 2024 ServiceNow
# Licensed under the Apache License, Version 2.0.
# Modified to record structured web-agent trajectories.

"""AgentLab adapter that records prompts and structured responses."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import logging

from browsergym.experiments.agent import AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, BaseMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import MainPrompt

logger = logging.getLogger(__name__)


class GenericAgentWithTrainingArgs(GenericAgentArgs):
    """Agent arguments for GenericAgentWithTraining."""

    def __post_init__(self):
        super().__post_init__()
        try:
            self.agent_name = f"GenericAgentWithTraining-{self.chat_model_args.model_name}".replace(
                "/", "_"
            )
        except AttributeError:
            pass

    def make_agent(self):
        return GenericAgentWithTraining(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class GenericAgentWithTraining(GenericAgent):
    """GenericAgent extended with training data saving functionality."""

    def __init__(
        self,
        chat_model_args,
        flags,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)

        self._current_step = None
        self._training_data_dir = None

    @cost_tracker_decorator
    def get_action(self, obs):
        """Override get_action to save training data before and after LLM call."""

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

        if self._training_data_dir is not None and self._current_step is not None:
            self._save_training_prompts(system_prompt, human_prompt, self._current_step)
        else:
            logger.debug(
                f"Not saving training prompts: training_data_dir={self._training_data_dir}, current_step={self._current_step}"
            )

        try:
            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0

            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        action = ans_dict["action"]
        self.actions.append(action)
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        if self._training_data_dir is not None and self._current_step is not None:
            self._save_training_output(ans_dict, self._current_step)
        else:
            logger.debug(
                f"Not saving training output: training_data_dir={self._training_data_dir}, current_step={self._current_step}"
            )

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return action, agent_info

    def _save_training_prompts(
        self, system_prompt: SystemMessage, human_prompt: BaseMessage, step: int
    ):
        """Save system prompt and user prompt separately for training data."""
        try:
            logger.info(f"Saving training prompts for step {step} to {self._training_data_dir}")
            training_dir = Path(self._training_data_dir)
            training_dir.mkdir(parents=True, exist_ok=True)

            system_dict = {
                "role": system_prompt.get("role", "system"),
                "content": deepcopy(system_prompt.get("content", "")),
            }

            system_file = training_dir / f"system_prompt_step_{step}.json"
            with open(system_file, "w", encoding="utf-8") as f:
                json.dump(system_dict, f, indent=2, ensure_ascii=False)

            system_text_file = training_dir / f"system_prompt_step_{step}.txt"
            with open(system_text_file, "w", encoding="utf-8") as f:
                f.write(str(system_prompt))

            user_dict = {
                "role": human_prompt.get("role", "user"),
                "content": deepcopy(human_prompt.get("content", "")),
            }

            user_file = training_dir / f"user_prompt_step_{step}.json"
            with open(user_file, "w", encoding="utf-8") as f:
                json.dump(user_dict, f, indent=2, ensure_ascii=False)

            user_text_file = training_dir / f"user_prompt_step_{step}.txt"
            with open(user_text_file, "w", encoding="utf-8") as f:
                f.write(str(human_prompt))

        except Exception as e:
            logger.warning(f"Failed to save training prompts for step {step}: {e}")

    def _save_training_output(self, ans_dict: dict, step: int):
        """Save the agent output (thinking/reasoning + action) for training data."""
        try:
            logger.info(f"Saving training output for step {step} to {self._training_data_dir}")
            training_dir = Path(self._training_data_dir)
            training_dir.mkdir(parents=True, exist_ok=True)

            output_dict = {
                "think": ans_dict.get("think", None),
                "action": ans_dict.get("action", None),
            }

            output_file = training_dir / f"agent_output_step_{step}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_dict, f, indent=2, ensure_ascii=False)

            output_text_parts = []
            if output_dict["think"]:
                output_text_parts.append(f"Thinking:\n{output_dict['think']}\n")
            if output_dict["action"]:
                output_text_parts.append(f"Action:\n{output_dict['action']}\n")

            output_text_file = training_dir / f"agent_output_step_{step}.txt"
            with open(output_text_file, "w", encoding="utf-8") as f:
                f.write("\n".join(output_text_parts) if output_text_parts else "")

        except Exception as e:
            logger.warning(f"Failed to save training output for step {step}: {e}")

    def set_training_data_dir(self, exp_dir: Path):
        """Set the directory where training data should be saved."""
        self._training_data_dir = Path(exp_dir) / "training_data"
        self._training_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Training data directory set to: {self._training_data_dir}")

    def set_current_step(self, step: int):
        """Set the current step number for saving training data."""
        self._current_step = step
