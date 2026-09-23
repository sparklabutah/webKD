"""Multi-turn policy-gradient baseline for web agents."""

from .envs.environment import Environment
from .envs.multistep_env import MultiStepEnv
from .envs.timewarp_env import TimeWarpEnv
from .trainers.grpo_env_trainer import GRPOEnvTrainer
from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer
from .utils.config_utils import get_default_grpo_config
from .utils.logging_utils import setup_logging, print_prompt_completions_sample

__version__ = "0.1.0"

setup_logging()

__all__ = [
    "Environment",
    "MultiStepEnv",
    "TimeWarpEnv",
    "GRPOEnvTrainer",
    "get_model",
    "get_tokenizer",
    "get_model_and_tokenizer",
    "get_default_grpo_config",
    "setup_logging",
    "print_prompt_completions_sample",
]
