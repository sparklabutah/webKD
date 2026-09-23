"""Multi-turn policy-gradient baseline for web agents."""

import platform

IS_MACOS = platform.system() == "Darwin"

if IS_MACOS:
    from .mock_vllm import LLM, SamplingParams
else:
    from vllm import LLM, SamplingParams

__all__ = ["LLM", "SamplingParams", "IS_MACOS"]
