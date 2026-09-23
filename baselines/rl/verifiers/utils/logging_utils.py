import logging
import sys
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel


def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """Setup basic logging configuration for the verifiers package."""
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    logger = logging.getLogger("verifiers")
    logger.setLevel(level.upper())
    logger.addHandler(handler)

    logger.propagate = False


def print_prompt_completions_sample(
    prompts: list[str],
    completions: list[dict],
    rewards: list[float],
    step: int,
) -> None:

    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    for prompt, completion, reward in zip(prompts, completions, rewards, strict=True):
        formatted_completion = Text()

        if isinstance(completion, dict):
            role = completion.get("role", "")
            if role == "user":
                role = "environment"
            content = completion.get("content", "")
            style = "bright_cyan" if role == "assistant" else "bright_magenta"
            formatted_completion.append(f"{role}: ", style="bold")
            formatted_completion.append(content, style=style)
        elif isinstance(completion, list):
            for i, message in enumerate(completion):
                if i > 0:
                    formatted_completion.append("\n\n")

                role = message.get("role", "")
                content = message.get("content", "")

                if role == "user":
                    role = "environment"

                style = "bright_cyan" if role == "assistant" else "bright_magenta"

                formatted_completion.append(f"{role}: ", style="bold")
                formatted_completion.append(content, style=style)
        else:
            formatted_completion = Text(str(completion))

        table.add_row(Text(prompt), formatted_completion, Text(f"{reward:.2f}"))
        table.add_section()

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)
