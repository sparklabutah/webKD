"""Select trajectory episodes by outcome, completion, and error status."""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional


def load_summary(task_dir: Path) -> Optional[dict]:
    summary_path = task_dir / "summary_info.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def last_action_is_send_message(task_dir: Path, n_steps: int) -> bool:
    """Return True if the final agent action is a send_msg_to_user call."""
    last_step = n_steps - 1
    output_path = task_dir / "training_data" / f"agent_output_step_{last_step}.txt"
    if not output_path.exists():
        return False
    try:
        text = output_path.read_text()
    except OSError:
        return False
    action_match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.DOTALL)
    if not action_match:
        return False
    action_body = action_match.group(1).strip()
    return action_body.startswith("send_msg_to_user(") or action_body.startswith(
        "send_message_to_user("
    )


def is_task_dir(entry: Path) -> bool:
    """A task directory contains summary_info.json."""
    return entry.is_dir() and (entry / "summary_info.json").exists()


def passes_filter(
    summary: dict,
    require_reward: bool,
    max_steps: Optional[int],
    no_error: bool = False,
    final_answer: bool = False,
    task_dir: Optional[Path] = None,
) -> bool:
    if no_error and summary.get("err_msg") is not None:
        return False
    if require_reward and summary.get("cum_reward", 0) != 1.0:
        return False
    if max_steps is not None and summary.get("n_steps", float("inf")) >= max_steps:
        return False
    if final_answer:
        n_steps = summary.get("n_steps", 0)
        if n_steps <= 0 or task_dir is None:
            return False
        if not last_action_is_send_message(task_dir, n_steps):
            return False
    return True


def build_output_name(
    input_path: str,
    require_reward: bool,
    max_steps: Optional[int],
    no_error: bool = False,
    final_answer: bool = False,
) -> str:
    base = input_path.rstrip("/")
    suffix_parts = []
    if require_reward:
        suffix_parts.append("reward1")
    if max_steps is not None:
        suffix_parts.append(f"maxsteps{max_steps}")
    if no_error:
        suffix_parts.append("noerr")
    if final_answer:
        suffix_parts.append("finalanswer")
    return f"{base}_filtered_{'_'.join(suffix_parts)}"


def main():
    parser = argparse.ArgumentParser(
        description="Filter training data by reward and/or step count."
    )
    parser.add_argument("input", help="Path to the training data folder")
    parser.add_argument(
        "--reward", action="store_true", help="Keep only tasks with cum_reward == 1.0"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Keep only tasks with n_steps < MAX_STEPS"
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Output directory (auto-generated if omitted)"
    )
    parser.add_argument(
        "--no-error",
        action="store_true",
        help="Exclude tasks that terminated due to an error (err_msg is set)",
    )
    parser.add_argument(
        "--final-answer",
        action="store_true",
        help="Keep only tasks whose last action is send_msg_to_user",
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of creating symlinks"
    )
    args = parser.parse_args()

    if not args.reward and args.max_steps is None and not args.no_error and not args.final_answer:
        parser.error(
            "Specify at least one filter: --reward, --max-steps, --no-error, and/or --final-answer"
        )

    input_root = Path(args.input).resolve()
    if not input_root.is_dir():
        sys.exit(f"Error: {input_root} is not a directory")

    output_root = (
        Path(args.output)
        if args.output
        else Path(
            build_output_name(
                args.input, args.reward, args.max_steps, args.no_error, args.final_answer
            )
        )
    )
    output_root = output_root.resolve()

    if output_root.exists():
        sys.exit(f"Error: output directory already exists: {output_root}")

    total_tasks = 0
    kept_tasks = 0

    batch_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()], key=lambda p: p.name)

    for batch_dir in batch_dirs:
        batch_kept = 0
        batch_total = 0

        for entry in sorted(batch_dir.iterdir(), key=lambda p: p.name):
            if not is_task_dir(entry):
                continue

            batch_total += 1
            total_tasks += 1

            summary = load_summary(entry)
            if summary is None:
                continue

            if not passes_filter(
                summary, args.reward, args.max_steps, args.no_error, args.final_answer, entry
            ):
                continue

            dest = output_root / batch_dir.name / entry.name
            dest.parent.mkdir(parents=True, exist_ok=True)

            if args.copy:
                shutil.copytree(entry, dest)
            else:
                os.symlink(entry, dest)

            batch_kept += 1
            kept_tasks += 1

        if batch_kept > 0:
            for entry in batch_dir.iterdir():
                if entry.is_dir():
                    continue
                dest = output_root / batch_dir.name / entry.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if args.copy:
                    shutil.copy2(entry, dest)
                else:
                    os.symlink(entry.resolve(), dest)

        if batch_total > 0:
            print(f"  {batch_dir.name}: kept {batch_kept}/{batch_total} tasks")

    print(f"\nTotal: kept {kept_tasks}/{total_tasks} tasks")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
