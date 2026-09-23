"""Multi-turn policy-gradient baseline for web agents."""

from __future__ import annotations

import importlib.resources
import json
import os

from datasets import Dataset

from verifiers.prompts.timewarp_prompts import build_initial_prompt, get_timewarp_flags

TRAIN_IDS = (104, 231)
TEST_IDS = (1, 103)


def _load_raw_configs(tw_wiki: str, tw_news: str, tw_webshop: str, tw_home: str) -> list[dict]:
    import browsergym.timewarp

    raw = importlib.resources.files(browsergym.timewarp).joinpath("data/test.raw.json").read_text()
    raw = raw.replace("__WIKI__", tw_wiki)
    raw = raw.replace("__WEBSHOP__", tw_webshop)
    raw = raw.replace("__NEWS__", tw_news)
    raw = raw.replace("__HOME__", tw_home)
    return json.loads(raw)


def build_timewarp_dataset(
    split: str = "train",
    seed: int = 42,
    flags=None,
    tw_wiki: str | None = None,
    tw_news: str | None = None,
    tw_webshop: str | None = None,
    tw_home: str | None = None,
    train_ids: tuple[int, int] = TRAIN_IDS,
    test_ids: tuple[int, int] = TEST_IDS,
    sites: list[str] | None = None,
    limit: int = -1,
) -> Dataset:
    """Return a HuggingFace Dataset of TimeWarp tasks for the given split."""
    tw_wiki = tw_wiki or os.environ["TW_WIKI"]
    tw_news = tw_news or os.environ["TW_NEWS"]
    tw_webshop = tw_webshop or os.environ["TW_WEBSHOP"]
    tw_home = tw_home or os.environ["TW_HOME"]

    if flags is None:
        flags = get_timewarp_flags(tw_wiki, tw_news, tw_webshop, tw_home)

    configs = _load_raw_configs(tw_wiki, tw_news, tw_webshop, tw_home)

    lo, hi = train_ids if split == "train" else test_ids
    sites_set = set(sites) if sites else None

    rows: list[dict] = []
    for cfg in configs:
        tid = cfg["task_id"]
        if not (lo <= tid <= hi):
            continue
        if sites_set is not None and not (sites_set & set(cfg.get("sites", []))):
            continue

        goal = cfg.get("intent", "Complete the task")
        additional = cfg.get("additional_instructions")
        prompt = build_initial_prompt(flags, goal, additional)

        ref = cfg.get("eval", {}).get("reference_answers", {}).get("fuzzy_match", "")
        if isinstance(ref, list):
            ref = "; ".join(ref)

        rows.append(
            {
                "prompt": prompt,
                "task_id": tid,
                "seed": seed,
                "answer": ref,
                "sites": cfg.get("sites", []),
                "split": split,
                "goal": goal,
                "additional_instructions": additional or "",
            }
        )

    rows.sort(key=lambda r: r["task_id"])
    if limit and limit > 0:
        rows = rows[:limit]
    return Dataset.from_list(rows)
