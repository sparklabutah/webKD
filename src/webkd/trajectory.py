"""Read, write, and parse structured web-agent trajectories."""

from __future__ import print_function

import glob
import json
import os
import re

TAGS = ("think", "plan", "step", "memory", "action")

_SECTION_PATTERNS = {
    "think": r"<think>\s*(.*?)\s*</think>",
    "plan": r"<plan>\s*(.*?)\s*</plan>",
    "step": r"<step>\s*(.*?)\s*</step>",
    "memory": r"<memory>\s*(.*?)\s*</memory>",
    "action": r"<action>\s*(.*?)\s*</action>",
}


def get_text(content):
    """Flatten an OpenAI-style content field (str or list-of-parts) to a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                out.append(item.get("text", ""))
            elif isinstance(item, str):
                out.append(item)
        return "".join(out)
    return ""


def parse_sections(content):
    """Extract {think,plan,step,memory,action} from a tag-formatted assistant string."""
    sections = dict((k, "") for k in TAGS)
    for name, pat in _SECTION_PATTERNS.items():
        m = re.search(pat, content, re.DOTALL)
        if m:
            sections[name] = m.group(1).strip()
    return sections


def sections_to_content(sections):
    """Inverse of parse_sections: render only the non-empty tags, converter-compatible."""
    parts = []
    if sections.get("think"):
        parts.append("<think>\n%s\n</think>" % sections["think"])
    if sections.get("plan"):
        parts.append("<plan>\n%s\n</plan>" % sections["plan"])
    if sections.get("step") != "" and sections.get("step") is not None:
        parts.append("<step>%s</step>" % sections["step"])
    if sections.get("memory"):
        parts.append("<memory>\n%s\n</memory>" % sections["memory"])
    if sections.get("action"):
        parts.append("<action>\n%s\n</action>" % sections["action"])
    return "\n\n".join(parts)


def _normalize_agent_output(data):
    """Return a {think,plan,step,memory,action} dict from either dump format."""
    if not isinstance(data, dict):
        return dict((k, "") for k in TAGS)
    if "content" in data:
        return parse_sections(get_text(data.get("content", "")))
    sec = dict((k, "") for k in TAGS)
    for k in TAGS:
        if data.get(k) is not None:
            sec[k] = str(data.get(k)).strip()
    return sec


def _step_num(path):
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def extract_axtree(user_text):
    m = re.search(r"## AXTree:.*?(?=\n## |\Z)", user_text, re.DOTALL)
    return m.group(0) if m else ""


def extract_goal(user_text):
    """Pull the '## Goal:' block (best-effort) out of a rendered user prompt."""
    m = re.search(r"## Goal:\s*(.*?)(?=\n## |\Z)", user_text, re.DOTALL)
    return m.group(1).strip() if m else ""


def task_id_from_dir(episode_dir):
    m = re.search(r"timewarp\.(\d+)", os.path.basename(os.path.normpath(episode_dir)))
    return int(m.group(1)) if m else None


def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_episode(episode_dir):
    """Load one episode's dumps into a dict; env-free. Returns None if unreadable."""
    td = os.path.join(episode_dir, "training_data")
    if not os.path.isdir(td):
        return None
    users, systems, outputs = {}, {}, {}
    for path in glob.glob(os.path.join(td, "*.json")):
        n = _step_num(path)
        if n < 0:
            continue
        base = os.path.basename(path)
        data = _load_json(path)
        if data is None:
            continue
        if base.startswith("user_prompt"):
            users[n] = data
        elif base.startswith("system_prompt"):
            systems[n] = data
        elif base.startswith("agent_output"):
            outputs[n] = data
    step_ids = sorted(set(list(users) + list(systems) + list(outputs)))
    if not step_ids:
        return None

    system_text = ""
    if step_ids[0] in systems:
        system_text = get_text(systems[step_ids[0]].get("content", ""))

    steps = []
    for n in step_ids:
        user_text = get_text(users[n].get("content", "")) if n in users else ""
        sec = _normalize_agent_output(outputs.get(n, {}))
        steps.append(
            {
                "step": n,
                "user_text": user_text,
                "axtree": extract_axtree(user_text),
                "goal": extract_goal(user_text),
                "sections": sec,
            }
        )

    summary = _load_json(os.path.join(episode_dir, "summary_info.json")) or {}
    return {
        "dir": episode_dir,
        "task_id": task_id_from_dir(episode_dir),
        "system": system_text,
        "steps": steps,
        "n_steps": summary.get("n_steps", len(steps)),
        "reward": summary.get("cum_reward", None),
        "err_msg": summary.get("err_msg", None),
        "summary": summary,
    }


def is_send_msg(action):
    a = (action or "").strip()
    return a.startswith("send_msg_to_user(") or a.startswith("send_message_to_user(")


def submitted_answer(episode):
    """Reconstruct the answer the judge would see, matching task.validate() semantics."""
    for st in reversed(episode["steps"]):
        act = st["sections"].get("action", "")
        if is_send_msg(act):
            m = re.search(r"send_msg(?:age)?_to_user\(\s*(['\"])(.*?)\1\s*\)", act, re.DOTALL)
            if m:
                return m.group(2)
            m = re.search(r"\((.*)\)", act, re.DOTALL)
            return m.group(1).strip().strip("'\"") if m else ""
    return ""


def axtree_upto(episode, t):
    return "\n".join(st["axtree"] for st in episode["steps"] if st["step"] <= t)


def write_agent_output(episode_dir, step, sections):
    """Write BOTH agent_output_step_N.{json,txt} in converter+filter-compatible tag format."""
    td = os.path.join(episode_dir, "training_data")
    if not os.path.isdir(td):
        os.makedirs(td)
    content = sections_to_content(sections)
    with open(os.path.join(td, "agent_output_step_%d.json" % step), "w") as f:
        json.dump({"role": "assistant", "content": content}, f, indent=2)
    with open(os.path.join(td, "agent_output_step_%d.txt" % step), "w") as f:
        f.write(content)


def copy_prompts(src_dir, dst_dir, step):
    """Copy the system_prompt/user_prompt dumps for a step (they are inputs, not edited)."""
    src_td = os.path.join(src_dir, "training_data")
    dst_td = os.path.join(dst_dir, "training_data")
    if not os.path.isdir(dst_td):
        os.makedirs(dst_td)
    for kind in ("system_prompt", "user_prompt"):
        for ext in ("json", "txt"):
            s = os.path.join(src_td, "%s_step_%d.%s" % (kind, step, ext))
            if os.path.isfile(s):
                with (
                    open(s, "rb") as fi,
                    open(os.path.join(dst_td, os.path.basename(s)), "wb") as fo,
                ):
                    fo.write(fi.read())


if __name__ == "__main__":
    import sys

    ep = load_episode(sys.argv[1])
    if ep is None:
        print("could not load", sys.argv[1])
        sys.exit(1)
    print(
        "task_id=%s reward=%s n_steps=%s err=%s"
        % (ep["task_id"], ep["reward"], ep["n_steps"], ep["err_msg"])
    )
    for st in ep["steps"]:
        act = st["sections"]["action"].replace("\n", " ")[:70]
        print("  step %d | axtree=%dch | action=%s" % (st["step"], len(st["axtree"]), act))
    print("submitted_answer:", repr(submitted_answer(ep)[:120]))
