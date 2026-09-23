"""Score teacher targets under a student language model."""

from __future__ import print_function

import argparse
import glob
import json
import os

from webkd import trajectory as T


def mean_nll(token_logprobs):
    """Mean negative log-prob over tokens (= -mean_log_prob of confidence_scoring)."""
    lp = [float(x) for x in token_logprobs if x is not None]
    if not lp:
        return None
    return -sum(lp) / len(lp)


def extract_token_logprobs(choice_logprobs):
    """Pull per-token logprobs from an OpenAI/vLLM ChoiceLogprobs object or dict."""
    content = getattr(choice_logprobs, "content", None)
    if content is None and isinstance(choice_logprobs, dict):
        content = choice_logprobs.get("content")
    if not content:
        return []
    out = []
    for tok in content:
        lp = getattr(tok, "logprob", None)
        if lp is None and isinstance(tok, dict):
            lp = tok.get("logprob")
        if lp is not None:
            out.append(float(lp))
    return out


def score_sample_live(system, user, gpt_text, base_url, model, api_key="EMPTY"):
    """Echo the assistant turn with logprobs and return its mean NLL under the served student."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": gpt_text},
        ],
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=1,
        extra_body={"prompt_logprobs": True, "echo": True},
    )
    lp = extract_token_logprobs(getattr(resp.choices[0], "logprobs", None))
    return mean_nll(lp)


def annotate_tree(edits_root, base_url, model):
    """Add realizability NLL to each edited episode's properties.json (live; needs a server)."""
    n = 0
    for props_path in glob.glob(os.path.join(edits_root, "**", "properties.json"), recursive=True):
        ep_dir = os.path.dirname(props_path)
        ep = T.load_episode(ep_dir)
        props = json.load(open(props_path))
        t_star = props.get("t_star", 0)
        nlls = []
        for st in ep["steps"]:
            if st["step"] < t_star:
                continue
            gpt = T.sections_to_content(st["sections"])
            try:
                v = score_sample_live(ep["system"], st["user_text"], gpt, base_url, model)
                if v is not None:
                    nlls.append(v)
            except Exception as e:
                print("  [nll skip] %s step %d: %s" % (os.path.basename(ep_dir), st["step"], e))
        props["realizability_nll"] = round(sum(nlls) / len(nlls), 4) if nlls else None
        json.dump(props, open(props_path, "w"), indent=2)
        n += 1
    print("annotated realizability for %d episodes" % n)


def main():
    ap = argparse.ArgumentParser(description="Per-sample realizability NLL under the student.")
    ap.add_argument("--edits", required=True)
    ap.add_argument(
        "--base-url", default=os.environ.get("VLLM_API_URL", "http://localhost:8000/v1")
    )
    ap.add_argument("--model", required=True, help="served student model id")
    args = ap.parse_args()
    annotate_tree(args.edits, args.base_url, args.model)


if __name__ == "__main__":
    main()
