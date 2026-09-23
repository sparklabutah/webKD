"""Teacher and self-correction clients for hosted or local language models."""

import os


def load_env_key(name, env_path=None):
    """Read credentials from the process environment or an explicitly supplied file."""
    value = os.environ.get(name)
    if value or env_path is None:
        return value
    with open(env_path) as handle:
        for line in handle:
            if line.strip().startswith(name + "="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return None


def default_provider_model():
    """Pick provider+model from the CORRECTOR_* / TW_JUDGE_* env, else GPT-5.5 (OPENAI key present)."""
    provider = (
        (
            os.environ.get("WEBKD_CORRECTOR_PROVIDER")
            or os.environ.get("TW_JUDGE_PROVIDER")
            or "openai"
        )
        .strip()
        .lower()
    )
    model = os.environ.get("WEBKD_CORRECTOR_MODEL") or os.environ.get("TW_JUDGE_MODEL")
    return provider, model


class LiveCorrector(object):
    """Real LLM corrector. Reuses evaluators.py's provider logic (re-implemented, not imported,"""

    def __init__(
        self,
        provider=None,
        model=None,
        base_url=None,
        max_tokens=None,
        reasoning_effort=None,
        temperature=None,
        enable_thinking=None,
    ):
        p, m = default_provider_model()
        self.provider = (provider or p).lower()
        self.model = model or m
        if not self.model:
            raise ValueError("Set WEBKD_CORRECTOR_MODEL or pass a model explicitly")
        self.base_url = (
            base_url
            or os.environ.get("WEBKD_CORRECTOR_BASE_URL")
            or os.environ.get("TW_JUDGE_BASE_URL")
        )
        if self.base_url:
            self.provider = "local"

        self.temperature = 0.7 if temperature is None else float(temperature)
        self.enable_thinking = True if enable_thinking is None else bool(enable_thinking)

        self.max_tokens = max_tokens or int(os.environ.get("WEBKD_CORRECTOR_MAX_TOKENS", "8192"))
        self.reasoning_effort = reasoning_effort or os.environ.get(
            "WEBKD_CORRECTOR_REASONING_EFFORT", "low"
        )

    def __call__(self, system, user, max_tokens=None):

        mt = max_tokens or self.max_tokens
        messages = [
            {"role": "system", "content": system or "You are a helpful assistant"},
            {"role": "user", "content": user},
        ]

        import time
        import re
        import random

        last = None
        for attempt in range(8):
            try:
                if self.provider == "anthropic":
                    return self._anthropic(system, user, mt)
                return self._openai(messages, mt)
            except Exception as e:
                last = e
                msg = str(e).lower()
                transient = any(
                    s in msg
                    for s in (
                        "rate limit",
                        "429",
                        "timeout",
                        "timed out",
                        "temporarily",
                        "overloaded",
                        "503",
                        "502",
                        "500",
                        "connection",
                        "internal server error",
                        "try again",
                    )
                )
                if not transient or attempt == 7:
                    raise
                m = re.search(r"try again in ([\d.]+)\s*(ms|s)", msg)
                if m:
                    wait = float(m.group(1)) / (1000.0 if m.group(2) == "ms" else 1.0)
                else:
                    wait = min(2**attempt, 30)
                time.sleep(max(wait, 0.5) + random.uniform(0, 1.0 + attempt))
        raise last

    def _anthropic(self, system, user, mt):
        from anthropic import Anthropic

        key = load_env_key("ANTHROPIC_API_KEY")
        client = Anthropic(api_key=key) if key else Anthropic()
        base = dict(
            model=self.model,
            system=system or "You are a helpful assistant",
            messages=[{"role": "user", "content": user}],
        )
        try:
            resp = client.messages.create(max_tokens=mt, thinking={"type": "disabled"}, **base)
        except Exception:
            resp = client.messages.create(max_tokens=max(mt, 4096), **base)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def _openai(self, messages, mt):
        from openai import OpenAI

        if self.provider == "local":
            key = load_env_key("OPENAI_API_KEY") or "EMPTY"
            client = OpenAI(api_key=key, base_url=self.base_url)
            kw = dict(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=mt,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
            )
        else:
            key = load_env_key("OPENAI_API_KEY")
            client = OpenAI(api_key=key)
            m = str(self.model).lower()
            is_reasoning = (
                m.startswith("gpt-5")
                or m.startswith("o1")
                or m.startswith("o3")
                or m.startswith("o4")
            )
            kw = dict(model=self.model, messages=messages, max_completion_tokens=mt)
            if is_reasoning:
                kw["reasoning_effort"] = self.reasoning_effort
            else:
                kw["temperature"] = 0.7
        resp = client.chat.completions.create(**kw)
        return resp.choices[0].message.content or ""


class MockCorrector(object):
    """Deterministic scripted corrector for CPU tests / dry-runs (no network, no SDK)."""

    def __init__(self, fn=None, answer="MOCK_ANSWER"):
        self.fn = fn
        self.answer = answer
        self.calls = []

    def __call__(self, system, user, max_tokens=None):
        self.calls.append((system, user))
        if self.fn is not None:
            return self.fn(system, user)

        if "erroneous" in user.lower() and "integer step index" in user.lower():
            return "1"
        return (
            "<think>\nThe prior step looped without progress; per the observed page I can "
            "answer directly.\n</think>\n\n<action>\nsend_msg_to_user('%s')\n</action>"
            % self.answer
        )


_CORRECTOR = None


def set_corrector(c):
    global _CORRECTOR
    _CORRECTOR = c


def get_corrector():
    return _CORRECTOR if _CORRECTOR is not None else LiveCorrector()
