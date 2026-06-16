"""Unified LLM client across Anthropic, OpenAI, and OpenRouter — plus an offline mock.

The rest of the package speaks one provider-neutral message/tool format; this module
translates to each backend. OpenRouter is reached through the OpenAI SDK by pointing
`base_url` at the OpenRouter gateway (it is OpenAI-API-compatible).

Normalized message format (list of dicts):
    {"role": "user"|"assistant"|"tool", "content": str,
     "tool_calls": [ToolCall, ...]?,          # assistant turns that call tools
     "tool_call_id": str?}                    # tool-result turns

Tool spec (provider-neutral):
    {"name": str, "description": str, "parameters": <json schema>}
"""
import json
import re
from dataclasses import dataclass, field

from .config import Config, load_config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)   # {input_tokens, output_tokens}


class _BaseLLM:
    """Shared cumulative token accounting + common sampling parameters.

    temperature/top_p/seed default to None, meaning "don't send it" — the provider's
    own default applies. That keeps us compatible with models that reject these params
    (e.g. some reasoning models only allow the default temperature). Set them in config
    to take control (temperature=0 is the right call for grounded, repeatable RAG)."""

    def __init__(self, config: Config | None = None):
        self.total_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        self.temperature = getattr(config, "temperature", None)
        self.top_p = getattr(config, "top_p", None)
        self.seed = getattr(config, "seed", None)
        self.default_max_tokens = getattr(config, "max_tokens", 2000) or 2000

    def _sampling(self, include_seed: bool) -> dict:
        kw = {}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.top_p is not None:
            kw["top_p"] = self.top_p
        if include_seed and self.seed is not None:
            kw["seed"] = self.seed
        return kw

    def _track(self, usage: dict) -> dict:
        self.total_usage["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        self.total_usage["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        self.total_usage["calls"] += 1
        return usage


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)            # ~4 chars/token heuristic (mock / fallback)


def make_llm(config: Config | None = None):
    config = config or load_config()
    provider = config.provider
    if provider == "anthropic":
        return AnthropicLLM(config)
    if provider in ("openai", "openrouter"):
        return OpenAICompatibleLLM(config)
    if provider == "mock":
        return MockLLM(config)
    raise ValueError(f"unknown LLM_PROVIDER: {provider!r}")


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class AnthropicLLM(_BaseLLM):
    def __init__(self, config: Config):
        import anthropic

        super().__init__(config)
        self.model = config.anthropic_model
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def chat(self, messages, system=None, tools=None, max_tokens=None) -> LLMResponse:
        kwargs = {"model": self.model, "max_tokens": max_tokens or self.default_max_tokens,
                  "messages": _to_anthropic(messages), **self._sampling(include_seed=False)}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t["description"],
                 "input_schema": t["parameters"]} for t in tools
            ]
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input))
                 for b in resp.content if b.type == "tool_use"]
        usage = self._track({"input_tokens": getattr(resp.usage, "input_tokens", 0),
                             "output_tokens": getattr(resp.usage, "output_tokens", 0)})
        return LLMResponse(text=text, tool_calls=calls, usage=usage)


def _to_anthropic(messages):
    """Normalized -> Anthropic. Consecutive tool results merge into one user turn."""
    out, pending = [], []

    def flush_pending():
        if pending:
            out.append({"role": "user", "content": pending.copy()})
            pending.clear()

    for m in messages:
        if m["role"] == "tool":
            pending.append({"type": "tool_result", "tool_use_id": m["tool_call_id"],
                            "content": m["content"]})
            continue
        flush_pending()
        if m["role"] == "assistant" and m.get("tool_calls"):
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.name,
                                "input": tc.arguments})
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    flush_pending()
    return out


# --------------------------------------------------------------------------- #
# OpenAI / OpenRouter (same SDK, different base_url + key)
# --------------------------------------------------------------------------- #
class OpenAICompatibleLLM(_BaseLLM):
    def __init__(self, config: Config):
        from openai import OpenAI

        super().__init__(config)
        if config.provider == "openrouter":
            self.model = config.openrouter_model
            self.client = OpenAI(
                api_key=config.openrouter_api_key,
                base_url=config.openrouter_base_url,
                default_headers={  # optional attribution headers OpenRouter recommends
                    "HTTP-Referer": "https://github.com/policypal",
                    "X-Title": "PolicyPal",
                },
            )
        else:
            self.model = config.openai_model
            self.client = OpenAI(api_key=config.openai_api_key,
                                 base_url=config.openai_base_url)

    def chat(self, messages, system=None, tools=None, max_tokens=None) -> LLMResponse:
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(_to_openai(messages))
        kwargs = {"model": self.model, "max_tokens": max_tokens or self.default_max_tokens,
                  "messages": oai_messages, **self._sampling(include_seed=True)}
        if tools:
            kwargs["tools"] = [
                {"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["parameters"]}} for t in tools
            ]
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        u = getattr(resp, "usage", None)
        usage = self._track({"input_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
                             "output_tokens": getattr(u, "completion_tokens", 0) if u else 0})
        return LLMResponse(text=msg.content or "", tool_calls=calls, usage=usage)


def _to_openai(messages):
    out = []
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                        "content": m["content"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content") or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name,
                                  "arguments": json.dumps(tc.arguments)}}
                    for tc in m["tool_calls"]
                ],
            })
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


# --------------------------------------------------------------------------- #
# Mock — deterministic, offline. Keeps the full pipeline runnable without keys.
# --------------------------------------------------------------------------- #
def _strip_section_prefix(body: str) -> str:
    """Drop the leading "[section path]\\n" prefix that chunk/tree text carries."""
    return re.sub(r"^\[.*?\]\n", "", body)


class MockLLM(_BaseLLM):
    """Schema-valid canned responses so verify/render run end-to-end offline.

    It quotes real text out of the prompt so mechanical verification passes — the
    point is to exercise the plumbing, not to be smart. Token usage is *estimated*
    from text length so the token-count / cost feature is demonstrable without a key.
    """

    def __init__(self, config: Config | None = None):
        super().__init__(config)

    def chat(self, messages, system=None, tools=None, max_tokens=None) -> LLMResponse:
        last_user = next((m["content"] for m in reversed(messages)
                          if m["role"] == "user" and isinstance(m.get("content"), str)),
                         "")
        sys = system or ""

        if tools:
            resp = self._mock_agent(messages, tools)
        elif "<toc>" in last_user or "tree structure" in sys:
            resp = self._mock_tree_search(last_user)
        else:
            resp = self._mock_generate(last_user)

        in_tok = _est_tokens(sys) + sum(
            _est_tokens(str(m.get("content", ""))) for m in messages)
        if tools:
            in_tok += _est_tokens(json.dumps(tools))
        out_tok = _est_tokens(resp.text) + sum(
            _est_tokens(json.dumps(tc.arguments)) for tc in resp.tool_calls)
        resp.usage = self._track({"input_tokens": in_tok, "output_tokens": out_tok})
        return resp

    # -- generation: quote the first chunk's body so the citation verifies --
    def _mock_generate(self, user_text: str) -> LLMResponse:
        m = re.search(r"<chunk id='([^']+)'.*?>\n(.*?)\n</chunk>", user_text, re.DOTALL)
        if not m:
            return LLMResponse(text=json.dumps({"abstained": True, "claims": []}))
        cid, body = m.group(1), m.group(2)
        quote = " ".join(_strip_section_prefix(body).split()[:12])
        payload = {"abstained": False, "claims": [
            {"text": f"(mock) {quote}",
             "citations": [{"source_id": cid, "quote": quote}]}]}
        return LLMResponse(text=json.dumps(payload))

    def _mock_tree_search(self, user_text: str) -> LLMResponse:
        ids = re.findall(r"\b(n[0-9a-f]+|root)\b", user_text)
        first = next((i for i in ids if i != "root"), None)
        return LLMResponse(text=json.dumps(
            {"thinking": "mock", "node_ids": [first] if first else []}))

    def _mock_agent(self, messages, tools) -> LLMResponse:
        tool_names = {t["name"] for t in tools}
        # if a search result is already on the transcript, submit using it; else search
        for m in reversed(messages):
            if m["role"] == "tool":
                hit = re.search(r"\[id=([^\]]+)\].*?\n(.*)", m["content"], re.DOTALL)
                if hit and "submit_answer" in tool_names:
                    sid, body = hit.group(1), hit.group(2)
                    quote = " ".join(_strip_section_prefix(body).split()[:12])
                    return LLMResponse(text="", tool_calls=[ToolCall(
                        "call_submit", "submit_answer",
                        {"claims": [{"text": f"(mock) {quote}",
                                     "citations": [{"source_id": sid, "quote": quote}]}]})])
                break
        query = next((m["content"] for m in messages if m["role"] == "user"), "")
        return LLMResponse(text="", tool_calls=[ToolCall(
            "call_search", "search_handbook", {"query": query})])
