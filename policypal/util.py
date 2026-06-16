"""Small shared helpers (JSON extraction, tokenization)."""
import json
import re


def parse_json(text: str) -> dict:
    """Tolerantly extract a JSON object from an LLM response.

    Handles ```json fences and leading/trailing prose by grabbing the first
    balanced {...} span.
    """
    text = text.strip()
    # strip code fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # strict=False: tolerate raw control chars (literal newlines/tabs) inside string
    # values — LLMs routinely copy verbatim multi-line quotes without escaping them.
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    # fall back to first balanced object, tracking string state so braces or quote chars
    # inside a JSON string value don't throw off the depth count.
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1], strict=False)
    raise ValueError(f"unbalanced JSON in model output: {text[:200]!r}")


_WORD = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    """Lowercase word/number tokens — used by BM25 and the local embedder."""
    return _WORD.findall(s.lower())
