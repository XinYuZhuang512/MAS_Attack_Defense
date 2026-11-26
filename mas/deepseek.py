"""Stream visible reasoning and answer chunks from the official DeepSeek endpoint."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com/chat/completions"
REASONING_EFFORT = "high"
MAX_TOKENS = int(os.environ.get("MAS_FLASH_MAX_TOKENS", "8192"))


def model_message(item):
    fields = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id")
    return {name: item[name] for name in fields if name in item}


class DeepSeek:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is unset or empty")

    def stream(self, inputs, *, tools=()):
        payload = {
            "model": MODEL,
            "messages": [model_message(item) for item in inputs],
            "thinking": {"type": "enabled"},
            "reasoning_effort": REASONING_EFFORT,
            "max_tokens": MAX_TOKENS,
            "stream": True,
        }
        if tools:
            payload["tools"] = list(tools)
        request = Request(ENDPOINT, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.api_key,
        }, data=json.dumps(payload).encode("utf-8"))
        finish_reason = None
        try:
            with urlopen(request, timeout=45) as response:
                for raw_line in response:
                    if not raw_line.startswith(b"data:"):
                        continue
                    data = raw_line[5:].strip()
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise ValueError("DeepSeek returned a stream error")
                    for choice in chunk.get("choices", []):
                        delta = choice["delta"]
                        for field in ("reasoning_content", "content", "tool_calls"):
                            if delta.get(field):
                                yield field, delta[field]
                        finish_reason = choice.get("finish_reason") or finish_reason
        except HTTPError as exc:
            raise ValueError(f"DeepSeek returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise ValueError(f"DeepSeek could not be reached: {exc.reason}") from exc
        if finish_reason not in ({"stop", "tool_calls"} if tools else {"stop"}):
            raise ValueError(f"DeepSeek stream did not finish normally: {finish_reason!r}")
