"""The small chat-completions subset used by llama-server."""

import json
import math
from dataclasses import dataclass
from urllib.parse import urlsplit

import aiohttp


class ModelError(Exception):
    pass


@dataclass
class ChatResponse:
    message: dict
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str


class ChatClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        timeout_s: float = 120,
        max_tokens: int = 1024,
        temperature: float = 0,
        seed: int = 0,
    ):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base URL must be an HTTP(S) endpoint without credentials or query")
        if not model.strip() or not math.isfinite(timeout_s) or timeout_s <= 0 or max_tokens < 1:
            raise ValueError("model, timeout and max_tokens must be valid")
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed

    async def chat(
        self, *, messages: list[dict], tools: list[dict], tool_choice: str | dict = "auto"
    ):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice, parallel_tool_calls=False)
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            ) as response:
                if response.status != 200:
                    detail = (await response.text())[:500]
                    raise ModelError(f"HTTP {response.status}: {detail}")
                data = await response.json()
        except TimeoutError:
            raise
        except (aiohttp.ClientError, ValueError) as exc:
            raise ModelError(str(exc)) from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]
            if message.get("role") != "assistant":
                raise ValueError("expected assistant message")
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("expected text content")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("expected tool_calls array")
            ids = set()
            for call in calls:
                ident = call["id"]
                function = call["function"]
                if (
                    not isinstance(ident, str)
                    or not ident
                    or ident in ids
                    or call.get("type") != "function"
                    or not isinstance(function["name"], str)
                    or not isinstance(function["arguments"], str)
                ):
                    raise ValueError("invalid or duplicate tool call")
                ids.add(ident)
            finish = choice["finish_reason"]
            if not isinstance(finish, str):
                raise ValueError("missing finish_reason")
            usage = data.get("usage") or {}
            counts = [usage.get(key) for key in ("prompt_tokens", "completion_tokens")]
            if any(n is not None and (type(n) is not int or n < 0) for n in counts):
                raise ValueError("invalid token usage")
            # Keep native reasoning separate from content and retain tool-call IDs verbatim.
            clean = {
                k: v
                for k, v in message.items()
                if k in {"role", "content", "tool_calls", "reasoning_content"}
            }
            return ChatResponse(clean, counts[0], counts[1], finish)
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            raise ModelError(f"Malformed chat response: {exc}") from exc

    async def preflight(self) -> dict:
        tool = {
            "type": "function",
            "function": {
                "name": "probe",
                "description": "Return the given integer.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
        messages = [{"role": "user", "content": "Call probe with value 7."}]
        first = await self.chat(
            messages=messages,
            tools=[tool],
            tool_choice={
                "type": "function",
                "function": {"name": "probe"},
            },
        )
        calls = first.message.get("tool_calls") or []
        try:
            valid = (
                first.finish_reason in {"stop", "tool_calls"}
                and len(calls) == 1
                and calls[0]["function"]["name"] == "probe"
                and json.loads(calls[0]["function"]["arguments"]) == {"value": 7}
            )
        except ValueError:
            valid = False
        if not valid:
            raise ModelError(
                "Tool-call preflight failed; check model and llama-server chat template"
            )
        messages.extend(
            [
                first.message,
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "content": "7",
                },
            ]
        )
        final = await self.chat(messages=messages, tools=[tool], tool_choice="none")
        if (
            final.finish_reason != "stop"
            or final.message.get("tool_calls")
            or not (final.message.get("content") or "").strip()
        ):
            raise ModelError("Tool-result preflight failed; check chat template and token limit")
        return {"base_url": self.base_url, "model": self.model, "native_tool_roundtrip": True}
