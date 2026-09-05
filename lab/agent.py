"""One loop for the single agent, manager and role-scoped workers."""

import json
import math
from dataclasses import dataclass

from jsonschema import ValidationError
from jsonschema.validators import validator_for

from lab.model import ModelError


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class Budget:
    max_calls: int = 20
    max_tool_calls: int = 40
    calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_complete: bool = True


@dataclass
class AgentResult:
    status: str
    answer: str = ""
    error: str | None = None


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        _invalid_constant(value)
    return number


async def run_agent(
    *, client, prompt, system, tools, dispatch, budget, emit, agent="single", max_calls=None
) -> AgentResult:
    validators = {}
    for tool in tools:
        function = tool["function"]
        schema = function["parameters"]
        validator = validator_for(schema)
        validator.check_schema(schema)
        validators[function["name"]] = validator(schema)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    local_calls = 0
    while budget.calls < budget.max_calls and (max_calls is None or local_calls < max_calls):
        budget.calls += 1
        local_calls += 1
        emit(
            {
                "type": "model_request",
                "agent": agent,
                "model": client.model,
                "messages": messages,
                "call": budget.calls,
            }
        )
        try:
            response = await client.chat(messages=messages, tools=tools)
        except (ModelError, TimeoutError) as exc:
            status = "timeout" if isinstance(exc, TimeoutError) else "model_error"
            budget.usage_complete = False
            emit({"type": "error", "agent": agent, "status": status, "error": str(exc)})
            return AgentResult(status, error=str(exc))
        for key in ("prompt_tokens", "completion_tokens"):
            value = getattr(response, key)
            if value is None:
                budget.usage_complete = False
            else:
                setattr(budget, key, getattr(budget, key) + value)
        message = response.message
        emit(
            {
                "type": "model_response",
                "agent": agent,
                "message": message,
                "finish_reason": response.finish_reason,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            }
        )
        messages.append(message)
        answer = message.get("content") or ""
        if response.finish_reason == "length":
            return AgentResult("budget_exhausted", answer, "Model output token limit reached")
        if response.finish_reason not in {"stop", "tool_calls"}:
            return AgentResult(
                "invalid_output", answer, f"Unexpected finish: {response.finish_reason}"
            )
        calls = message.get("tool_calls") or []
        if not calls:
            if response.finish_reason != "stop" or not answer.strip():
                return AgentResult("invalid_output", answer, "Missing final answer")
            return AgentResult("completed", answer)
        for call in calls:
            if budget.tool_calls >= budget.max_tool_calls:
                return AgentResult("budget_exhausted", error="Shared tool-call limit reached")
            budget.tool_calls += 1
            name = call["function"]["name"]
            try:
                args = json.loads(
                    call["function"]["arguments"],
                    parse_constant=_invalid_constant,
                    parse_float=_finite_float,
                )
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be an object")
                if name not in validators:
                    raise ValueError(f"Unknown tool: {name}")
                validators[name].validate(args)
                result = await dispatch(name, args)
            except (ValueError, ValidationError) as exc:
                result = ToolResult(str(exc), is_error=True)
            except Exception as exc:
                result = ToolResult(f"Tool execution failed: {exc}", is_error=True)
            budget.tool_errors += int(result.is_error)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.content})
            emit(
                {
                    "type": "tool_result",
                    "agent": agent,
                    "name": name,
                    "tool_call_id": call["id"],
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )
    return AgentResult("budget_exhausted", error="Model-call limit reached")
