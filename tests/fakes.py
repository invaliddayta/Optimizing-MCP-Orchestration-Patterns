import copy
import hashlib
import json

from lab.model import ChatResponse

PROPS = {
    "chat_template": "test template with tools",
    "build_info": "b6981-test",
    "model_path": "/models/test.gguf",
}


def declaration(base_url, model="fake", format="native"):
    return {
        "base_url": base_url,
        "model": model,
        "format": format,
        "handler": "Hermes 2 Pro" if format == "native" else "Generic",
        "handler_source": "Test fixture server log",
        "training_source": "Test fixture model card",
        "gguf_sha256": "a" * 64,
        "quantization": "Q4_K_M",
        "build_info": PROPS["build_info"],
        "chat_template_sha256": hashlib.sha256(PROPS["chat_template"].encode()).hexdigest(),
        "chat_template_tool_use_sha256": None,
    }


TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    },
}


def response(content: str | None = "42", calls=None, finish=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return ChatResponse(message, 10, 2, finish or ("tool_calls" if calls else "stop"))


def call(name="calculate", args=None, ident="call-1"):
    return {
        "id": ident,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args if args is not None else {"value": 2}),
        },
    }


def protocol_reply(request):
    names = {tool["function"]["name"] for tool in request.get("tools", [])}
    if names != {"probe", "lookup_probe"}:
        return None
    messages = request["messages"]
    if isinstance(request["tool_choice"], dict):
        return response(None, [call("probe", {"value": 7})])
    if messages[-1]["role"] == "tool":
        return response(messages[-1]["content"])
    if "READY" in messages[0]["content"]:
        return response("READY")
    return response(None, [call("lookup_probe", {"key": "alpha"})])


class FakeClient:
    model = "fake"

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    async def chat(self, **request):
        self.requests.append(copy.deepcopy(request))
        reply = next(self.responses)
        if isinstance(reply, Exception):
            raise reply
        return reply
