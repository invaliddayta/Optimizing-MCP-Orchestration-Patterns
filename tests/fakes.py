import copy
import json

from lab.model import ChatResponse

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
