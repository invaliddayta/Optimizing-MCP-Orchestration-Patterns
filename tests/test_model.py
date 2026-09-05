import asyncio
import unittest

import aiohttp
from aiohttp import web

from lab.model import ChatClient, ModelError
from tests.fakes import call, protocol_reply, response


class ModelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.reply = response()
        self.status = 200
        self.delay = 0
        self.requests = []
        self.script = None

        async def handler(request):
            body = await request.json()
            self.requests.append(body)
            reply = self.script(body) if self.script else self.reply
            if self.delay:
                await asyncio.sleep(self.delay)
            return web.json_response(
                {
                    "choices": [{"message": reply.message, "finish_reason": reply.finish_reason}],
                    "usage": {
                        "prompt_tokens": reply.prompt_tokens,
                        "completion_tokens": reply.completion_tokens,
                    },
                },
                status=self.status,
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.session = aiohttp.ClientSession()
        self.client = ChatClient(self.session, f"http://127.0.0.1:{port}/v1", "test")

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()

    async def test_native_payload_and_usage(self):
        from tests.fakes import TOOL

        self.reply = response(None, [call()])
        result = await self.client.chat(messages=[], tools=[TOOL])
        self.assertEqual(result.message["tool_calls"], [call()])
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(self.requests[0]["temperature"], 0)
        self.assertFalse(self.requests[0]["stream"])
        self.assertEqual(self.requests[0]["tools"], [TOOL])

    async def test_http_failure(self):
        self.status = 404
        with self.assertRaisesRegex(ModelError, "HTTP 404"):
            await self.client.chat(messages=[], tools=[])

    async def test_timeout(self):
        self.delay = 0.05
        self.client.timeout_s = 0.005
        with self.assertRaises(TimeoutError):
            await self.client.chat(messages=[], tools=[])

    async def test_duplicate_ids_and_malformed_responses(self):
        for message in (
            {"role": "user", "content": "42"},
            {"role": "assistant", "tool_calls": [call(), call()]},
            {"role": "assistant", "tool_calls": [{"id": "broken"}]},
            {"role": "assistant", "content": [42]},
        ):
            self.reply.message = message
            with self.assertRaises(ModelError):
                await self.client.chat(messages=[], tools=[])

    async def test_preflight_rejects_text_protocol(self):
        self.reply = response('CALL: probe {"value": 7}')
        with self.assertRaisesRegex(ModelError, "preflight failed"):
            await self.client.preflight()

    async def test_preflight_checks_auto_roundtrip_and_no_tool(self):
        self.script = protocol_reply
        events = []
        result = await self.client.preflight(emit=events.append)
        self.assertTrue(result["protocol_roundtrip"])
        self.assertNotIn("native_tool_roundtrip", result)
        self.assertEqual(
            set(result["checks"]), {"forced_call", "auto_call", "tool_result", "no_tool"}
        )
        self.assertEqual([r["tool_choice"] for r in self.requests[1:]], ["auto"] * 3)
        self.assertEqual(len(self.requests[2]["messages"]), 3)
        self.assertEqual([e["stage"] for e in events], list(result["checks"]))

    async def test_forced_success_does_not_mask_auto_failure(self):
        self.script = lambda body: (
            protocol_reply(body) if isinstance(body["tool_choice"], dict) else response("7")
        )
        events = []
        with self.assertRaisesRegex(ModelError, "auto_call preflight failed"):
            await self.client.preflight(emit=events.append)
        self.assertTrue(events[0]["passed"])
        self.assertFalse(events[1]["passed"])
        self.assertEqual(events[1]["message"]["content"], "7")

    async def test_tool_result_must_be_used(self):
        self.script = lambda body: (
            response("7") if body["messages"][-1]["role"] == "tool" else protocol_reply(body)
        )
        with self.assertRaisesRegex(ModelError, "tool_result preflight failed"):
            await self.client.preflight()

    async def test_unnecessary_tool_call_fails(self):
        self.script = lambda body: (
            response(None, [call("probe", {"value": 7})])
            if "READY" in body["messages"][0]["content"]
            else protocol_reply(body)
        )
        with self.assertRaisesRegex(ModelError, "no_tool preflight failed"):
            await self.client.preflight()

    async def test_probe_http_failure_records_stage(self):
        self.status = 500
        events = []
        with self.assertRaises(ModelError):
            await self.client.preflight(emit=events.append)
        self.assertFalse(events[0]["passed"])
        self.assertEqual(events[0]["stage"], "forced_call")
