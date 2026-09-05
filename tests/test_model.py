import asyncio
import unittest

import aiohttp
from aiohttp import web

from lab.model import ChatClient, ModelError
from tests.fakes import call, response


class ModelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.reply = response()
        self.status = 200
        self.delay = 0
        self.requests = []

        async def handler(request):
            self.requests.append(await request.json())
            if self.delay:
                await asyncio.sleep(self.delay)
            return web.json_response(
                {
                    "choices": [
                        {"message": self.reply.message, "finish_reason": self.reply.finish_reason}
                    ],
                    "usage": {
                        "prompt_tokens": self.reply.prompt_tokens,
                        "completion_tokens": self.reply.completion_tokens,
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
