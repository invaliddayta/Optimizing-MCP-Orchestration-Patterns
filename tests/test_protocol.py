import copy
import unittest

import aiohttp
from aiohttp import web

from lab.model import ChatClient
from lab.protocol import inspect_protocol
from tests.fakes import PROPS, declaration


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.props = copy.deepcopy(PROPS)
        self.status = 200
        self.paths = []

        async def props(request):
            self.paths.append(request.path)
            return web.json_response(self.props, status=self.status)

        app = web.Application()
        app.router.add_get("/prefix/props", props)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.session = aiohttp.ClientSession()
        self.client = ChatClient(self.session, f"http://127.0.0.1:{port}/prefix/v1", "fake")
        self.declaration = declaration(self.client.base_url)

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()

    async def inspect(self):
        return await inspect_protocol(self.client, {"tool_protocols": [self.declaration]})

    async def test_properties_alone_never_prove_native_training(self):
        evidence = await inspect_protocol(self.client, None)
        self.assertEqual(evidence["classification"], "unverified")
        self.assertEqual(evidence["observed"]["templates"]["chat_template"], PROPS["chat_template"])
        self.assertEqual(self.paths, ["/prefix/props"])

    async def test_complete_matching_declaration(self):
        evidence = await self.inspect()
        self.assertEqual(evidence["classification"], "native_declared")
        self.assertEqual(evidence["issues"], [])
        self.assertIn("user-declared", evidence["verification_scope"])

    async def test_generic_classification_separate(self):
        self.declaration.update(format="generic", handler="Generic")
        self.assertEqual((await self.inspect())["classification"], "generic_declared")

    async def test_mismatch_or_missing_evidence_cannot_pass(self):
        original = copy.deepcopy(self.declaration)
        for key, value in (
            ("chat_template_sha256", "b" * 64),
            ("build_info", "different"),
            ("gguf_sha256", "not-a-hash"),
            ("training_source", ""),
            ("handler_source", ""),
            ("handler", "Generic"),
            ("handler", {}),
            ("format", []),
        ):
            self.declaration = {**original, key: value}
            with self.subTest(key=key, value=value):
                self.assertEqual((await self.inspect())["classification"], "unverified")

    async def test_tool_use_template_must_also_match(self):
        self.props["chat_template_tool_use"] = "different tool template"
        evidence = await self.inspect()
        self.assertEqual(evidence["classification"], "unverified")
        self.declaration["chat_template_tool_use_sha256"] = evidence["observed"]["template_sha256"][
            "chat_template_tool_use"
        ]
        self.assertEqual((await self.inspect())["classification"], "native_declared")

    async def test_unavailable_properties_stay_unverified(self):
        self.status = 404
        evidence = await self.inspect()
        self.assertEqual(evidence["classification"], "unverified")
        self.assertTrue(any("unavailable" in issue for issue in evidence["issues"]))

    async def test_malformed_properties_stay_unverified(self):
        self.props = ["not an object"]
        self.assertEqual((await self.inspect())["classification"], "unverified")

    async def test_endpoint_and_model_must_match(self):
        self.declaration["model"] = "another-model"
        self.assertEqual((await self.inspect())["classification"], "unverified")

    async def test_duplicate_and_malformed_declarations_fail(self):
        for entries in ([self.declaration, self.declaration], ["invalid"], [{"base_url": 3}]):
            with self.assertRaises(ValueError):
                await inspect_protocol(self.client, {"tool_protocols": entries})
