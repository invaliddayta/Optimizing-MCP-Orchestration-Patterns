import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web

from lab.__main__ import execute, parser
from lab.eval import read_events, summarize
from lab.mcp import ToolRegistry
from tests.fakes import call


class CLITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fail_requests = False
        self.fail_cases = False
        self.final_answer = "29060.4"
        self.requests = []

        async def handler(request):
            body = await request.json()
            self.requests.append(body)
            if self.fail_requests:
                return web.json_response({"error": "unavailable"}, status=503)
            messages = body["messages"]
            if self.fail_cases and messages[0]["content"] != "Call probe with value 7.":
                return web.json_response({"error": "case failure"}, status=500)
            pending = set()
            for message in messages:
                if message["role"] == "assistant":
                    self.assertFalse(pending, "assistant response before resolving tool calls")
                    pending.update(item["id"] for item in message.get("tool_calls", []))
                elif message["role"] == "tool":
                    self.assertIn(message["tool_call_id"], pending)
                    pending.remove(message["tool_call_id"])
            self.assertFalse(pending)
            tools = [tool["function"]["name"] for tool in body.get("tools", [])]
            steps = sum(m["role"] == "tool" for m in messages)
            calls = None
            content = self.final_answer
            if messages[0]["content"] == "Call probe with value 7.":
                if steps == 0:
                    calls = [call("probe", {"value": 7})]
                else:
                    content = "7"
            elif tools == ["delegate"]:
                if steps < 3:
                    roles = ["inventory-expert", "stockmarket-expert", "calculator-expert"]
                    calls = [
                        call(
                            "delegate",
                            {"role": roles[steps], "task": "Compute with 120 and 242.17"},
                        )
                    ]
            elif len(tools) == 1:
                name = tools[0]
                if steps == 0:
                    calls = [self.fixture_call(name)]
                else:
                    content = {
                        "inventory-expert__get_inventory": '{"stock_units":120}',
                        "stockmarket-expert__get_price_history": "242.17",
                    }.get(name, "29060.4")
            elif steps < 3:
                names = [
                    "inventory-expert__get_inventory",
                    "stockmarket-expert__get_price_history",
                    "calculator-expert__calculate",
                ]
                calls = [self.fixture_call(names[steps])]
            message = {"role": "assistant", "content": None if calls else content}
            if calls:
                message["tool_calls"] = calls
            return web.json_response(
                {
                    "choices": [
                        {"message": message, "finish_reason": "tool_calls" if calls else "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    def fixture_call(self, name):
        arguments = {
            "inventory-expert__get_inventory": {"product": "chip_module"},
            "stockmarket-expert__get_price_history": {
                "symbol": "TSLA",
                "period": "1d",
                "interval": "1d",
            },
            "calculator-expert__calculate": {"expression": "120*242.17"},
        }
        return call(name, arguments[name])

    async def test_both_modes_with_real_mcp_and_fake_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            args = parser().parse_args(
                [
                    "run",
                    "--model",
                    "fake",
                    "--base-url",
                    self.base_url,
                    "--limit",
                    "1",
                    "--output",
                    str(directory),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                code = await execute(args)
            self.assertEqual(code, 0)
            summary = summarize(directory)["modes"]
            self.assertEqual(summary["single"]["accuracy"], 1)
            self.assertEqual(summary["delegated"]["accuracy"], 1)
            self.assertEqual(summary["single"]["model_calls"], 4)
            self.assertEqual(summary["delegated"]["model_calls"], 10)
            self.assertEqual(summary["delegated"]["tool_calls_including_delegation"], 6)
            results = [e for e in read_events(directory) if e["type"] == "case_result"]
            self.assertEqual(len(results), 2)
            self.assertTrue(all(e["raw_answer"] == "29060.4" for e in results))
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertIn("flake.lock", manifest["source_sha256"])
            self.assertEqual(manifest["cases"][0]["expected"], "29060.4")

    async def test_setup_failure_preserves_planned_matrix(self):
        self.fail_requests = True
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            args = parser().parse_args(
                [
                    "run",
                    "--model",
                    "fake",
                    "--base-url",
                    self.base_url,
                    "--limit",
                    "1",
                    "--output",
                    str(directory),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(Exception):
                await execute(args)
            summary = summarize(directory)["modes"]
            self.assertEqual(summary["single"]["statuses"]["missing"], 1)
            self.assertEqual(summary["delegated"]["statuses"]["missing"], 1)
            self.assertEqual(list(read_events(directory))[-1]["type"], "suite_error")

    async def test_doctor_does_not_write_results(self):
        args = parser().parse_args(["doctor", "--model", "fake", "--base-url", self.base_url])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(await execute(args), 0)
        self.assertEqual(len(json.loads(output.getvalue())["tools"]), 4)

    async def test_failed_case_does_not_abort_remaining_cases(self):
        self.fail_cases = True
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            args = parser().parse_args(
                [
                    "run",
                    "--model",
                    "fake",
                    "--base-url",
                    self.base_url,
                    "--mode",
                    "single",
                    "--limit",
                    "2",
                    "--output",
                    str(directory),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(await execute(args), 1)
            summary = summarize(directory)
            self.assertEqual(summary["suite_status"], "finished")
            self.assertEqual(
                summary["modes"]["single"]["statuses"], {"model_error": 2, "missing": 0}
            )
            self.assertFalse(summary["modes"]["single"]["usage_complete"])

    async def test_close_error_is_not_reported_as_finished_suite(self):
        class FailingClose(ToolRegistry):
            async def __aexit__(self, *args):
                await super().__aexit__(*args)
                raise RuntimeError("close failed")

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            args = parser().parse_args(
                [
                    "run",
                    "--model",
                    "fake",
                    "--base-url",
                    self.base_url,
                    "--mode",
                    "single",
                    "--limit",
                    "1",
                    "--output",
                    str(directory),
                ]
            )
            with (
                patch("lab.__main__.ToolRegistry", FailingClose),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "close failed"),
            ):
                await execute(args)
            self.assertNotIn("suite_finished", [e["type"] for e in read_events(directory)])
            self.assertEqual(summarize(directory)["suite_status"], "failed")

    async def test_invalid_answer_has_explicit_error(self):
        self.final_answer = "FINAL: 29060.4"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            args = parser().parse_args(
                [
                    "run",
                    "--model",
                    "fake",
                    "--base-url",
                    self.base_url,
                    "--mode",
                    "single",
                    "--limit",
                    "1",
                    "--output",
                    str(directory),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(await execute(args), 1)
            result = next(e for e in read_events(directory) if e["type"] == "case_result")
            self.assertEqual(result["status"], "invalid_output")
            self.assertIn("finite number", result["error"])


class ArgumentsTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("MCP_BENCH_APP"), "packaged launcher checked by Nix")
    def test_packaged_app_ignores_cwd_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "lab.py").write_text('raise RuntimeError("cwd module imported")')
            result = subprocess.run(
                [os.environ["MCP_BENCH_APP"], "--help"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Local MCP orchestration lab", result.stdout)

    def test_invalid_limits_and_nonfinite_tolerance(self):
        for option, value in (
            ("--max-calls", "0"),
            ("--relative-tolerance", "nan"),
            ("--case-timeout", "0"),
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(["run", "--model", "fake", option, value])
