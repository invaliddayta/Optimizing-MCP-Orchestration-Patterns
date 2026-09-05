import asyncio
import unittest

from lab.agent import Budget, ToolResult, run_agent
from lab.model import ModelError
from tests.fakes import TOOL, FakeClient, call, response


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def run_loop(self, client, budget=None, dispatch=None):
        self.dispatched = []
        self.events = []

        async def execute(name, arguments):
            self.dispatched.append((name, arguments))
            return ToolResult("4")

        self.budget = budget or Budget()
        return await run_agent(
            client=client,
            prompt="Double 2",
            system="Use tools",
            tools=[TOOL],
            dispatch=dispatch or execute,
            budget=self.budget,
            emit=self.events.append,
        )

    async def test_transcript_keeps_assistant_and_call_id(self):
        client = FakeClient(response(None, [call()]), response("4"))
        result = await self.run_loop(client)
        self.assertEqual(result.status, "completed")
        transcript = client.requests[1]["messages"]
        self.assertEqual([m["role"] for m in transcript], ["system", "user", "assistant", "tool"])
        self.assertEqual(transcript[2]["tool_calls"][0]["id"], transcript[3]["tool_call_id"])
        self.assertEqual(self.budget.calls, 2)
        self.assertEqual(self.budget.prompt_tokens, 20)

    async def test_invalid_arguments_and_unknown_tools_never_dispatch(self):
        for requested in (call(args={"bogus": 2}), call(name="calcu"), call(args=[])):
            with self.subTest(requested=requested):
                await self.run_loop(FakeClient(response(None, [requested]), response()))
                self.assertEqual(self.dispatched, [])
                self.assertTrue(
                    next(e for e in self.events if e["type"] == "tool_result")["is_error"]
                )

    async def test_nonfinite_arguments_rejected(self):
        for number in ("NaN", "Infinity", "1e999"):
            requested = call()
            requested["function"]["arguments"] = '{"value":' + number + "}"
            await self.run_loop(FakeClient(response(None, [requested]), response()))
            self.assertEqual(self.dispatched, [])

    async def test_tool_error_recovery(self):
        async def failing(name, args):
            raise RuntimeError("tool offline")

        result = await self.run_loop(
            FakeClient(response(None, [call()]), response("4")), dispatch=failing
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.budget.tool_errors, 1)

    async def test_model_failures_are_not_answers(self):
        for error, status in ((ModelError("HTTP 404"), "model_error"), (TimeoutError(), "timeout")):
            result = await self.run_loop(FakeClient(error))
            self.assertEqual(result.status, status)
            self.assertEqual(result.answer, "")
            self.assertFalse(self.budget.usage_complete)

    async def test_empty_and_truncated_answers(self):
        result = await self.run_loop(FakeClient(response("")))
        self.assertEqual(result.status, "invalid_output")
        result = await self.run_loop(FakeClient(response("42", finish="length")))
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.answer, "42")

    async def test_multiple_tools_and_tool_budget(self):
        client = FakeClient(response(None, [call(ident="a"), call(ident="b")]), response())
        await self.run_loop(client)
        self.assertEqual(len(self.dispatched), 2)
        self.assertEqual(
            [m["tool_call_id"] for m in client.requests[1]["messages"] if m["role"] == "tool"],
            ["a", "b"],
        )
        result = await self.run_loop(
            FakeClient(response(None, [call(ident="a"), call(ident="b")])), Budget(max_tool_calls=1)
        )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(len(self.dispatched), 1)

    async def test_nested_loop_uses_shared_budget(self):
        budget = Budget(max_calls=2)
        worker = FakeClient(response("worker answer"))

        async def delegate(name, args):
            child = await run_agent(
                client=worker,
                prompt="subtask",
                system="worker",
                tools=[],
                dispatch=None,
                budget=budget,
                emit=lambda e: None,
            )
            return ToolResult(child.answer)

        manager = FakeClient(response(None, [call()]))
        result = await self.run_loop(manager, budget, delegate)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(budget.calls, 2)
        self.assertEqual(len(manager.requests), 1)
        self.assertEqual(len(worker.requests), 1)

    async def test_cancellation_propagates(self):
        async def cancelled(name, args):
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await self.run_loop(FakeClient(response(None, [call()])), dispatch=cancelled)

    async def test_missing_usage_not_silently_zero(self):
        reply = response()
        reply.prompt_tokens = None
        await self.run_loop(FakeClient(reply))
        self.assertFalse(self.budget.usage_complete)
