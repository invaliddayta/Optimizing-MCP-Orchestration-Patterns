from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from lab.mcp import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG = ROOT / "config" / "servers_config.json"
CALCULATOR = ROOT / "mcpservers" / "calculator_mcp.py"

HANG_SCRIPT = """
import os
import sys
import time
from pathlib import Path
Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
time.sleep(3600)
"""

COLLIDE_A = """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("a")
@mcp.tool(name="b__c")
def f() -> str:
    return "a"
if __name__ == "__main__":
    mcp.run()
"""

COLLIDE_B = """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("ab")
@mcp.tool(name="c")
def f() -> str:
    return "b"
if __name__ == "__main__":
    mcp.run()
"""

IMAGE_SCRIPT = """
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
mcp = FastMCP("img")
@mcp.tool()
def picture() -> list:
    return [ImageContent(type="image", data="QQ==", mimeType="image/png")]
if __name__ == "__main__":
    mcp.run()
"""

EMPTY_STRUCT_SCRIPT = """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("empty")
@mcp.tool()
def blank() -> dict:
    return {}
if __name__ == "__main__":
    mcp.run()
"""

DIE_AFTER_READY = """
import os
import threading
import time
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("die")

@mcp.tool()
def ping() -> str:
    return "pong"

def _die():
    time.sleep(1.5)
    os._exit(1)

threading.Thread(target=_die, daemon=True).start()
if __name__ == "__main__":
    mcp.run()
"""


def _write_config(directory: Path, servers: dict, defaults: dict | None = None) -> Path:
    payload: dict = {"servers": servers}
    if defaults is not None:
        payload["defaults"] = defaults
    path = directory / "servers_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class ConfigValidationTests(unittest.TestCase):
    def test_missing_config(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ToolRegistry(Path("/no/such/servers_config.json"))

    def test_unknown_keys_and_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "servers_config.json"
            path.write_text(json.dumps({"servers": {}, "extra": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                ToolRegistry(path)
            path.write_text(
                json.dumps({"servers": {"bad name": {"command": "x"}}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                ToolRegistry(path)
            path.write_text(
                json.dumps({"servers": {"ok": {"command": "x", "nope": 1}}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                ToolRegistry(path)

    def test_unknown_role_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), {})
            registry = ToolRegistry(path)
            self.assertEqual(registry.roles, [])
            with self.assertRaises(ValueError):
                registry.tools("calculator-expert")
            with self.assertRaises(ValueError):
                ToolRegistry(path, timeout_s=0)
            with self.assertRaises(ValueError):
                ToolRegistry(path, timeout_s=float("inf"))
            with self.assertRaises(ValueError):
                ToolRegistry(path, timeout_s=float("nan"))

    def test_roles_from_repo_config(self) -> None:
        registry = ToolRegistry(REPO_CONFIG)
        self.assertTrue(registry._config_path.is_absolute())
        self.assertEqual(
            registry.roles,
            [
                "calculator-expert",
                "inventory-expert",
                "stockmarket-expert",
                "unit-conversion-expert",
            ],
        )
        for params in registry._params.values():
            self.assertTrue(Path(params.cwd).is_absolute())


class ToolRegistryProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_calculator_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(
                Path(tmp),
                {
                    "calculator-expert": {
                        "command": "${PYTHON}",
                        "args": [str(CALCULATOR)],
                    }
                },
            )
            async with ToolRegistry(path, timeout_s=15) as registry:
                tools = registry.tools()
                names = [item["function"]["name"] for item in tools]
                self.assertEqual(names, ["calculator-expert__calculate"])
                self.assertEqual(names, sorted(names))
                spec = tools[0]["function"]["parameters"]
                self.assertEqual(spec.get("type"), "object")
                self.assertIn("expression", spec.get("properties", {}))
                with self.assertRaises(ValueError):
                    registry.tools("missing-role")
                with self.assertRaises(ValueError):
                    await registry.call("calculate", {"expression": "1+1"})
                result = await registry.call(
                    "calculator-expert__calculate",
                    {"expression": "1+2*3"},
                )
                self.assertFalse(result.is_error)
                self.assertIn("7", result.content)
                self.assertIn("{", result.content)
                zero = await registry.call(
                    "calculator-expert__calculate",
                    {"expression": "0"},
                )
                self.assertFalse(zero.is_error)
                self.assertIn("0", zero.content)
                self.assertIn("{", zero.content)

    async def test_plain_text_mcp_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(
                Path(tmp),
                {
                    "calculator-expert": {
                        "command": "${PYTHON}",
                        "args": [str(CALCULATOR)],
                    }
                },
            )
            async with ToolRegistry(path, timeout_s=15) as registry:
                result = await registry.call(
                    "calculator-expert__calculate",
                    {"expression": "1/0"},
                )
                self.assertTrue(result.is_error)
                self.assertTrue(result.content)
                self.assertNotEqual(result.content, "Tool error")

    async def test_failed_spawn_closes_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pid_path = directory / "pid"
            hang = directory / "hang.py"
            hang.write_text(HANG_SCRIPT, encoding="utf-8")
            path = _write_config(
                directory,
                {
                    "hang-server": {
                        "command": "${PYTHON}",
                        "args": [str(hang), str(pid_path)],
                    },
                    "missing-server": {
                        "command": "/no/such/mcp-server-binary",
                        "args": [],
                    },
                },
            )
            registry = ToolRegistry(path, timeout_s=5)
            with self.assertRaises((RuntimeError, TimeoutError, OSError)):
                await registry.__aenter__()
            if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip():
                pid = int(pid_path.read_text(encoding="utf-8"))
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and _alive(pid):
                    await asyncio.sleep(0.05)
                self.assertFalse(_alive(pid))

    async def test_hanging_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            hang = directory / "hang.py"
            hang.write_text(HANG_SCRIPT, encoding="utf-8")
            pid_path = directory / "pid"
            path = _write_config(
                directory,
                {
                    "hang-server": {
                        "command": "${PYTHON}",
                        "args": [str(hang), str(pid_path)],
                    }
                },
            )
            registry = ToolRegistry(path, timeout_s=1)
            start = time.monotonic()
            with self.assertRaises((TimeoutError, RuntimeError)):
                await registry.__aenter__()
            self.assertLess(time.monotonic() - start, 12)
            if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip():
                pid = int(pid_path.read_text(encoding="utf-8"))
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and _alive(pid):
                    await asyncio.sleep(0.05)
                self.assertFalse(_alive(pid))

    async def test_cancellation_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            hang = directory / "hang.py"
            hang.write_text(HANG_SCRIPT, encoding="utf-8")
            pid_path = directory / "pid"
            path = _write_config(
                directory,
                {
                    "hang-server": {
                        "command": "${PYTHON}",
                        "args": [str(hang), str(pid_path)],
                    }
                },
            )
            registry = ToolRegistry(path, timeout_s=30)
            task = asyncio.create_task(registry.__aenter__())
            for _ in range(100):
                if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip():
                    break
                await asyncio.sleep(0.05)
            else:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.fail("hang server did not start")
            pid = int(pid_path.read_text(encoding="utf-8"))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and _alive(pid):
                await asyncio.sleep(0.05)
            self.assertFalse(_alive(pid))

    async def test_namespace_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            left = directory / "a.py"
            right = directory / "b.py"
            left.write_text(COLLIDE_A, encoding="utf-8")
            right.write_text(COLLIDE_B, encoding="utf-8")
            path = _write_config(
                directory,
                {
                    "a": {"command": "${PYTHON}", "args": [str(left)]},
                    "a__b": {"command": "${PYTHON}", "args": [str(right)]},
                },
            )
            registry = ToolRegistry(path, timeout_s=15)
            with self.assertRaises(ValueError):
                await registry.__aenter__()

    async def test_nontext_block_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            script = directory / "img.py"
            script.write_text(IMAGE_SCRIPT, encoding="utf-8")
            path = _write_config(
                directory,
                {"img-server": {"command": "${PYTHON}", "args": [str(script)]}},
            )
            async with ToolRegistry(path, timeout_s=15) as registry:
                result = await registry.call("img-server__picture", {})
                self.assertFalse(result.is_error)
                payload = json.loads(result.content)
                self.assertTrue(payload)
                blob = payload[0] if isinstance(payload, list) else payload
                self.assertEqual(blob.get("type"), "image")

    async def test_structured_empty_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            script = directory / "empty.py"
            script.write_text(EMPTY_STRUCT_SCRIPT, encoding="utf-8")
            path = _write_config(
                directory,
                {"empty-server": {"command": "${PYTHON}", "args": [str(script)]}},
            )
            async with ToolRegistry(path, timeout_s=15) as registry:
                result = await registry.call("empty-server__blank", {})
                self.assertFalse(result.is_error)
                self.assertIn("{}", result.content)

    async def test_repo_servers_config(self) -> None:
        async with ToolRegistry(REPO_CONFIG, timeout_s=20) as registry:
            names = [item["function"]["name"] for item in registry.tools()]
            self.assertEqual(names, sorted(names))
            self.assertIn("calculator-expert__calculate", names)
            self.assertIn("inventory-expert__get_inventory", names)
            calc = [item for item in registry.tools("calculator-expert")]
            self.assertEqual(
                [item["function"]["name"] for item in calc], ["calculator-expert__calculate"]
            )
            result = await registry.call(
                "calculator-expert__calculate",
                {"expression": "(2+3)*4"},
            )
            self.assertFalse(result.is_error)
            self.assertIn("20", result.content)

    async def test_short_timeout_still_reaps_hang(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            hang = directory / "hang.py"
            hang.write_text(HANG_SCRIPT, encoding="utf-8")
            pid_path = directory / "pid"
            path = _write_config(
                directory,
                {
                    "hang-server": {
                        "command": "${PYTHON}",
                        "args": [str(hang), str(pid_path)],
                    }
                },
            )
            registry = ToolRegistry(path, timeout_s=0.5)
            with self.assertRaises((TimeoutError, RuntimeError)):
                await registry.__aenter__()
            self.assertTrue(pid_path.exists())
            pid = int(pid_path.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and _alive(pid):
                await asyncio.sleep(0.05)
            self.assertFalse(_alive(pid))

    async def test_dead_server_fails_call_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            script = directory / "die.py"
            script.write_text(DIE_AFTER_READY, encoding="utf-8")
            path = _write_config(
                directory,
                {"die-server": {"command": "${PYTHON}", "args": [str(script)]}},
            )
            async with ToolRegistry(path, timeout_s=8) as registry:
                self.assertEqual(
                    [item["function"]["name"] for item in registry.tools()],
                    ["die-server__ping"],
                )
                await asyncio.sleep(2.5)
                start = time.monotonic()
                result = await registry.call("die-server__ping", {})
                self.assertLess(time.monotonic() - start, 2)
                self.assertTrue(result.is_error)
                self.assertTrue(result.content)

    async def test_config_resolve_survives_chdir(self) -> None:
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(
                Path(tmp),
                {
                    "calculator-expert": {
                        "command": "${PYTHON}",
                        "args": [str(CALCULATOR)],
                    }
                },
            )
            registry = ToolRegistry(path, timeout_s=15)
            os.chdir("/")
            try:
                async with registry:
                    result = await registry.call(
                        "calculator-expert__calculate",
                        {"expression": "2+2"},
                    )
                    self.assertFalse(result.is_error)
                    self.assertIn("4", result.content)
            finally:
                os.chdir(old)

    async def test_stuck_owner_reported_and_retained(self) -> None:
        import lab.mcp as mcp_mod

        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(_write_config(Path(tmp), {}), timeout_s=0.05)
            loop = asyncio.get_running_loop()

            async def stubborn() -> None:
                try:
                    await loop.create_future()
                except asyncio.CancelledError:
                    await loop.create_future()

            task = asyncio.create_task(stubborn(), name="mcp:stuck")
            registry._tasks["stuck"] = task
            registry._parked.add("stuck")
            previous = mcp_mod._STDIO_CLOSE_S
            mcp_mod._STDIO_CLOSE_S = 0.05
            try:
                with self.assertRaises(TimeoutError) as ctx:
                    await registry._shutdown()
            finally:
                mcp_mod._STDIO_CLOSE_S = previous
            self.assertIn("stuck", str(ctx.exception))
            self.assertIs(registry._tasks.get("stuck"), task)
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_startup_failure_preserves_primary_cleanup_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(
                Path(tmp),
                {"missing-server": {"command": "/no/such/mcp-server-binary", "args": []}},
            )
            registry = ToolRegistry(path, timeout_s=5)

            async def fail_cleanup() -> None:
                raise TimeoutError("timed out closing MCP servers: ['missing-server']")

            registry._shutdown = fail_cleanup
            with self.assertRaises(RuntimeError) as ctx:
                await registry.__aenter__()
            exc = ctx.exception
            self.assertIn("failed to start", str(exc))
            notes = getattr(exc, "__notes__", [])
            self.assertTrue(any("cleanup" in note.lower() for note in notes))
            self.assertIsInstance(exc.__cause__, TimeoutError)
