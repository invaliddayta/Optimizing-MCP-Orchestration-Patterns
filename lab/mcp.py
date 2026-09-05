from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import PROCESS_TERMINATION_TIMEOUT, get_default_environment, stdio_client
from mcp.types import CallToolResult, TextContent, Tool

from lab.agent import ToolResult

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_NAME = 64
_NS = "__"
_MAX_TOOL_PAGES = 1024
_TOP_KEYS = frozenset({"servers", "defaults"})
_DEFAULT_KEYS = frozenset({"environment"})
_SERVER_KEYS = frozenset({"command", "args", "environment", "cwd"})
_INHERIT_ENV = ("PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONUNBUFFERED")
_STDIO_CLOSE_S = float(PROCESS_TERMINATION_TIMEOUT) * 2


def _check_name(name: str, kind: str) -> str:
    if (
        not isinstance(name, str)
        or not _NAME_RE.fullmatch(name)
        or not (1 <= len(name) <= _MAX_NAME)
    ):
        raise ValueError(f"invalid {kind} name: {name!r}")
    return name


def _expand(value: str) -> str:
    if value == "${PYTHON}":
        return sys.executable
    return value


def _json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _env_map(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    out: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{where} values must be strings")
        out[key] = item
    return out


def _format_result(result: CallToolResult) -> ToolResult:
    is_error = bool(result.isError)
    texts = [block.text for block in result.content if isinstance(block, TextContent)]
    nontext = [
        block.model_dump(mode="json", by_alias=True)
        for block in result.content
        if not isinstance(block, TextContent)
    ]
    structured = result.structuredContent
    text = "\n".join(texts)
    if structured is None and not nontext:
        content = text
    else:
        parts: list[str] = []
        if structured is not None:
            parts.append(_json(structured))
        if text:
            parts.append(text)
        if nontext:
            parts.append(_json(nontext))
        content = "\n".join(parts)
    if is_error and content == "":
        content = "Tool error"
    return ToolResult(content=content, is_error=is_error)


class _OwnedSession(ClientSession):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def _receive_loop(self) -> None:
        try:
            await super()._receive_loop()
        finally:
            if not self.closed.done():
                self.closed.set_result(None)


class ToolRegistry:
    def __init__(self, config_path: Path, timeout_s: float = 30) -> None:
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be a positive finite number")
        self._timeout_s = timeout
        self._config_path = Path(config_path).expanduser().resolve()
        self._roles: list[str] = []
        self._params: dict[str, StdioServerParameters] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._failures: dict[str, BaseException] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stops: dict[str, asyncio.Future[None]] = {}
        self._booting: set[str] = set()
        self._parked: set[str] = set()
        self._dispatch: dict[str, tuple[str, str]] = {}
        self._defs: list[dict[str, Any]] = []
        self._defs_by_role: dict[str, list[dict[str, Any]]] = {}
        self._shutting = False
        self._load_config()

    @property
    def roles(self) -> list[str]:
        return list(self._roles)

    def tools(self, role: str | None = None) -> list[dict[str, Any]]:
        if role is None:
            return list(self._defs)
        if role not in self._defs_by_role:
            raise ValueError(f"unknown role: {role!r}")
        return list(self._defs_by_role[role])

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self._dispatch:
            raise ValueError(f"unknown tool: {name!r}")
        role, tool = self._dispatch[name]
        failure = self._failures.get(role)
        if failure is not None:
            return ToolResult(content=f"MCP server {role!r} failed: {failure}", is_error=True)
        session = self._sessions.get(role)
        if session is None:
            return ToolResult(content=f"MCP server {role!r} is not running", is_error=True)
        payload = arguments if arguments is not None else {}
        try:
            result = await session.call_tool(
                tool,
                payload,
                read_timeout_seconds=timedelta(seconds=self._timeout_s),
            )
        except Exception as exc:
            return ToolResult(content=str(exc), is_error=True)
        return _format_result(result)

    async def __aenter__(self) -> ToolRegistry:
        self._shutting = False
        self._failures.clear()
        self._sessions.clear()
        self._tasks.clear()
        self._stops.clear()
        self._booting.clear()
        self._parked.clear()
        try:
            await self._start()
            return self
        except BaseException as primary:
            try:
                await asyncio.shield(self._shutdown())
            except BaseException as cleanup:
                if cleanup is primary:
                    raise
                primary.add_note(f"MCP cleanup error: {cleanup!r}")
                raise primary from cleanup
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            await asyncio.shield(self._shutdown())
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(f"MCP cleanup error: {cleanup!r}")
            raise exc from cleanup

    def _load_config(self) -> None:
        path = self._config_path
        if not path.is_file():
            raise FileNotFoundError(f"config not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("config must be an object")
        extra = set(raw) - _TOP_KEYS
        if extra:
            raise ValueError(f"unknown config keys: {sorted(extra)}")
        if "servers" not in raw or not isinstance(raw["servers"], dict):
            raise ValueError("config.servers must be an object")
        defaults = raw.get("defaults", {})
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise ValueError("config.defaults must be an object")
        extra_defaults = set(defaults) - _DEFAULT_KEYS
        if extra_defaults:
            raise ValueError(f"unknown defaults keys: {sorted(extra_defaults)}")
        default_env = _env_map(defaults.get("environment", {}), "defaults.environment")
        config_dir = path.parent
        default_cwd = path.parent.parent
        roles = sorted(raw["servers"])
        for role in roles:
            _check_name(role, "server")
            spec = raw["servers"][role]
            if not isinstance(spec, dict):
                raise ValueError(f"server {role!r} must be an object")
            extra_server = set(spec) - _SERVER_KEYS
            if extra_server:
                raise ValueError(f"unknown keys for server {role!r}: {sorted(extra_server)}")
            if "command" not in spec or not isinstance(spec["command"], str) or not spec["command"]:
                raise ValueError(f"server {role!r} requires command")
            args_raw = spec.get("args", [])
            if not isinstance(args_raw, list) or any(not isinstance(a, str) for a in args_raw):
                raise ValueError(f"server {role!r} args must be a list of strings")
            env = get_default_environment()
            for key in _INHERIT_ENV:
                if key in os.environ:
                    env[key] = os.environ[key]
            bindir = str(Path(sys.executable).resolve().parent)
            path_env = env.get("PATH", "")
            parts = path_env.split(os.pathsep) if path_env else []
            if bindir not in parts:
                env["PATH"] = os.pathsep.join([bindir, *parts] if parts else [bindir])
            env.update(default_env)
            env.update(_env_map(spec.get("environment", {}), f"servers.{role}.environment"))
            cwd_raw = spec.get("cwd")
            if cwd_raw is None:
                cwd = default_cwd
            elif not isinstance(cwd_raw, str) or not cwd_raw:
                raise ValueError(f"server {role!r} cwd must be a string")
            else:
                cwd_path = Path(cwd_raw)
                cwd = cwd_path if cwd_path.is_absolute() else (config_dir / cwd_path)
            self._params[role] = StdioServerParameters(
                command=_expand(spec["command"]),
                args=[_expand(a) for a in args_raw],
                env=env,
                cwd=cwd.resolve(),
            )
        self._roles = roles
        self._defs_by_role = {role: [] for role in roles}

    async def _start(self) -> None:
        loop = asyncio.get_running_loop()
        readies: list[tuple[str, asyncio.Future[tuple[ClientSession, list[Tool]]]]] = []
        for role in self._roles:
            ready: asyncio.Future[tuple[ClientSession, list[Tool]]] = loop.create_future()
            self._booting.add(role)
            stop: asyncio.Future[None] = loop.create_future()
            self._stops[role] = stop
            task = loop.create_task(self._serve(role, ready, stop), name=f"mcp:{role}")
            self._tasks[role] = task
            readies.append((role, ready))
        gathered = await asyncio.gather(
            *[self._wait_ready(role, ready) for role, ready in readies],
            return_exceptions=True,
        )
        errors: list[BaseException] = []
        started: dict[str, tuple[ClientSession, list[Tool]]] = {}
        for item in gathered:
            if isinstance(item, asyncio.CancelledError):
                raise item
            if isinstance(item, BaseException):
                errors.append(item)
            else:
                role, payload = item
                started[role] = payload
        if errors:
            raise errors[0]
        self._index(started)

    async def _wait_ready(
        self,
        role: str,
        ready: asyncio.Future[tuple[ClientSession, list[Tool]]],
    ) -> tuple[str, tuple[ClientSession, list[Tool]]]:
        try:
            payload = await asyncio.wait_for(asyncio.shield(ready), timeout=self._timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(f"timed out initializing MCP server {role!r}") from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RuntimeError(f"MCP server {role!r} failed to start") from exc
        return role, payload

    async def _serve(
        self,
        role: str,
        ready: asyncio.Future[tuple[ClientSession, list[Tool]]],
        stop: asyncio.Future[None],
    ) -> None:
        failure: BaseException | None = None
        self._booting.add(role)
        try:
            async with stdio_client(self._params[role]) as streams:
                read, write = streams
                timeout = timedelta(seconds=self._timeout_s)
                session = _OwnedSession(read, write, read_timeout_seconds=timeout)
                async with session:
                    try:
                        await session.initialize()
                        tools = await self._list_tools(session)
                    finally:
                        self._booting.discard(role)
                    if not ready.done():
                        ready.set_result((session, tools))
                    self._parked.add(role)
                    try:
                        await asyncio.wait(
                            {stop, session.closed},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        self._parked.discard(role)
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except BaseException as exc:
            failure = exc
            if not ready.done():
                ready.set_exception(exc)
            else:
                raise
        finally:
            self._booting.discard(role)
            self._parked.discard(role)
            self._retire(role, failure)

    def _retire(self, role: str, exc: BaseException | None) -> None:
        self._sessions.pop(role, None)
        if role in self._failures:
            return
        if exc is None:
            self._failures[role] = RuntimeError(f"MCP server {role!r} closed")
        else:
            self._failures[role] = exc

    async def _list_tools(self, session: ClientSession) -> list[Tool]:
        tools: list[Tool] = []
        cursor: str | None = None
        for _ in range(_MAX_TOOL_PAGES):
            page = await session.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                return tools
        raise RuntimeError("MCP list_tools pagination exceeded page limit")

    def _index(self, started: dict[str, tuple[ClientSession, list[Tool]]]) -> None:
        dispatch: dict[str, tuple[str, str]] = {}
        defs: list[dict[str, Any]] = []
        by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in self._roles}
        for role in self._roles:
            if role in self._failures:
                raise RuntimeError(f"MCP server {role!r} failed to start") from self._failures[role]
            session, tools = started[role]
            self._sessions[role] = session
            for tool in sorted(tools, key=lambda item: item.name):
                _check_name(tool.name, "tool")
                public = f"{role}{_NS}{tool.name}"
                _check_name(public, "tool")
                if public in dispatch:
                    raise ValueError(f"tool name collision: {public!r}")
                dispatch[public] = (role, tool.name)
                schema = copy.deepcopy(tool.inputSchema) if tool.inputSchema else {"type": "object"}
                spec = {
                    "type": "function",
                    "function": {
                        "name": public,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                }
                defs.append(spec)
                by_role[role].append(spec)
        defs.sort(key=lambda item: item["function"]["name"])
        for role in by_role:
            by_role[role].sort(key=lambda item: item["function"]["name"])
        self._dispatch = dispatch
        self._defs = defs
        self._defs_by_role = by_role

    async def _shutdown(self) -> None:
        if self._shutting:
            return
        self._shutting = True
        self._sessions.clear()
        self._dispatch.clear()
        for stop in self._stops.values():
            if not stop.done():
                stop.set_result(None)
        owned = dict(self._tasks)
        parked = [owned[role] for role in self._parked if role in owned and not owned[role].done()]
        if parked:
            await asyncio.wait(parked, timeout=self._timeout_s)
        for role, task in owned.items():
            if task.done():
                continue
            if role in self._booting or role in self._parked:
                task.cancel()
        pending = [task for task in owned.values() if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=_STDIO_CLOSE_S)
        error: BaseException | None = None
        for task in owned.values():
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None and error is None:
                    error = exc
        self._tasks = {role: task for role, task in owned.items() if not task.done()}
        if self._tasks:
            timeout = TimeoutError(f"timed out closing MCP servers: {sorted(self._tasks)}")
            if error is not None:
                raise timeout from error
            raise timeout
        if error is not None:
            raise error
