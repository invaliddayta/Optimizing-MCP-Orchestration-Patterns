import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

# ---------------------------
# normalization helpers
# ---------------------------

def _extract_tools(result) -> List[Any]:
    if result is None:
        return []
    
    # check standard attributes and dict keys
    if hasattr(result, "tools") and result.tools is not None:
        return result.tools
    if isinstance(result, dict) and "tools" in result:
        return result["tools"]
    
    # check tuple/list variations
    if isinstance(result, tuple):
        first = result[0] if result else []
        if hasattr(first, "tools"): return getattr(first, "tools")
        if isinstance(first, list): return first
        if isinstance(first, dict) and "tools" in first: return first["tools"]
        
    if isinstance(result, list):
        return result
    if hasattr(result, "items") and result.items is not None:
        return list(result.items)
        
    return []

def _extract_resources(result) -> List[Any]:
    if result is None:
        return []

    if hasattr(result, "resources") and result.resources is not None:
        return result.resources
    if isinstance(result, dict) and "resources" in result:
        return result["resources"]

    if isinstance(result, tuple):
        first = result[0] if result else []
        if hasattr(first, "resources"): return getattr(first, "resources")
        if isinstance(first, list): return first
        if isinstance(first, dict) and "resources" in first: return first["resources"]

    if isinstance(result, list):
        return result
    if hasattr(result, "items") and result.items is not None:
        return list(result.items)

    return []

def _get_name(tool_obj) -> Optional[str]:
    n = getattr(tool_obj, "name", None)
    if isinstance(n, str): return n
    if isinstance(tool_obj, dict): return tool_obj.get("name")
    if isinstance(tool_obj, tuple) and tool_obj and isinstance(tool_obj[0], str): return tool_obj[0]
    return None

def _get_uri(res_obj) -> Optional[str]:
    u = getattr(res_obj, "uri", None)
    if isinstance(u, str): return u
    if isinstance(res_obj, dict): return res_obj.get("uri")
    return None

def _clone_with_name(tool_obj, name: str):
    # try pydantic copy then dict mutation
    for attr in ("model_copy", "copy"):
        if hasattr(tool_obj, attr):
            try: return getattr(tool_obj, attr)(update={"name": name})
            except: pass
    
    if isinstance(tool_obj, dict):
        d = dict(tool_obj)
        d["name"] = name
        return d
        
    try:
        tool_obj.name = name
        return tool_obj
    except:
        raise TypeError(f"Unable to rename tool descriptor: {tool_obj}")

# ---------------------------
# multi-server proxy
# ---------------------------

class MultiServerSession:
    def __init__(
        self,
        params_by_role: Dict[str, StdioServerParameters],
        *,
        namespace_collisions: bool = True,
        namespace_sep: str = ":",
    ):
        self.params_by_role = params_by_role
        self.namespace_collisions = namespace_collisions
        self.namespace_sep = namespace_sep

        self._sessions: Dict[str, ClientSession] = {}
        self._server_tasks: Dict[str, asyncio.Task] = {}
        self._ready_events: Dict[str, asyncio.Event] = {}
        self._stop_events: Dict[str, asyncio.Event] = {}

        self._tool_index: Dict[str, Tuple[str, str]] = {}
        self._tool_index_unqualified: Dict[str, Tuple[str, str]] = {}

    async def _server_task(self, role: str, params: StdioServerParameters, ready: asyncio.Event, stop: asyncio.Event):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._sessions[role] = session
                ready.set()
                await stop.wait()

    async def __aenter__(self):
        # start tasks
        for role, params in self.params_by_role.items():
            ready = asyncio.Event()
            stop = asyncio.Event()
            self._ready_events[role] = ready
            self._stop_events[role] = stop
            self._server_tasks[role] = asyncio.create_task(
                self._server_task(role, params, ready, stop),
                name=f"mcp-server-{role}",
            )

        await asyncio.gather(*[ev.wait() for ev in self._ready_events.values()])
        await self._index_all_tools()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        for stop in self._stop_events.values():
            stop.set()
        await asyncio.gather(*self._server_tasks.values(), return_exceptions=True)

    async def _index_all_tools(self):
        per_role_tools: Dict[str, List[Any]] = {}

        async def fetch(role: str, sess: ClientSession):
            per_role_tools[role] = _extract_tools(await sess.list_tools())

        await asyncio.gather(*[fetch(r, s) for r, s in self._sessions.items()])

        # check collisions
        counts: Dict[str, int] = {}
        for tools in per_role_tools.values():
            for t in tools:
                n = _get_name(t)
                if n: counts[n] = counts.get(n, 0) + 1

        for role, tools in per_role_tools.items():
            for t in tools:
                actual = _get_name(t)
                if not actual: continue
                
                if counts.get(actual, 0) > 1 and self.namespace_collisions:
                    public = f"{role}{self.namespace_sep}{actual}"
                    self._tool_index[public] = (role, actual)
                else:
                    self._tool_index[actual] = (role, actual)
                    self._tool_index_unqualified[actual] = (role, actual)

    async def list_tools(self):
        roles = list(self._sessions.keys())
        
        async def fetch(sess: ClientSession):
            return _extract_tools(await sess.list_tools())

        results = await asyncio.gather(*[fetch(self._sessions[r]) for r in roles])
        merged: List[Any] = []
        
        for role, tools in zip(roles, results):
            for t in tools:
                orig = _get_name(t) or "unknown"
                ns_name = f"{role}{self.namespace_sep}{orig}"
                
                # prefer existing index mapping
                exposed = ns_name if ns_name in self._tool_index else orig
                merged.append(_clone_with_name(t, exposed))

        return SimpleNamespace(tools=merged)

    async def call_tool(self, name: str, arguments: Any):
        # check index
        if name in self._tool_index:
            role, actual = self._tool_index[name]
            return await self._sessions[role].call_tool(actual, arguments)

        if name in self._tool_index_unqualified:
            role, actual = self._tool_index_unqualified[name]
            return await self._sessions[role].call_tool(actual, arguments)

        # fallback: manual role parsing
        if self.namespace_sep in name:
            role, actual = name.split(self.namespace_sep, 1)
            sess = self._sessions.get(role)
            if sess:
                return await sess.call_tool(actual, arguments)

        raise ValueError(f"Tool '{name}' not found.")

    async def list_resources(self):
        roles = list(self._sessions.keys())
        
        async def fetch(sess: ClientSession):
            return _extract_resources(await sess.list_resources())

        results = await asyncio.gather(*[fetch(self._sessions[r]) for r in roles])
        merged: List[Any] = []
        
        for role, lst in zip(roles, results):
            for r in lst:
                uri = _get_uri(r) or ""
                namespaced_uri = f"{role}{self.namespace_sep}{uri}"
                
                # clone with new uri
                try:
                    if hasattr(r, "model_copy"):
                        merged.append(r.model_copy(update={"uri": namespaced_uri}))
                    elif hasattr(r, "copy"):
                        merged.append(r.copy(update={"uri": namespaced_uri}))
                    else:
                        d = dict(r)
                        d["uri"] = namespaced_uri
                        merged.append(d)
                except:
                    pass

        return SimpleNamespace(resources=merged)

    async def read_resource(self, uri: str):
        if self.namespace_sep in uri:
            role, inner = uri.split(self.namespace_sep, 1)
            if role in self._sessions:
                return await self._sessions[role].read_resource(inner)

        # try all if no role specified
        for sess in self._sessions.values():
            try: return await sess.read_resource(uri)
            except: continue
        raise ValueError(f"Resource '{uri}' not found.")

    async def ping(self):
        return await asyncio.gather(*[sess.ping() for sess in self._sessions.values()])

async def open_multi_mcp(params_by_role: Dict[str, StdioServerParameters], **kwargs):
    proxy = MultiServerSession(params_by_role, **kwargs)
    return await proxy.__aenter__()
