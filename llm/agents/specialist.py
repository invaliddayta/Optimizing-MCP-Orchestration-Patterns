import asyncio
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import aiohttp
from mcp import ClientSession

from tools.callparser import CallParser
from tools.toolresolver import ToolResolver, ToolInventory
from llm.llmclient import LLMClient
from llm.agents.base import BaseAgent

@dataclass
class SpecialistResult:
    response: str
    total_prompt_tokens: int
    total_completion_tokens: int
    history: List[Dict[str, Any]] = None 

class SpecialistAgent(BaseAgent):
    def __init__(
        self,
        *,
        llm: LLMClient,
        system_prompt: str,
        call_parser: Optional[CallParser] = None,
        tool_resolver: Optional[ToolResolver] = None,
        max_steps: int = 2,
    ):
        super().__init__(llm=llm, system_prompt=system_prompt, max_steps=max_steps)
        self.parser = call_parser or CallParser()
        self.resolver = tool_resolver or ToolResolver()

    async def solve(
        self, 
        *, 
        session: ClientSession, 
        orch_instr: str, 
        http_session: aiohttp.ClientSession,
        cached_inventory: Optional[str] = None, 
        cached_tools: Optional[List[Any]] = None
    ) -> SpecialistResult:
        
        if cached_tools is not None:
            tools = cached_tools
        else:
            tools_desc = await session.list_tools()
            tools = tools_desc.tools or []

        if cached_inventory is not None:
            inventory = cached_inventory
        else:
            cards = ToolInventory.from_mcp(tools)
            inventory = ToolInventory.as_inventory_json(cards)

        messages = [
            {"role": "system", "content": self.system_prompt + "\n\nAvailable tools:\n" + inventory},
            {"role": "user", "content": orch_instr},
        ]
        
        total_p = 0
        total_c = 0

        for _ in range(self.max_steps):
            resp = await self.llm.chat(messages=messages, session=http_session)
            out = resp.content.strip()
            total_p += resp.prompt_tokens
            total_c += resp.completion_tokens
            
            imperative, tname_req, targs = self.parser.parse_call(out)
            
            # return if final answer
            if imperative == "final":
                return SpecialistResult(out, total_p, total_c, history=messages)
                
            # return if chat/reasoning only
            if imperative != "call":
                return SpecialistResult(out, total_p, total_c, history=messages)

            # --- tool execution logic ---
            real_tname = None

            if tname_req:
                real_tname = self.resolver.resolve(tname_req, tools)
                if not real_tname:
                    messages.append({"role": "user", "content": f"SYSTEM: Tool '{tname_req}' not found. Available tools: {[t.name for t in tools]}"})
                    continue
            else:
                # inference for single tool
                if len(tools) == 1:
                    real_tname = tools[0].name
                else:
                    messages.append({"role": "user", "content": "SYSTEM: You must specify the tool name. Usage: CALL: <tool_name> <args>"})
                    continue

            try:
                call = await asyncio.wait_for(session.call_tool(real_tname, arguments=targs or {}), timeout=40)
            except Exception as e:
                messages.append({"role": "user", "content": f"SYSTEM: Tool call failed: {e}"})
                continue

            tool_response = json.dumps(self.resolver.extract_payload(call))
            messages.append({"role": "tool", "content": f'TOOL_RESULT name="{real_tname}": {tool_response}'})

        return SpecialistResult("Loop limit reached.", total_p, total_c, history=messages)
