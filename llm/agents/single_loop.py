import asyncio
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import aiohttp
from mcp import ClientSession

from tools.callparser import CallParser
from tools.toolresolver import ToolResolver, ToolInventory
from llm.llmclient import LLMClient
from llm.agents.base import BaseAgent

@dataclass
class AgentResult:
    response: str
    turns: int
    tool_counts: Dict[str, int]
    prompt_tokens: int
    completion_tokens: int
    history: List[Dict[str, Any]] = None 

class SingleLoopAgent(BaseAgent):
    def __init__(
        self,
        *,
        llm: LLMClient,
        system_prompt: str,
        call_parser: Optional[CallParser] = None,
        tool_resolver: Optional[ToolResolver] = None,
        max_steps: int = 20,
    ):
        super().__init__(llm=llm, system_prompt=system_prompt, max_steps=max_steps)
        self.parser = call_parser or CallParser()
        self.resolver = tool_resolver or ToolResolver()

    async def solve(
        self, 
        *, 
        session: ClientSession, 
        prompt: str, 
        http_session: aiohttp.ClientSession, 
        cached_inventory: Optional[str] = None, 
        cached_tools: Optional[List[Any]] = None,
        policies_inject: str = "" 
    ) -> AgentResult:
        
        # resolve tools
        if cached_tools is not None:
            tools = cached_tools
        else:
            tools_desc = await session.list_tools()
            tools = tools_desc.tools or []
        
        # resolve inventory string
        if cached_inventory is not None:
            tool_inventory = cached_inventory
        else:
            cards = ToolInventory.from_mcp(tools)
            tool_inventory = ToolInventory.as_inventory_json(cards)

        injected_system_prompt = self.system_prompt.replace("DOMAIN_POLICIES_PLACEHOLDER", policies_inject)
        messages = [
            {"role": "system", "content": injected_system_prompt + "\n\nAvailable tools:\n" + tool_inventory},
            {"role": "user", "content": prompt},
        ]

        tool_counts: Dict[str, int] = defaultdict(int)
        action_log: Dict[Tuple[str, str], str] = {}
        total_p, total_c = 0, 0

        for turn in range(self.max_steps):
            if turn == 5:
                messages.append({
                    "role": "user", 
                    "content": "SYSTEM HINT: You have taken 5 steps. Please review the 'TOOL_RESULT' history above. You likely have all the numbers you need. combine them using 'calculate' or output FINAL."
                })

            resp = await self.llm.chat(messages=messages, session=http_session)
            out = resp.content.strip()
            total_p += resp.prompt_tokens
            total_c += resp.completion_tokens

            imperative, tname_req, targs = self.parser.parse_call(out)
            if not tname_req or imperative == "final":
                return AgentResult(out, turn + 1, dict(tool_counts), total_p, total_c, history=messages)

            real_tname = self.resolver.resolve(tname_req, tools)
            if not real_tname:
                messages.append({"role": "user", "content": f"System: Tool '{tname_req}' not found. Check the Available Tools list."})
                continue 
            
            call_sig = (real_tname, json.dumps(targs, sort_keys=True))
            if call_sig in action_log:
                messages.append({
                    "role": "user", 
                    "content": f"SYSTEM BLOCK: You have ALREADY called '{real_tname}' with these exact arguments... STOP REPEATING YOURSELF."
                })
                continue

            tool_counts[real_tname] += 1
            try:
                call = await asyncio.wait_for(session.call_tool(real_tname, arguments=targs or {}), timeout=40)
            except Exception as e:
                messages.append({"role": "user", "content": f"System: Tool execution failed: {e}. Check your arguments."})
                continue
            
            payload = self.resolver.extract_payload(call)
            tool_response_str = json.dumps(payload)
            action_log[call_sig] = (tool_response_str[:100] + '...') if len(tool_response_str) > 100 else tool_response_str
            messages.append({"role": "tool", "content": f'TOOL_RESULT name="{real_tname}": {tool_response_str}'})

        return AgentResult("FINAL: Max turns reached", self.max_steps, dict(tool_counts), total_p, total_c, history=messages)
