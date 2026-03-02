import aiohttp
from typing import Tuple, Optional, List, Dict, Any

from llm.llmclient import LLMResponse
from llm.agents.base import BaseAgent
from llm.agents.utils import parse_final_marker, parse_delegate, Delegate

class OrchestratorAgent(BaseAgent):
    async def route(
            self,
            *,
            eval_prompt: str,
            roles_inject: str,
            spec_return: List[Tuple[str, str]],
            http_session: aiohttp.ClientSession
    ) -> Tuple[str, Optional[Delegate], LLMResponse, List[Dict[str, Any]]]:
        
        injected_prompt = self.system_prompt.replace("ALL_ROLES_PLACEHOLDER", roles_inject)
        messages = [
            {"role": "system", "content": injected_prompt},
            {"role": "user", "content": eval_prompt} 
        ]
        
        if spec_return:
            for role, content in spec_return:
                if role == "orchestrator":
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": "user", "content": f"The specialist {role} has returned: {content}"})

        resp = await self.llm.chat(messages=messages, session=http_session)
        out = resp.content.strip()
        
        # check for final answer marker
        final = parse_final_marker(out)
        if final is not None:
            return final, None, resp, messages

        delegate = parse_delegate(out)
        return out, delegate, resp, messages
