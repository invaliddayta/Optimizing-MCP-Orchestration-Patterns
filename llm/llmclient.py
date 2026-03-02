from typing import Optional, List, Dict, Protocol, Any
import os, asyncio, aiohttp, re
from dataclasses import dataclass

# =============================== Contracts ===============================

@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    
class LLMClient(Protocol):
    async def chat(self, *, messages: List[Dict[str, str]], timeout_s: int = 120) -> LLMResponse:
        """Sends an async chat request to the llm api."""
        ...
    
# ============================= Implementation ============================
class OllamaClient:
    """LLM client for a local ollama server."""
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, think_mode: bool = False) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model 
        self.think_mode = think_mode 

    @staticmethod
    def _strip_think(s: str) -> str:
        return re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", s).strip()

    async def chat(self, *, messages: list[dict[str, str]], timeout_s: int = 45, session: Optional[aiohttp.ClientSession] = None) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model, 
            "messages": messages, 
            "stream": False, 
            "think": self.think_mode,
            "options": {"temperature": 0.0}
        }

        if session:
            return await self._execute_request(session, url, payload, timeout_s)
        
        async with aiohttp.ClientSession() as temp_session:
            return await self._execute_request(temp_session, url, payload, timeout_s)

    async def _execute_request(self, session: aiohttp.ClientSession, url: str, payload: dict, timeout_s: int) -> LLMResponse:
        try:
            async with asyncio.timeout(timeout_s):
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as e:
            return LLMResponse(content=f"final: [model error] {e}", prompt_tokens=0, completion_tokens=0)

        msg = (data.get("message") or {})
        content = (msg.get("content") or "").strip()
        p_tok = data.get("prompt_eval_count", 0)
        c_tok = data.get("eval_count", 0)

        if content:
            return LLMResponse(content=self._strip_think(content), prompt_tokens=p_tok, completion_tokens=c_tok)
        return LLMResponse(content="final: [model returned no content]", prompt_tokens=p_tok, completion_tokens=c_tok)
