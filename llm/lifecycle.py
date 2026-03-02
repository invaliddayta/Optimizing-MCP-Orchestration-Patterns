import aiohttp
import asyncio
import os
from typing import Optional, List

class OllamaLifecycleManager:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    async def load_model(self, model: str, session: aiohttp.ClientSession) -> None:
        """Forces the model to load into VRAM."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [],
            "keep_alive": -1 
        }
        try:
            async with session.post(url, json=payload) as resp:
                await resp.read() # Consume response
            print(f"[System] Pre-load request sent for {model}")
        except Exception as e:
            print(f"[Warning] Failed to pre-load model {model}: {e}")

    async def unload_model(self, model: str, session: aiohttp.ClientSession) -> None:
        """Explicitly unloads the model from VRAM."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [],
            "keep_alive": 0 
        }
        try:
            async with session.post(url, json=payload) as resp:
                await resp.read()
            print(f"[System] Unload request sent for {model}")
        except Exception as e:
            print(f"[Warning] Failed to unload model {model}: {e}")

    async def wait_for_unload(self, model: str, session: aiohttp.ClientSession, retries: int = 5) -> bool:
        """Polls /api/ps to verify the model is gone."""
        print(f"[System] Verifying VRAM release for {model}...")
        for _ in range(retries):
            try:
                async with session.get(f"{self.base_url}/api/ps") as resp:
                    data = await resp.json()
                    loaded_models = [m['name'] for m in data.get('models', [])]
                    if not any(model in m for m in loaded_models):
                        return True
            except Exception:
                pass
            await asyncio.sleep(1)
        return False
