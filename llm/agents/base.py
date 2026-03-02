from llm.llmclient import LLMClient

class BaseAgent:
    def __init__(self, *, llm: LLMClient, system_prompt: str, max_steps: int = 2):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_steps = max_steps
