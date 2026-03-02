import itertools
import json
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
from mcp import ClientSession

# internal
from tools.toolresolver import ToolInventory
from llm.llmclient import OllamaClient
from config.models import PipelineConfig, ExperimentSpec
from llm.agents.utils import parse_all_delegates
from llm.agents.orchestrator import OrchestratorAgent
from llm.agents.specialist import SpecialistAgent
from llm.agents.single_loop import SingleLoopAgent, AgentResult

class BaseExperimentStrategy(ABC):
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.log = {
            "turns": 0, 
            "specialist_calls": {}, 
            "prompt_tokens": 0, 
            "completion_tokens": 0, 
            "trace": []
        }

    @abstractmethod
    async def execute(
        self, 
        *, 
        eval_idx: int, 
        eval_prompt: str, 
        exp: ExperimentSpec,
        mcp_registry: Dict[str, ClientSession], 
        http_session: aiohttp.ClientSession,
        max_turns: int = 20,
        tool_cache: Optional[Dict[str, str]] = None,
        tool_objects: Optional[Dict[str, List[Any]]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        pass

    def _format_policy_injection(self, exp: ExperimentSpec, include_system_prompt: bool = False) -> str:
        # helper to generate prompt injections
        pool = self.cfg.expand_specialist_pool(exp)
        active_roles = sorted(set(item["role"] for item in pool))
        lines = []
        
        for i, role in enumerate(active_roles, 1):
            spec = self.cfg.get_role_spec(role)
            if include_system_prompt:
                # for orchestrator routing
                desc = spec.description or "Specialized agent."
                lines.append(f"- {role}: {desc}")
            elif spec.policy_injection:
                # for single loop policies
                lines.append(f"{i}) {spec.policy_injection}")
                
        if include_system_prompt:
            return "\n".join(lines)
        return "Domain policies:\n" + "\n".join(lines) if lines else "No specific policies."


class OrchestratorStrategy(BaseExperimentStrategy):
    def __init__(self, cfg: PipelineConfig):
        super().__init__(cfg)
        self._role_cycle_cache: Dict[Tuple[str, str], itertools.cycle] = {}

    def _pick_specialist(self, exp: ExperimentSpec, role: str) -> Tuple[str, int]|None:
        # load balancing logic
        key = (exp.name, role)
        if key not in self._role_cycle_cache:
            pool = self.cfg.expand_specialist_pool(exp)
            candidates = [(p["model"], p["instance"]) for p in pool if p["role"] == role]
            if not candidates: 
                return None
            self._role_cycle_cache[key] = itertools.cycle(candidates)
        return next(self._role_cycle_cache[key])

    async def execute(
        self, *, eval_idx: int, eval_prompt: str, exp: ExperimentSpec,
        mcp_registry: Dict[str, ClientSession], http_session: aiohttp.ClientSession,
        max_turns: int = 20, tool_cache=None, tool_objects=None
    ) -> Tuple[str, Dict[str, Any]]:
        
        # reset log
        self.log = {k: 0 if isinstance(v, int) else ([] if isinstance(v, list) else {}) for k, v in self.log.items()}
        
        # fallback inventory generation
        role_inventories = tool_cache if tool_cache else {}
        if not role_inventories:
            for role, session in mcp_registry.items():
                try:
                    tools_desc = await session.list_tools()
                    cards = ToolInventory.from_mcp(tools_desc.tools or [])
                    role_inventories[role] = ToolInventory.as_inventory_json(cards)
                except Exception:
                    role_inventories[role] = ""

        delegation_log = set()
        orch_llm = OllamaClient(model=exp.orchestrator_model)
        orchestrator = OrchestratorAgent(llm=orch_llm, system_prompt=self.cfg.prompts.orchestrator)
        roles_inject = self._format_policy_injection(exp, include_system_prompt=True)
        spec_return = []

        for turn in range(max_turns):
            self.log["turns"] = turn + 1
            if turn == 6:
                spec_return.append(("system", "SYSTEM HINT: You have taken 6 turns. Review the information above."))

            # route
            out, _, orch_resp, orch_history = await orchestrator.route(
                eval_prompt=eval_prompt, roles_inject=roles_inject,
                spec_return=spec_return, http_session=http_session 
            )
            
            self.log["prompt_tokens"] += orch_resp.prompt_tokens
            self.log["completion_tokens"] += orch_resp.completion_tokens
            self.log["trace"].append({"turn": turn + 1, "agent": "orchestrator", "messages": orch_history, "response": out})

            all_calls = parse_all_delegates(out)
            if not all_calls: 
                return out, self.log

            spec_return.append(("orchestrator", out))

            # execute calls
            for delegate in all_calls:
                role = delegate.role
                current_call_sig = (role, json.dumps(delegate.args, sort_keys=True))
                
                # prevent loops
                if current_call_sig in delegation_log:
                    spec_return.append(("system", f"SYSTEM BLOCK: Duplicate assignment to {role}."))
                    continue
                delegation_log.add(current_call_sig)
                
                self.log["specialist_calls"][role] = self.log["specialist_calls"].get(role, 0) + 1
                spec_model = self._pick_specialist(exp, role)
                
                if not spec_model or role not in mcp_registry:
                    spec_return.append(("system", f"[Error] Role {role} invalid."))
                    continue
                
                session = mcp_registry[role]
                spec_llm = OllamaClient(model=spec_model[0])
                spec_prompt = self.cfg.get_role_spec(role).system_prompt
                specialist = SpecialistAgent(llm=spec_llm, system_prompt=spec_prompt)
                
                orch_instr = delegate.args.get("query") or delegate.args.get("_") or json.dumps(delegate.args)

                spec_res = await specialist.solve(
                    session=session, orch_instr=orch_instr, http_session=http_session,
                    cached_inventory=role_inventories.get(role)
                )
                
                self.log["prompt_tokens"] += spec_res.total_prompt_tokens
                self.log["completion_tokens"] += spec_res.total_completion_tokens
                self.log["trace"].append({"turn": turn + 1, "agent": role, "messages": spec_res.history, "response": spec_res.response})
                spec_return.append((role, spec_res.response))

        return "[FINAL] Reached max_turns without a FINAL.", self.log


class SingleLoopStrategy(BaseExperimentStrategy):
    async def execute(
        self, *, eval_idx: int, eval_prompt: str, exp: ExperimentSpec,
        mcp_registry: Dict[str, ClientSession], http_session: aiohttp.ClientSession,
        max_turns: int = 20, tool_cache: Optional[Dict[str, str]] = None, 
        tool_objects: Optional[Dict[str, List[Any]]] = None
    ) -> Tuple[str, Dict[str, Any]]:

        self.log = {k: 0 if isinstance(v, int) else ([] if isinstance(v, list) else {}) for k, v in self.log.items()}
        
        policies_inject = self._format_policy_injection(exp, include_system_prompt=False)
        single_loop_llm = OllamaClient(model=exp.orchestrator_model)
        
        single_loop_agent = SingleLoopAgent(llm=single_loop_llm, system_prompt=self.cfg.prompts.orchestrator, max_steps=max_turns)
        
        # grab active session
        active_session = list(mcp_registry.values())[0] if mcp_registry else None
        
        # global cache keys
        cached_inv_str = tool_cache.get("global") if tool_cache else None
        cached_tools_list = tool_objects.get("global") if tool_objects else None

        result: AgentResult = await single_loop_agent.solve(
            session=active_session, prompt=eval_prompt, http_session=http_session,
            cached_inventory=cached_inv_str, cached_tools=cached_tools_list,
            policies_inject=policies_inject 
        )

        self.log["turns"] = result.turns
        self.log["specialist_calls"] = result.tool_counts
        self.log["prompt_tokens"] = result.prompt_tokens
        self.log["completion_tokens"] = result.completion_tokens
        self.log["trace"].append({"turn": 1, "agent": "single-loop", "messages": result.history, "response": result.response})
        
        return result.response, self.log
