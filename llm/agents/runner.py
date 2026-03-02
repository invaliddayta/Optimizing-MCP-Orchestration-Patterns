import itertools
import json
import aiohttp
import asyncio
from typing import Dict, Tuple, List, Optional, Any
from mcp import ClientSession

# internal
# [REMOVED] from tools.asyncspinner import AsyncSpinner
from llm.llmclient import OllamaClient
from config.models import PipelineConfig, ExperimentSpec

# agents
from llm.agents.utils import parse_delegate
from llm.agents.orchestrator import OrchestratorAgent
from llm.agents.specialist import SpecialistAgent
from llm.agents.single_loop import SingleLoopAgent, AgentResult

class MultiAgentRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._role_cycle_cache: Dict[Tuple[str, str], itertools.cycle] = {}
        self.log = {"turns": 0, "specialist_calls": {}, "trace": []}

    def return_log(self):
        return self.log

    def _pick_specialist(self, exp: ExperimentSpec, role: str) -> Tuple[str, int]|None:
        key = (exp.name, role)
        if key not in self._role_cycle_cache:
            pool = self.cfg.expand_specialist_pool(exp)
            candidates = [(p["model"], p["instance"]) for p in pool if p["role"] == role]
            if not candidates: return None
            self._role_cycle_cache[key] = itertools.cycle(candidates)
        return next(self._role_cycle_cache[key])

    def _create_specialist_injection(self, exp: ExperimentSpec) -> str:
        pool = self.cfg.expand_specialist_pool(exp)
        active_roles = sorted(set(item["role"] for item in pool))
        lines = []
        for role in active_roles:
            spec = self.cfg.get_role_spec(role)
            desc = spec.description if spec.description else "Specialized agent."
            
            # IMPROVEMENT: Append the policy injection so the Orchestrator knows strictly how to use this agent.
            policy = spec.policy_injection if spec.policy_injection else ""
            if policy:
                # Add the policy details clearly to the description visible to the Orchestrator
                lines.append(f"- {role}: {desc}\n   USAGE POLICY: {policy}")
            else:
                lines.append(f"- {role}: {desc}")
                
        return "\n".join(lines)

    def _create_policy_injection(self, exp: ExperimentSpec) -> str:
        pool = self.cfg.expand_specialist_pool(exp)
        active_roles = sorted(set(item["role"] for item in pool))
        lines = []
        for i, role in enumerate(active_roles, 1):
            spec = self.cfg.get_role_spec(role)
            if spec.policy_injection:
                lines.append(f"{i}) {spec.policy_injection}")
        
        return "Domain policies (mix of all specialists):\n" + "\n".join(lines) if lines else "No specific domain policies."

    def _parse_all_delegates(self, text: str) -> list:
        calls = []
        for line in text.splitlines():
            s = line.strip()
            if s.upper().startswith("FINAL"): break
            if s.startswith("CALL:"):
                d = parse_delegate(s)
                if d: calls.append(d)
        return calls

    async def _execute_specialist(
        self, 
        delegate, 
        turn, 
        exp, 
        mcp_registry, 
        http_session, 
        tool_cache_str, 
        tool_cache_objs
    ):
        """Helper to run a single specialist execution (enables parallelism)."""
        role = delegate.role
        
        spec_model = self._pick_specialist(exp, role)
        
        if not spec_model or role not in mcp_registry:
            return ("system", f"[Error] Role {role} invalid.", 0, 0, [])
        
        session = mcp_registry[role]
        spec_llm = OllamaClient(model=spec_model[0])
        spec_prompt = self.cfg.get_role_spec(role).system_prompt
        specialist = SpecialistAgent(llm=spec_llm, system_prompt=spec_prompt)
        
        orch_instr = delegate.args.get("query") or delegate.args.get("_") or json.dumps(delegate.args)

        spec_res = await specialist.solve(
            session=session, orch_instr=orch_instr, http_session=http_session,
            cached_inventory=tool_cache_str.get(role), cached_tools=tool_cache_objs.get(role) 
        )
        
        # trace entry
        trace_entry = {
            "turn": turn + 1, 
            "agent": role, 
            "messages": spec_res.history, 
            "response": spec_res.response
        }
        
        return (role, spec_res.response, spec_res.total_prompt_tokens, spec_res.total_completion_tokens, trace_entry)

    async def run_once(
            self, *, eval_idx: int, eval_prompt: str, exp: ExperimentSpec,
            mcp_registry: Dict[str, ClientSession], http_session: aiohttp.ClientSession,
            max_turns: int = 20, tool_cache: Optional[Dict[str, Tuple[List[Any], str]]] = None 
        ) -> tuple[str, bool | None] | None:
        
        self.log = {"turns": 0, "specialist_calls": {}, "prompt_tokens": 0, "completion_tokens": 0, "trace": []}
        delegation_log = set()
        
        tool_cache_objs, tool_cache_str = {}, {}
        if tool_cache:
            for role, (t_objs, t_str) in tool_cache.items():
                tool_cache_objs[role] = t_objs
                tool_cache_str[role] = t_str
        else:
            pass

        orch_llm = OllamaClient(model=exp.orchestrator_model)
        orchestrator = OrchestratorAgent(llm=orch_llm, system_prompt=self.cfg.prompts.orchestrator)
        roles_inject = self._create_specialist_injection(exp)
        spec_return = []

        for turn in range(max_turns):
            self.log["turns"] = turn + 1
            if turn == 6:
                spec_return.append(("system", "SYSTEM HINT: You have taken 6 turns. Review the information above."))

            out, _, orch_resp, orch_history = await orchestrator.route(
                eval_prompt=eval_prompt, roles_inject=roles_inject,
                spec_return=spec_return, http_session=http_session 
            )
            
            self.log["prompt_tokens"] += orch_resp.prompt_tokens
            self.log["completion_tokens"] += orch_resp.completion_tokens
            self.log["trace"].append({"turn": turn + 1, "agent": "orchestrator", "messages": orch_history, "response": out})

            all_calls = self._parse_all_delegates(out)
            if not all_calls: return out, None

            spec_return.append(("orchestrator", out))

            # --- PARALLEL EXECUTION BLOCK ---
            tasks = []
            for delegate in all_calls:
                role = delegate.role
                current_call_sig = (role, json.dumps(delegate.args, sort_keys=True))
                
                if current_call_sig in delegation_log:
                    spec_return.append(("system", f"SYSTEM BLOCK: Duplicate assignment to {role}."))
                    continue
                delegation_log.add(current_call_sig)
                
                self.log["specialist_calls"][role] = self.log["specialist_calls"].get(role, 0) + 1
                
                tasks.append(
                    self._execute_specialist(
                        delegate, turn, exp, mcp_registry, http_session, tool_cache_str, tool_cache_objs
                    )
                )

            if tasks:
                results = await asyncio.gather(*tasks)
                
                for res_role, res_response, res_p_tok, res_c_tok, res_trace in results:
                    # Aggregate metrics
                    self.log["prompt_tokens"] += res_p_tok
                    self.log["completion_tokens"] += res_c_tok
                    self.log["trace"].append(res_trace)
                    
                    # Add to context for next turn
                    spec_return.append((res_role, res_response))
            # -------------------------------

        return "[FINAL] Reached max_turns without a FINAL.", True

    async def run_once_single_loop(
        self, *, eval_idx: int, eval_prompt: str, exp: ExperimentSpec,
        active_mcp_session: ClientSession, http_session: aiohttp.ClientSession, 
        max_turns: int = 20, cached_inventory: Optional[str] = None, cached_tools: Optional[List[Any]] = None  
    ) -> tuple[str, None]:

        self.log = {"turns": 0, "specialist_calls": {}, "prompt_tokens": 0, "completion_tokens": 0, "trace": []}
        policies_inject = self._create_policy_injection(exp)
        single_loop_llm = OllamaClient(model=exp.orchestrator_model)
        
        single_loop_agent = SingleLoopAgent(llm=single_loop_llm, system_prompt=self.cfg.prompts.orchestrator, max_steps=max_turns)
        
        result: AgentResult = await single_loop_agent.solve(
            session=active_mcp_session, prompt=eval_prompt, http_session=http_session,
            cached_inventory=cached_inventory, cached_tools=cached_tools,
            policies_inject=policies_inject 
        )

        self.log["turns"] = result.turns
        self.log["specialist_calls"] = result.tool_counts
        self.log["prompt_tokens"] = result.prompt_tokens
        self.log["completion_tokens"] = result.completion_tokens
        self.log["trace"].append({"turn": 1, "agent": "single-loop", "messages": result.history, "response": result.response})
        
        return result.response, None
