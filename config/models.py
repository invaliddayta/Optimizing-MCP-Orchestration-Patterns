from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Tuple, Any

@dataclass(frozen=True)
class RoleSpec:
    description: str
    policy_injection: str
    system_prompt: str

@dataclass(frozen=True)
class SpecialistSpec:
    role: str
    model: Optional[str] = None
    count: int = 1

@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    orchestrator_model: str
    specialists: List[SpecialistSpec] = field(default_factory=list)

@dataclass(frozen=True)
class GlobalPromptsSpec:
    orchestrator: str
    roles: Dict[str, RoleSpec]

@dataclass(frozen=True)
class PipelineConfig:
    version: str
    eval_prompts: List[str]
    eval_truth: List[str]
    prompts: GlobalPromptsSpec
    experiments: List[ExperimentSpec]

    def iter_runs(self) -> Iterator[Tuple[int, str, ExperimentSpec, str]]:
        for i, p in enumerate(self.eval_prompts):
            for exp in self.experiments:
                yield i, p, exp, self.eval_truth[i]

    def expand_specialist_pool(self, exp: ExperimentSpec) -> List[Dict[str, Any]]:
        pool: List[Dict[str, Any]] = []
        for s in exp.specialists:
            for k in range(1, max(1, s.count) + 1):
                pool.append({
                    "role": s.role, 
                    "model": s.model,
                    "instance": k
                })
        return pool

    def get_role_spec(self, role: str) -> RoleSpec:
        """Safe retrieval of role configuration."""
        return self.prompts.roles.get(role, RoleSpec(
            description="Specialized agent.", 
            policy_injection="", 
            system_prompt=""
        ))
