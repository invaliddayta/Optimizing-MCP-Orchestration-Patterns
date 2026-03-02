import json
from pathlib import Path
from typing import Union, List

from config.models import (
    PipelineConfig, ExperimentSpec, SpecialistSpec, 
    GlobalPromptsSpec, RoleSpec
)

class ConfigLoader:
    def __init__(self, json_config_path: Union[str, Path]):
        self.json_config = Path(json_config_path)

    def load_config(self) -> PipelineConfig:
        with self.json_config.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # load roles
        roles_path = Path("config/roles_config.json")
        if not roles_path.exists():
            roles_path = self.json_config.parent / "roles_config.json"
        
        with roles_path.open("r", encoding="utf-8") as rf:
            roles_data = json.load(rf)

        parsed_roles = {}
        for role, info in roles_data.get("roles", {}).items():
            parsed_roles[role] = RoleSpec(
                description=info.get("description", ""),
                policy_injection=info.get("policy_injection", ""),
                system_prompt=info.get("system_prompt", "")
            )

        # load eval set
        eval_path = Path("config/eval_set.json")
        if not eval_path.exists():
             eval_path = self.json_config.parent / "eval_set.json"

        with eval_path.open("r", encoding="utf-8") as ef:
            eval_data = json.load(ef)

        eval_prompts = [x.strip() for x in eval_data.get("eval-prompts", []) if x.strip()]
        ground_truth = [x.strip() for x in eval_data.get("ground_truth", []) if x.strip()]

        # build config
        pipeline_prompts = data.get("prompts", {})
        orch_prompt = pipeline_prompts.get("orchestrator", roles_data.get("orchestrator_system", ""))

        prompts_spec = GlobalPromptsSpec(
            orchestrator=orch_prompt,
            roles=parsed_roles
        )

        experiments: List[ExperimentSpec] = []
        for exp in data.get("experiments", []):
            specs: List[SpecialistSpec] = []
            for s in exp.get("specialists", []):
                specs.append(SpecialistSpec(
                    role=str(s["role"]).strip(),
                    model=str(s.get("model")).strip() if s.get("model") else None,
                    count=int(s.get("count", 1))
                ))

            experiments.append(ExperimentSpec(
                name=str(exp["name"]).strip(),
                orchestrator_model=str(exp["orchestrator"]["model"]).strip(),
                specialists=specs
            ))

        return PipelineConfig(
            version=str(data.get("version", "0.0")),
            eval_prompts=eval_prompts,
            eval_truth=ground_truth,
            prompts=prompts_spec,
            experiments=experiments
        )
