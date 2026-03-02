import asyncio
import time
import json
import os
import argparse
import aiohttp
from asyncio import Semaphore, Lock

from config.loader import ConfigLoader
from config.models import PipelineConfig
from filesystem.logger import ExperimentLogger
from llm.lifecycle import OllamaLifecycleManager
from mcpservers.factory import server_params_for_role
from mcpservers.multi_mcp_proxy import MultiServerSession
from tools.toolresolver import ToolInventory
from llm.agents.utils import extract_numeric_value
from llm.strategies import OrchestratorStrategy, SingleLoopStrategy

async def run_experiment(
    mode: str,
    cfg: PipelineConfig, 
    logger: ExperimentLogger, 
    http_session: aiohttp.ClientSession,
    debug_log_path: str
):
    sem = Semaphore(5)
    csv_lock = Lock()
    lifecycle = OllamaLifecycleManager()

    for exp in cfg.experiments:
        print(f"run: {exp.name}")

        # setup mcp
        roles = set(s.role for s in exp.specialists)
        params_by_role = {r: server_params_for_role(r) for r in roles}
        
        async with MultiServerSession(params_by_role, namespace_collisions=True) as mcp_session:
            registry = mcp_session._sessions 

            # cache tools
            tool_cache_str = {}
            tool_cache_objs = {}
            
            targets = list(registry.items())
            if mode == "single-loop":
                targets.append(("global", mcp_session))

            for tag, session in targets:
                tools_desc = await session.list_tools()
                t_list = tools_desc.tools or []
                cards = ToolInventory.from_mcp(t_list)
                tool_cache_objs[tag] = t_list
                tool_cache_str[tag] = ToolInventory.as_inventory_json(cards)

            # preload models
            await lifecycle.load_model(exp.orchestrator_model, http_session)
            
            if mode == "orchestrator":
                unique_specialists = set(s.model for s in exp.specialists if s.model)
                for m in unique_specialists:
                    if m != exp.orchestrator_model:
                        await lifecycle.load_model(m, http_session)

            # execute
            async def process_prompt(i, p, truth):
                async with sem:
                    runner = OrchestratorStrategy(cfg) if mode == "orchestrator" else SingleLoopStrategy(cfg)
                    run_registry = registry if mode == "orchestrator" else {"main": mcp_session}

                    start_time = time.perf_counter()
                    
                    answer_raw, logs = await runner.execute(
                        eval_idx=i, 
                        eval_prompt=p, 
                        exp=exp,
                        mcp_registry=run_registry,
                        http_session=http_session,
                        tool_cache=tool_cache_str,
                        tool_objects=tool_cache_objs
                    )

                    elapsed = time.perf_counter() - start_time
                    answer = extract_numeric_value(answer_raw)
                    if not answer: 
                        answer = str(answer_raw)

                    async with csv_lock:
                        print(f"prompt {i}: {elapsed:.2f}s | {answer}")
                        
                        logger.log_result(
                            cfg=cfg,
                            eval_idx=i,
                            eval_prompt=p,
                            experiment=exp,
                            message=answer,
                            elapsed_s=elapsed,
                            calls=logs["specialist_calls"],
                            turns=logs["turns"],
                            prompt_tokens=logs.get("prompt_tokens", 0),
                            completion_tokens=logs.get("completion_tokens", 0),
                            truth=truth,
                        )
                        
                        debug_entry = {
                            "experiment": exp.name,
                            "mode": mode,
                            "eval_idx": i,
                            "prompt": p,
                            "truth": truth,
                            "final_parsed_answer": answer,
                            "raw_answer": answer_raw,
                            "metrics": {"turns": logs["turns"], "latency": elapsed},
                            "conversation_trace": logs.get("trace", []) 
                        }
                        
                        with open(debug_log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(debug_entry, ensure_ascii=False) + "\n")

            tasks = [
                process_prompt(i, p, cfg.eval_truth[i] if i < len(cfg.eval_truth) else "")
                for i, p in enumerate(cfg.eval_prompts)
            ]
            
            await asyncio.gather(*tasks)

        # cleanup
        await lifecycle.unload_model(exp.orchestrator_model, http_session)
        
        if mode == "orchestrator":
            specialist_models = set(s.model for s in exp.specialists if s.model)
            for m in specialist_models:
                if m != exp.orchestrator_model:
                        await lifecycle.unload_model(m, http_session)

        await lifecycle.wait_for_unload(exp.orchestrator_model, http_session)

async def main():
    parser = argparse.ArgumentParser(description="Unified Agent Runner")
    parser.add_argument("config", help="pipeline config path")
    parser.add_argument("--mode", choices=["orchestrator", "single-loop"], required=True)
    
    args = parser.parse_args()

    loader = ConfigLoader(args.config)
    cfg = loader.load_config()
    
    logger = ExperimentLogger("logs/runs.csv")

    debug_log_path = "logs/detailed_conversations.jsonl"
    os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)

    async with aiohttp.ClientSession() as http_session:
        await run_experiment(args.mode, cfg, logger, http_session, debug_log_path)

if __name__ == "__main__":
    asyncio.run(main())
