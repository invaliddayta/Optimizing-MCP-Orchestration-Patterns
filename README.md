# MCP Orchestration Lab

A small local benchmark for one question: **when does delegating to tool-scoped
workers outperform one model with all the tools?**

One agent loop, two configurations. Native tool calls through llama.cpp's
`llama-server`, actual MCP subprocesses, and inspectable results. No Ollama,
agent framework, model scheduler, or Python environment bootstrap.

## Nix Development

Nix with flakes enabled is the only prerequisite. `flake.lock` pins Python,
aiohttp, the MCP SDK, JSON Schema validation, Ruff, and llama.cpp through nixpkgs.
There is no `uv`, `pip install`, or virtualenv step.

```sh
nix develop
python -m lab --help
python -m unittest discover -s tests -v
```

The flake defines Linux environments for `aarch64-linux` and `x86_64-linux`.
The default llama.cpp build is CPU-based; GPU acceleration is intentionally not
configured in this first version. GGUF model weights are supplied separately,
not downloaded during builds or checks.

For a clean shell that does not inherit your usual environment:

```sh
nix develop --ignore-environment
```

If working with newly created, still-untracked flake/source files, use `path:.`
instead of the default Git source, for example `nix develop path:.` and
`nix flake check path:.`. No staging is needed for that workflow.

## First Run

In the development shell, start a tool-capable GGUF model:

```sh
llama-server -m /absolute/path/to/model.gguf --alias local \
  --host 127.0.0.1 --port 8080 --jinja -c 8192
```

In a second development shell:

```sh
python -m lab doctor --model local
python -m lab run --model local --limit 3 --output results/smoke
python -m lab report results/smoke
python -m lab trace results/smoke --case case-001 --mode delegated
```

`doctor` checks a native tool-call/result round trip and starts/discovers all MCP
tools. Model and chat-template compatibility matters: a model emitting `CALL:`
text does not pass. `run` performs the same preflight before executing cases.
Preflight calls and server startup are excluded from task metrics.

`--output` must be a new directory. Omitting it creates a unique directory under
`results/`. The default is both modes, one repetition, sequential execution.

```sh
python -m lab run --model local --mode single --case 1
python -m lab run --model local --mode both --repeat 3 --limit 10
```

The packaged entry point also works without entering a development shell:

```sh
nix run . -- run --model local --limit 3
```

It includes the code and fixtures in the Nix store and can run from any directory.
Use `python -m lab` from the repository when developing against live source files.

## The Comparison

- **Single:** one model sees every MCP tool.
- **Delegated:** the manager sees only `delegate(role, task)`. Each invocation runs
  a stateless worker with that role's MCP tools using the same loop. Workers cannot
  delegate recursively; the manager must pass the values they need explicitly.

By default both modes and all workers use the same model and endpoint, isolating
the orchestration difference. To test smaller workers, start a second server
separately and pass `--worker-model small --worker-base-url http://127.0.0.1:8081/v1`.
The harness does not load, unload, or swap models. Account for memory residency and
server configuration when interpreting comparisons.

`--max-calls 20` is shared across manager and workers, not 20 calls per worker.
`--worker-calls 4` additionally bounds an individual delegation. Other limits are
`--max-tool-calls 40`, `--max-tokens 1024` per response, `--request-timeout 120`,
`--tool-timeout 30`, and `--case-timeout 300` (timeouts are seconds).
Tool calls run sequentially, even if a model emits several at once. The total
tool-call metric includes delegation attempts; traces distinguish delegation
from actual MCP calls. Invalid arguments and tool errors are returned to the
model for recovery and counted, never silently repaired or fuzzy-matched.

Temperature is zero and the seed defaults to zero (`--seed` overrides it).
This does not guarantee bit-identical inference. Across repetitions, mode order
alternates; measurements still depend on caches, model residency, and hardware.

## Results

Each run contains two files:

- `manifest.json`: the planned case/mode/repetition matrix, paired prompts and
  answers, settings, Python/dependency versions, code and configuration hashes,
  and optional model metadata.
- `events.jsonl`: flushed, append-only model requests/responses, tool results,
  worker outcomes, case results, and suite failures. This is the results source
  of truth; `report` derives its summary from these records.

Raw answers and parsed values are separate. Only a complete, finite numeric
answer is scored. Labels, prose, `FINAL:`, truncated responses, and HTTP error
messages are not salvaged into numbers. The legacy suite uses 1% relative and
1e-9 absolute tolerance by default, configurable with `--relative-tolerance` and
`--absolute-tolerance`.

Execution statuses include `completed`, `invalid_output`, `model_error`,
`timeout`, `budget_exhausted`, and `harness_error`. A completed answer may still be
wrong. Recoverable tool errors remain visible in events and `tool_errors`.
Missing cases count against reported accuracy, including interrupted or failed
preflight runs. A partially written final event line is ignored on recovery.
Token usage is marked incomplete if an endpoint does not report it or a request
fails. Latency is end-to-end per finished case, not isolated inference time.

Exit code 1 means setup or execution failed for at least one case; a wrong but
well-formed answer does not itself fail the command. Ctrl-C exits 130 and preserves
already written records. Reports never append to or overwrite historical logs.

Model aliases do not identify weights or templates. For reproducible experiments,
pass `--model-metadata metadata.json` with your GGUF SHA-256, quantization, template,
llama.cpp version, launch flags, and hardware. This JSON is preserved as
user-supplied metadata, not independently verified. Traces contain prompts and tool
results; keep sensitive data out of public run artifacts.

## Development And Scope

```sh
nix flake check
nix develop --command ruff check lab tests mcpservers/calculator_mcp.py
nix develop --command ruff format --check lab tests mcpservers/calculator_mcp.py
```

Checks need neither a GPU nor model weights. Tests include a scripted HTTP model
with real MCP subprocesses for both modes, native transcript validation, shared
budgets, scoring failures, and subprocess startup/cancellation cleanup.

The implementation is in `lab/{model,agent,mcp,eval,__main__}.py`. MCP server
commands are in `config/servers_config.json`; only configure trusted executables.
Default subprocess cwd is the config's parent directory's parent. An explicit
server `cwd` is resolved relative to the config directory. Tools are namespaced
as `server__tool`; ambiguous or invalid names fail instead of guessing.

The 40 deterministic lookup/arithmetic cases in `config/eval_set.json` are a
regression baseline, not evidence about general agent capability. The local
calculator accepts only finite arithmetic with `+ - * /`, parentheses, and unary
signs; it does not execute Python expressions beyond that subset.

Original thesis configurations are in `legacy/`, original results in `logs/`,
and the removed Ollama implementation in Git history. Old and new scores are not
directly comparable. Larger tool catalogs, realistic task families, and GPU/model
scheduling are intentionally outside this first working slice.
