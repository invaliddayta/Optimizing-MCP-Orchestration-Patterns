# Unified Agent Runner: LLM Architecture Evaluation

This repository contains the codebase for evaluating and comparing different Large Language Model (LLM) agent architectures. Specifically, it tests a **Manager-Specialist (Orchestrator)** architecture against a traditional **Single-Loop** architecture when interacting with various Model Context Protocol (MCP) tool servers.

## What This Project Does

The system runs automated experiments to evaluate how different agent configurations perform on a set of tasks. It supports:
* **Orchestrator Mode:** A manager LLM delegates tasks to specialized sub-agents based on their roles and available tools.
* **Single-Loop Mode:** A single LLM attempts to solve the prompt directly using all available tools in one continuous loop.
* **Multi-MCP Proxy:** Dynamically loads and proxies multiple MCP servers (e.g., calculators, stock market simulators, unit converters) into the LLM context, handling tool namespace collisions automatically.
* **Local Execution:** Uses local LLMs via Ollama to ensure consistent, offline, and cost-free evaluation.

## Prerequisites

Before running the project, ensure you have the following installed:
1.  **Python 3.13+**
2.  **[uv](https://docs.astral.sh/uv/)**: An extremely fast Python package and project manager.
3.  **[Ollama](https://ollama.com/)**: Must be installed and running locally on your machine (default: `http://localhost:11434`).

## Installation

1. Clone this repository to your local machine.
2. Ensure Ollama is running in the background.
3. Because this project uses `uv`, dependencies are handled automatically when you run the scripts. You do not need to manually create a virtual environment or run `pip install`. The required dependencies (like `aiohttp`, `mcp`, and `pandas`) are defined in the `pyproject.toml`.

## How to Run

There are two primary ways to run the experiments: manually via the Python CLI or automatically using the provided bash script.

### Option 1: Automated Run (Recommended)
To run the full suite of experiments (10 iterations of both Orchestrated and Single-Loop configurations), simply execute the provided bash script:

```bash
chmod +x runall.sh
./runall.sh
```
This script will sequentially run the orchestrator and single-loop pipelines and automatically separate the logs.

### Option 2: Manual Run
You can run a specific configuration and mode manually using `uv run main.py`. 

**Run the Manager-Specialist Architecture:**
```bash
uv run main.py config/pipeline_orchestrated.json --mode orchestrator
```

**Run the Single-Loop Architecture:**
```bash
uv run main.py config/pipeline_single_loop.json --mode single-loop
```

## Configuration Files

The behavior of the agents and the evaluation datasets are controlled by JSON configuration files located in the `config/` directory:
* **`pipeline_orchestrated.json`**: Defines the prompts, experiments, and specialist pools for the manager-specialist architecture.
* **`pipeline_single_loop.json`**: Defines the prompts and experiments for the standard single-loop architecture.

## ⚙️ Highly Customizable: Models and Prompts

One of the core strengths of this evaluation framework is its flexibility. You do not need to modify any Python code to change the underlying LLMs or adjust their behavior; everything is driven by the JSON configuration files.

### Switching Models (Ollama)
Because the framework interfaces with Ollama, swapping out the models for the Orchestrator or the Specialists is as simple as changing a string in your JSON config files. 

Any valid Ollama model tag (e.g., `"llama3"`, `"mistral"`, `"qwen2.5:14b"`, `"phi3"`) can be used. The system will automatically detect the model string, load the model into memory via Ollama before the experiment starts, and unload it when finished.

**Example from `config/pipeline_orchestrated.json`:**
```json
"experiments": [
  {
    "name": "Llama3-Manager with Mistral-Specialists",
    "orchestrator_model": "llama3:8b", 
    "specialists": [
      {
        "role": "calculator_expert",
        "model": "mistral:7b",
        "count": 1
      }
    ]
  }
]
```
*Want to test a heavier model?* Just change `"llama3:8b"` to `"llama3:70b"`, ensure you have pulled it via `ollama pull llama3:70b`, and run the script. The framework handles the rest.

### Adjusting System and Expert Prompts
Prompt engineering is critical for agent evaluation. The framework fully decouples prompts from the codebase. Both the global orchestrator prompts and the individual specialist prompts are defined in the JSON configuration (often referencing a `roles_config.json` or embedded directly in the pipeline config).

You can easily iterate on prompt designs by modifying the following fields:
* **`system_prompt`**: The core instructions given to the agent defining its persona, output format, and rules of engagement.
* **`policy_injection`**: Specific behavioral constraints or tool-usage policies you want to inject into the agent's context dynamically.

**Example Role Configuration:**
```json
"roles": {
  "calculator_expert": {
    "description": "An agent specialized in mathematical operations.",
    "system_prompt": "You are a precise mathematical assistant. You MUST use the calculator tool for any math operations. Do not attempt to calculate in your head. Return ONLY the final numerical answer.",
    "policy_injection": "Always verify your calculations twice."
  }
}
```
By keeping models, system prompts, and expert instructions entirely in JSON, you can rapidly duplicate configuration files to run A/B tests across different prompt strategies and model combinations!

## Outputs & Logging

All experiment results are tracked and saved in the `logs/` directory:
* **`logs/runs.csv`**: Contains high-level metrics for each run, including latency, token usage, turn counts, and the final extracted answers compared against the ground truth.
* **`logs/detailed_conversations.jsonl`**: Contains deep debug traces of the LLM interactions, including raw outputs, tool call parsing, and full conversation histories.

### 📊 Analyzing Results
After your experiments have finished, you can generate a high-level performance summary (including strict vs. semantic accuracy, average latency, and token usage) by running the analysis script:

```bash
uv run analyze_results.py
```

This will:
* **Print a Results Table**: Displays a comparison of all experiments (Orchestrator vs. Single-Loop) directly in your terminal.
* **Generate `logs/analysis_summary.csv`**: Saves the aggregated statistics for further reporting.
* **Generate `logs/failures_semantic.csv`**: Automatically extracts and saves every instance where the model failed to provide the correct answer, allowing for easy error analysis.
