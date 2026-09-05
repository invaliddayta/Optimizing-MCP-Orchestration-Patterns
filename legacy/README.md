# Thesis Artifacts

These are the original Ollama experiment configurations, not inputs to the new CLI.
The original evaluation cases remain in `config/eval_set.json`; recorded results
remain unchanged in `logs/`. The previous runner and analyzer are in Git history.

Do not combine historical results with new runs. The new harness uses native
tool calls, complete assistant/tool transcripts, shared call budgets, a local
calculator fixture, sequential cases, and strict scoring of the raw final answer.
These changes make it a different experimental condition, not a reproduction of
the thesis numbers. In particular, the old analyzer scored pre-extracted answers
and did not reliably distinguish infrastructure errors from model output.
