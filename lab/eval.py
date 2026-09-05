"""Strict numeric scoring and self-contained, append-only experiment records."""

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    prompts, truths = data["eval-prompts"], data["ground_truth"]
    if not isinstance(prompts, list) or not isinstance(truths, list) or len(prompts) != len(truths):
        raise ValueError("Evaluation prompts and truths must be equally sized arrays")
    if not prompts:
        raise ValueError("Evaluation suite is empty")
    cases = []
    for index, (prompt, truth) in enumerate(zip(prompts, truths), 1):
        if not isinstance(prompt, str) or not prompt.strip() or numeric(str(truth)) is None:
            raise ValueError(f"Invalid prompt or numeric truth at case {index}")
        cases.append({"id": f"case-{index:03}", "prompt": prompt, "expected": str(truth)})
    return cases


def numeric(text: str) -> float | None:
    stripped = text.strip()
    if not NUMBER.fullmatch(stripped):
        return None
    value = float(stripped)
    return value if math.isfinite(value) else None


def score(status: str, answer: str, expected: str, rel_tol: float, abs_tol: float) -> dict:
    parsed = numeric(answer) if status == "completed" else None
    if status == "completed" and parsed is None:
        status = "invalid_output"
    return {
        "status": status,
        "raw_answer": answer,
        "parsed_answer": parsed,
        "correct": parsed is not None
        and math.isclose(parsed, float(expected), rel_tol=rel_tol, abs_tol=abs_tol),
    }


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunLog:
    def __init__(self, directory: Path, manifest: dict):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self._stream = (directory / "events.jsonl").open("x")

    def emit(self, event: dict):
        self._stream.write(json.dumps(event, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self):
        self._stream.close()


def read_events(directory: Path):
    with (directory / "events.jsonl").open() as stream:
        for line in stream:
            # An interrupted write may leave one incomplete final line.
            if not line.endswith("\n"):
                break
            yield json.loads(line)


def summarize(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    planned = {(p["case_id"], p["mode"], p["iteration"]) for p in manifest["planned"]}
    records = {}
    suite_status = "incomplete"
    for event in read_events(directory):
        if event["type"] == "suite_finished":
            suite_status = "finished"
        elif event["type"] == "suite_error":
            suite_status = "failed"
        if event["type"] == "case_result":
            key = (event["case_id"], event["mode"], event["iteration"])
            if key not in planned or key in records:
                raise ValueError(f"Unexpected or duplicate result: {key}")
            records[key] = event
    modes = {}
    for mode in sorted({key[1] for key in planned}):
        total = sum(key[1] == mode for key in planned)
        rows = [row for key, row in records.items() if key[1] == mode]
        statuses = Counter(row["status"] for row in rows)
        statuses["missing"] = total - len(rows)
        modes[mode] = {
            "planned": total,
            "finished": len(rows),
            "statuses": dict(statuses),
            "accuracy": sum(row["correct"] for row in rows) / total,
            "median_latency_s": statistics.median(row["latency_s"] for row in rows)
            if rows
            else None,
            "model_calls": sum(row["metrics"]["calls"] for row in rows),
            "tool_calls_including_delegation": sum(row["metrics"]["tool_calls"] for row in rows),
            "tool_errors": sum(row["metrics"]["tool_errors"] for row in rows),
            "reported_prompt_tokens": sum(row["metrics"]["prompt_tokens"] for row in rows),
            "reported_completion_tokens": sum(row["metrics"]["completion_tokens"] for row in rows),
            "usage_complete": len(rows) == total
            and all(row["metrics"]["usage_complete"] for row in rows),
        }
    return {"run_id": manifest["run_id"], "suite_status": suite_status, "modes": modes}
