import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from lab.agent import Budget
from lab.eval import RunLog, load_cases, numeric, score, summarize


class EvalTests(unittest.TestCase):
    def test_only_complete_numeric_answers_are_scored(self):
        for answer in ("FINAL: 42", "HTTP 404", "42 then 17", "NaN", "inf", "1e999", "", "$42"):
            self.assertIsNone(numeric(answer))
            self.assertFalse(score("completed", answer, "42", 0.01, 1e-9)["correct"])
        self.assertEqual(numeric(" -1.25e2 \n"), -125)
        self.assertTrue(score("completed", "42", "42", 0.01, 1e-9)["correct"])
        self.assertFalse(score("model_error", "404", "404", 0.01, 1e-9)["correct"])
        self.assertFalse(score("budget_exhausted", "42", "42", 0.01, 1e-9)["correct"])

    def test_zero_and_tolerance(self):
        self.assertTrue(score("completed", "0", "0", 0.01, 1e-9)["correct"])
        self.assertFalse(score("completed", "0.1", "0", 0.01, 1e-9)["correct"])

    def test_legacy_suite_is_validated(self):
        self.assertEqual(
            len(load_cases(Path(__file__).resolve().parents[1] / "config/eval_set.json")), 40
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite.json"
            for prompts, truths in ((["x"], []), ([""], ["42"]), (["x"], ["nan"])):
                path.write_text(json.dumps({"eval-prompts": prompts, "ground_truth": truths}))
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_missing_results_stay_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            planned = [
                {"case_id": f"case-{i:03}", "mode": "single", "iteration": 1} for i in (1, 2)
            ]
            manifest = {"run_id": "test", "planned": planned}
            log = RunLog(directory, manifest)
            log.emit({"type": "preflight", "models": [{"native_tool_roundtrip": True}]})
            log.emit(
                {
                    "type": "case_result",
                    **planned[0],
                    "status": "completed",
                    "correct": True,
                    "latency_s": 1,
                    "metrics": asdict(Budget()),
                }
            )
            log.close()
            with (directory / "events.jsonl").open("a") as stream:
                stream.write('{"type":')
            summary = summarize(directory)["modes"]["single"]
            self.assertEqual(summary["protocol_condition"], {"manager": "unverified"})
            self.assertEqual(summary["accuracy"], 0.5)
            self.assertEqual(summary["statuses"]["missing"], 1)
            self.assertFalse(summary["usage_complete"])
            with self.assertRaises(FileExistsError):
                RunLog(directory, manifest)
