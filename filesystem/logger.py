import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Union, Any, Sequence

from config.models import PipelineConfig, ExperimentSpec

class ExperimentLogger:
    def __init__(self, csv_out_path: Union[str, Path]):
        self.csv_out = Path(csv_out_path)
        self._ensure_header()

    def _ensure_header(self) -> None:
        self.csv_out.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_out.exists() or self.csv_out.stat().st_size == 0:
            with self.csv_out.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([
                    "timestamp_iso", "eval_idx", "eval_prompt", "experiment",
                    "orchestrator_model", "specialist_pool_json", "special_calls",
                    "turns", "prompt_tokens", "completion_tokens", "message_last",
                    "truth", "time_in_seconds",
                ])

    @staticmethod
    def _last_item(msg: Union[str, bytes, Sequence[Any], Any]) -> Any:
        if isinstance(msg, (list, tuple)):
            return msg[-1] if msg else ""
        return msg

    @staticmethod
    def _to_text(val: Any) -> str:
        if val is None: return ""
        if isinstance(val, bytes):
            try: return val.decode("utf-8", errors="replace")
            except Exception: return str(val)
        if isinstance(val, str):
            return val
        try:
            if hasattr(val, "__dataclass_fields__"):
                val = asdict(val)
            return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(val)

    def log_result(
        self, *, 
        cfg: PipelineConfig, 
        eval_idx: int, 
        eval_prompt: str, 
        experiment: ExperimentSpec,
        message: Any, 
        elapsed_s: float, 
        calls: Any, 
        turns: int,
        prompt_tokens: int = 0, 
        completion_tokens: int = 0, 
        truth: Any,
    ) -> None:
        
        last_text = self._to_text(self._last_item(message))
        if len(last_text) >= 4000: last_text = last_text[:4000] + " …[truncated]"
        
        with self.csv_out.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                eval_idx, 
                self._to_text(eval_prompt), 
                self._to_text(experiment.name),
                self._to_text(experiment.orchestrator_model),
                self._to_text(cfg.expand_specialist_pool(experiment)),
                self._to_text(calls), 
                self._to_text(turns),
                prompt_tokens, 
                completion_tokens, 
                last_text, 
                truth, 
                float(elapsed_s)
            ])
