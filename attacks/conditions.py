"""Build offline run metadata kept separate from an agent's visible context."""
import json
from pathlib import Path


def run_condition(pair_id, condition_id, *, source_case_id=None, target_id=None):
    spec = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))[condition_id]
    if not pair_id:
        raise ValueError("Missing paired run identifier")
    if condition_id == "clean":
        if source_case_id is not None or target_id is not None:
            raise ValueError("A clean run has no intervention target")
    elif not source_case_id or not target_id:
        raise ValueError("Supply both the benchmark case ID and the affected resource or agent ID")
    return dict(pair_id=pair_id, condition_id=condition_id, source_case_id=source_case_id,
                target_id=target_id, **spec)
