"""Index observed attack exposure and hijacked steps in the local AgentDrift release."""
import argparse
import json
from pathlib import Path


def records(files):
    for path in files:
        with path.open(encoding="utf-8") as reader:
            for line in reader:
                trace = json.loads(line)
                steps = trace["steps"]
                labels = [step["label"] for step in steps]
                if all(label == "benign" for label in labels):
                    continue
                yield {"source": "AgentDrift", "split": path.stem, "trace_id": trace["id"],
                       "category": trace["category"],
                       "injection_steps": [i for i, label in enumerate(labels, 1) if label == "injection_point"],
                       "first_hijacked_step": next((i for i, label in enumerate(labels, 1)
                                                    if label == "hijacked"), None),
                       "failed_injection_steps": [i for i, label in enumerate(labels, 1)
                                                  if label == "failed_injection"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    for row in records([args.dataset_dir / f"{split}.jsonl" for split in ("train", "val", "test")]):
        print(json.dumps(row, ensure_ascii=False))
