"""Validate and archive already recorded MAS traces for offline comparison."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from uuid import UUID

from mas.replay import read_trace


COLLECTION = Path(__file__).resolve().parents[1] / "datasets" / "injection_collection"


def load_pairs(path):
    with path.open(encoding="utf-8") as source:
        rows = list(map(json.loads, source))
    pairs = {(row["system"], row["dataset"], row["task_id"]): row for row in rows}
    if len(pairs) != len(rows):
        raise ValueError("Pair manifest contains duplicate task identities")
    return pairs


def match_trace(path, pairs):
    with path.open(encoding="utf-8") as f:
        start = json.loads(next(f))
        config = json.loads(next(f))
    if start["event"] != "run_started" or config["event"] != "experiment_configured":
        raise ValueError("Expected a run header and experiment configuration")
    settings = config["data"]
    condition = settings.get("condition_id")
    if condition == "tool_result":
        condition = "indirect_prompt_injection"
    if condition != "indirect_prompt_injection":
        raise ValueError("Trace metadata does not declare the requested condition")

    identity = (settings["system"], settings["dataset"], start["data"]["task_id"])
    pair = pairs.get(identity)
    if pair is None:
        raise ValueError("Task and MAS are absent from the target cohort")
    for key, value in pair["model_config"].items():
        if settings.get(key) != value:
            raise ValueError(f"Model setting differs from its normal run: {key}")
    if "dataset_revision" in settings and settings["dataset_revision"] != pair["dataset_revision"]:
        raise ValueError("Dataset revision differs from its normal run")

    run_id, messages, calls = read_trace(path)
    if run_id == pair["normal_run_id"]:
        raise ValueError("Trace reuses the normal run ID")
    if not calls:
        raise ValueError("Trace contains no model calls")
    roots = [m["content"] for m in messages.values()
             if m["role"] == "user" and m["producer_call_id"] is None]
    if (len(roots) != 1 or not isinstance(roots[0], str)
            or sha256(roots[0].encode("utf-8")).hexdigest() != pair["query_sha256"]):
        raise ValueError("Original user query differs from its normal run")
    return run_id, pair


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def archive_trace(source, pairs, output):
    with NamedTemporaryFile(dir=output, prefix=".incoming-") as snapshot:
        with source.open("rb") as raw:
            copyfileobj(raw, snapshot)
        snapshot.flush()
        run_id, pair = match_trace(Path(snapshot.name), pairs)
        # Use the validated baseline's identifier, never a path supplied by the recording.
        normal_id = UUID(pair["normal_run_id"]).hex
        destination = output / f"{normal_id}.jsonl"
        try:
            os.link(snapshot.name, destination)
            status = "archived"
        except FileExistsError:
            if digest(Path(snapshot.name)) != digest(destination):
                raise ValueError("A different recording is already stored for this pair") from None
            status = "existing"
    return {"source": str(source), "archive": str(destination), "status": status,
            "pair_id": pair["pair_id"], "run_id": run_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="A saved JSONL trace or a directory of trace.jsonl files")
    parser.add_argument("--manifest", type=Path, default=COLLECTION / "pair_manifest.jsonl")
    parser.add_argument("--output", type=Path, default=COLLECTION / "raw_traces")
    args = parser.parse_args()
    files = sorted(args.source.rglob("trace.jsonl")) if args.source.is_dir() else [args.source]
    if not files:
        parser.error("No trace.jsonl files found")
    pairs = load_pairs(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    rejected = False
    for source in files:
        try:
            result = archive_trace(source, pairs, args.output)
        except (OSError, StopIteration, KeyError, ValueError, TypeError) as exc:
            rejected = True
            result = {"source": str(source), "status": "rejected",
                      "reason": str(exc) or type(exc).__name__}
        print(json.dumps(result, ensure_ascii=False))
    return int(rejected)


if __name__ == "__main__":
    raise SystemExit(main())
