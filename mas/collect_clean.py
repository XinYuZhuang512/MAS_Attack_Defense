"""Resumable collection of normal benchmark trajectories."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
import json
from pathlib import Path
import random
import re
import threading
import time
from uuid import uuid4

from mas.deepseek import DeepSeek, MAX_TOKENS, MODEL, REASONING_EFFORT
from mas.experiment import punctuation_spans, write_jsonl
from mas.replay import annotation_jobs
from mas.trace import Trace

DATA = Path(__file__).resolve().parents[1] / "datasets" / "clean_collection"


class RecordedRun:
    def __init__(self, trace, task, environment):
        self.trace = trace
        self.task = task
        self.environment = environment
        self.client = DeepSeek()
        self.turns = Counter()
        self.query_id = trace.message("user", task["query"])

    def respond(self, agent, prompt, messages, triggers, *, use_tools=False):
        inputs = [{"role": "system", "content": prompt}]
        for mid, sender, text in messages:
            self.trace.deliver(mid, agent)
            inputs.append({"role": "user", "message_id": mid,
                           "content": text if sender == "user" else f"Message from {sender}:\n{text}"})
        return self.respond_payload(agent, inputs, triggers, use_tools=use_tools)

    def respond_payload(self, agent, inputs, triggers, *, use_tools=False):
        tools = self.environment.tools if use_tools and self.environment else []
        for _ in range(16):
            output_id, assistant, calls = self.complete(agent, inputs, triggers, tools)
            if not calls:
                return output_id, assistant["content"]
            inputs.append({"message_id": output_id, **assistant})
            triggers = []
            for call in calls:
                name = call["function"]["name"]
                result, arguments, failed = self.environment.invoke(name, call["function"]["arguments"])
                reply_id = self.trace.message("tool", result)
                self.trace.event("tool_executed", agent_id=agent, name=name, arguments=arguments,
                                 request_message_id=output_id, result_message_id=reply_id,
                                 tool_call_id=call["id"], failed=failed)
                self.trace.deliver(reply_id, agent)
                inputs.append({"message_id": reply_id, "role": "tool", "content": result,
                               "tool_call_id": call["id"]})
                triggers.append(reply_id)
        raise ValueError(f"{agent} exhausted the tool-turn limit")

    def complete(self, agent, inputs, triggers, tools):
        self.turns[agent] += 1
        call_id = self.trace.start_call(agent, self.turns[agent], inputs, triggers)
        self.trace.event("model_request_configured", call_id=call_id, tools=tools)
        reasoning, answer, calls = [], [], {}
        try:
            for channel, value in self.client.stream(inputs, tools=tools):
                if channel == "reasoning_content":
                    reasoning.append(value)
                    self.trace.reasoning(call_id, value)
                elif channel == "content":
                    answer.append(value)
                    self.trace.event("output_chunk", call_id=call_id, text=value)
                else:
                    self.trace.event("tool_call_chunk", call_id=call_id, delta=value)
                    for delta in value:
                        request = calls.setdefault(delta["index"], {
                            "id": "", "type": "function", "function": {"name": "", "arguments": ""},
                        })
                        request["id"] = delta.get("id") or request["id"]
                        for name, piece in delta.get("function", {}).items():
                            request["function"][name] += piece
            if any(not item["id"] for item in calls.values()):
                raise ValueError("Incomplete model tool-call response")
            text = "".join(answer)
            if not calls and not text.strip():
                raise ValueError("Empty model response")
        except Exception:
            self.trace.finish_call(call_id, [], "observable" if reasoning else "unavailable", status="failed")
            raise
        assistant = {"role": "assistant", "content": text or None}
        actions = [calls[index] for index in sorted(calls)]
        if actions:
            assistant.update(tool_calls=actions, reasoning_content="".join(reasoning))
        output_id = self.trace.message("assistant", assistant if actions else text, call_id)
        self.trace.finish_call(call_id, [output_id], "observable" if reasoning else "unavailable")
        return output_id, assistant, actions


def instructions(dataset, role):
    answer = {"mmlu": "End with a line containing only the selected option letter A, B, C, or D.",
              "humaneval": "Return the complete Python function requested by the task.",
              "pear": "Use the supplied task environment when available. Report verified results."}[dataset]
    return f"{role} Solve only the original user task. {answer}"


def evaluate(task, answer, environment):
    if environment is not None:
        try:
            return {"passed": environment.evaluate(answer)}
        except KeyError as exc:
            # Some official utility functions index the expected destination
            # directly. An absent record is a normal task failure.
            return {"passed": False, "evaluation": "missing_expected_simulator_record",
                    "missing_key": str(exc)}
    if task["dataset"] == "mmlu":
        final = answer.strip().splitlines()[-1].strip()
        parsed = re.fullmatch(r"(?:answer\s*[:=]\s*)?\**([A-D])\**[.)]?", final, re.IGNORECASE)
        return {"parsed_answer": parsed[1].upper() if parsed else None,
                "passed": bool(parsed and parsed[1].upper() == task["answer"])}
    return {"passed": None, "evaluation": "functional_evaluation_pending"}


def run_one(task, system, invoke, destination):
    environment = None
    if task["dataset"] == "pear":
        from mas.pear_native import PearEnvironment

        environment = PearEnvironment(task)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "trace.jsonl"
    with Trace(path, uuid4().hex, task["id"]) as trace:
        trace.event("experiment_configured", condition_id="clean", dataset=task["dataset"],
                    system=system, model=MODEL, reasoning_effort=REASONING_EFFORT,
                    thinking="enabled", max_tokens=MAX_TOKENS)
        run = RecordedRun(trace, task, environment)
        output_id, answer = invoke(run)
        trace.deliver(output_id, "user")
        score = evaluate(task, answer, environment)
        trace.event("task_evaluated", answer_message_id=output_id, **score)
    write_jsonl(destination / "annotation_inputs.jsonl", annotation_jobs(path, punctuation_spans))
    return score


def collect(system, names, workers, output, invocation, limit, task_ids=()):
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for name in names:
        with (DATA / f"{name}.jsonl").open(encoding="utf-8") as handle:
            tasks.extend(map(json.loads, handle))
    if task_ids:
        requested = set(task_ids)
        tasks = [task for task in tasks if task["id"] in requested]
        if len(tasks) != len(requested):
            raise ValueError("One or more requested task IDs are absent from the selected datasets")
    if "pear" in names:
        from mas.pear_native import DOMAINS, load_initial_state, tools_for

        for domain in DOMAINS:
            load_initial_state(domain)
            tools_for(domain)
    random.Random(20261125).shuffle(tasks)
    if limit is not None:
        tasks = tasks[:limit]
    lock = threading.Lock()
    counts = Counter()
    start = time.monotonic()

    def status(event, **fields):
        with lock, (output / f"{system}.status.jsonl").open("a", encoding="utf-8") as log:
            row = dict(time=time.time(), event=event, system=system, total=len(tasks), **fields)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")

    def execute(task):
        key = task["id"].replace("/", "__")
        target = output / "runs" / system / task["dataset"] / key
        if (target / "result.json").exists():
            return "existing"
        for retry in range(3):
            attempt = output / "attempts" / system / task["dataset"] / f"{key}-{uuid4().hex[:8]}"
            try:
                score = run_one(task, system, invocation, attempt)
                (attempt / "result.json").write_text(json.dumps(score) + "\n", encoding="utf-8")
                target.parent.mkdir(parents=True, exist_ok=True)
                attempt.rename(target)
                status("saved", task=task["id"], passed=score["passed"])
                return "saved"
            except Exception as exc:
                status("attempt_failed", task=task["id"], attempt=retry+1, error=str(exc)[:400])
                if retry < 2:
                    time.sleep((2 ** retry) * 4)
        return "failed"

    stop = threading.Event()

    def monitor():
        while not stop.wait(1800):
            status("checkpoint", elapsed_seconds=round(time.monotonic() - start), **dict(counts))

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    status("started", workers=workers, datasets=names)
    try:
        with ThreadPoolExecutor(workers) as pool:
            for future in as_completed(pool.submit(execute, task) for task in tasks):
                counts[future.result()] += 1
    finally:
        stop.set()
        watcher.join()
    status("finished", elapsed_seconds=round(time.monotonic() - start), **dict(counts))
    print(system, dict(counts), flush=True)
    return int(bool(counts["failed"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=("autogen", "camel", "metagpt"))
    parser.add_argument("--datasets", nargs="+", choices=("pear", "humaneval", "mmlu"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append", default=[])
    args = parser.parse_args()
    invoke = import_module(f"mas.collect_{args.system}").invoke
    raise SystemExit(collect(args.system, list(dict.fromkeys(args.datasets)), args.workers,
                             args.output, invoke, args.limit, args.task_id))


if __name__ == "__main__":
    main()
