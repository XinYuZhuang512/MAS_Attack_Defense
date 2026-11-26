"""Read a MAS event log and build independent annotation prefixes for each call."""
from bisect import bisect_left
import json


def read_trace(path):
    messages = {}
    delivered = set()
    calls = {}
    agents_by_call = {}
    run_id = None
    run_status = None
    with path.open(encoding="utf-8") as source:
        for sequence, line in enumerate(source):
            event = json.loads(line)
            kind, data = event["event"], event["data"]
            if sequence == 0:
                if kind != "run_started":
                    raise ValueError("The first event must be run_started")
                run_id = event["run_id"]
            if event["run_id"] != run_id or event["sequence"] != sequence or run_status:
                raise ValueError(f"Invalid run or event sequence at line {sequence + 1}")
            try:
                if kind == "message_created":
                    message_id = data["message_id"]
                    parent = data["producer_call_id"]
                    if message_id in messages or parent is not None and parent not in calls:
                        raise ValueError("Duplicate message or unknown producer")
                    messages[message_id] = data
                elif kind == "message_delivered":
                    message_id = data["message_id"]
                    if message_id not in messages:
                        raise ValueError("Delivery references an unknown message")
                    delivered.add((message_id, data["recipient_agent_id"]))
                elif kind == "call_started":
                    call_id = data["call_id"]
                    if call_id in calls:
                        raise ValueError("Duplicate call ID")
                    inputs = []
                    for item in data["input_messages"]:
                        # An input snapshot can carry an inline system prompt or a message ID.
                        if item.get("message_id") is not None:
                            message_id = item["message_id"]
                            message = messages[message_id]
                            producer = agents_by_call.get(message["producer_call_id"])
                            if producer != data["agent_id"] and (message_id, data["agent_id"]) not in delivered:
                                raise ValueError("An input message must have reached the recipient before the call")
                            item = {"role": message["role"], "content": message["content"], **item}
                        if "role" not in item or "content" not in item:
                            raise ValueError("Each input needs a role and content")
                        inputs.append(item)
                    for message_id in data["trigger_message_ids"]:
                        if (message_id, data["agent_id"]) not in delivered:
                            raise ValueError("A trigger message must have reached the recipient before the call")
                    agents_by_call[call_id] = data["agent_id"]
                    calls[call_id] = {**data, "input_messages": inputs, "chunks": [], "finished": False}
                elif kind in {"reasoning_chunk", "call_finished"}:
                    call = calls[data["call_id"]]
                    if call["finished"]:
                        raise ValueError("Call event appears after the call has finished")
                    if kind == "reasoning_chunk":
                        call["chunks"].append(event)
                    else:
                        for message_id in data["output_message_ids"]:
                            if messages[message_id]["producer_call_id"] != data["call_id"]:
                                raise ValueError("An output must reference a message produced by its call")
                        call["finished"] = True
                elif kind == "run_finished":
                    run_status = data["status"]
                    if run_status not in {"completed", "failed"}:
                        raise ValueError("Unknown run status")
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid trace at line {sequence + 1}: {exc}") from exc
    if run_status is None:
        raise ValueError("Trace is empty or incomplete")
    if run_status == "completed" and any(not call["finished"] for call in calls.values()):
        raise ValueError("A completed run contains an unfinished call")
    return run_id, messages, calls


def annotation_jobs(path, sentence_spans, *, query_message_id=None):
    run_id, messages, calls = read_trace(path)
    if query_message_id is None:
        roots = [m["message_id"] for m in messages.values()
                 if m["role"] == "user" and m["producer_call_id"] is None]
        if len(roots) != 1:
            raise ValueError("Expected one original user message; specify --query-message-id")
        query_message_id = roots[0]
    query = messages.get(query_message_id)
    if not query or query["role"] != "user" or query["producer_call_id"] is not None:
        raise ValueError("The original query must be a user message without an agent producer")

    for call in calls.values():
        chunks = [event for event in call["chunks"] if event["data"]["kind_of_text"] == "observable"]
        ends = []
        for chunk in chunks:
            ends.append((ends[-1] if ends else 0) + len(chunk["data"]["text"]))
        text = "".join(chunk["data"]["text"] for chunk in chunks)
        if not text.strip():
            continue
        for sentence_id, (start, end) in enumerate(sentence_spans(text), 1):
            event = chunks[bisect_left(ends, end)]
            yield {
                "run_id": run_id,
                "agent_id": call["agent_id"],
                "call_id": call["call_id"],
                "agent_turn": call["agent_turn"],
                "query": query["content"],
                "input_messages": call["input_messages"],
                "reasoning_prefix": text[:end],
                "sentence_id": sentence_id,
                "sentence": text[start:end],
                "start_char": start,
                "end_char": end,
                "end_char_event_sequence": event["sequence"],
                "end_char_received_monotonic_ns": event["monotonic_ns"],
            }
