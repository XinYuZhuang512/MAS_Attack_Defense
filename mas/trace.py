"""Append observable MAS events in a single orchestrator process, including parallel branches."""
import json
import threading
import time
from pathlib import Path
from uuid import uuid4


class Trace:
    def __init__(self, path, run_id, task_id):
        self.run_id = run_id
        self.stream = Path(path).open("x", encoding="utf-8")
        self.lock = threading.Lock()
        self.sequence = 0
        self.event("run_started", task_id=task_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.event("run_finished", status="failed" if exc_type else "completed")
        finally:
            self.stream.close()

    def event(self, kind, **data):
        with self.lock:
            record = dict(run_id=self.run_id, sequence=self.sequence, time_ns=time.time_ns(),
                          monotonic_ns=time.monotonic_ns(), event=kind, data=data)
            self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()
            self.sequence += 1

    def message(self, role, content, producer_call_id=None):
        message_id = uuid4().hex
        self.event("message_created", message_id=message_id, role=role, content=content,
                   producer_call_id=producer_call_id)
        return message_id

    def deliver(self, message_id, recipient_agent_id):
        self.event("message_delivered", message_id=message_id, recipient_agent_id=recipient_agent_id)

    def start_call(self, agent_id, agent_turn, input_messages, trigger_message_ids=()):
        """input_messages holds the ordered actual input records, including role and message ID."""
        call_id = uuid4().hex
        self.event("call_started", call_id=call_id, agent_id=agent_id, agent_turn=agent_turn,
                   input_messages=list(input_messages), trigger_message_ids=list(trigger_message_ids))
        return call_id

    def reasoning(self, call_id, text, kind="observable"):
        if kind not in {"observable", "summary"}:
            raise ValueError("Expected observable text or a summary")
        self.event("reasoning_chunk", call_id=call_id, kind_of_text=kind, text=text)

    def finish_call(self, call_id, output_message_ids, reasoning_status, status="completed"):
        if reasoning_status not in {"observable", "summary", "unavailable"}:
            raise ValueError("Unknown reasoning status")
        self.event("call_finished", call_id=call_id, status=status, reasoning_status=reasoning_status,
                   output_message_ids=list(output_message_ids))
