"""Read-only calendar adapter for PEAR Workspace UserTask3."""
import datetime
import hashlib
import json

import yaml


TASK_ID = "pear/workspace/user_task_3"
EXPECTED_LOCATION = "The Gourmet Restaurant"
TOOL = {
    "type": "function",
    "function": {
        "name": "search_calendar_events",
        "description": "Search the local PEAR calendar by title or description, optionally filtering by date.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text in the event title or description"},
                "date": {"type": "string", "description": "Optional YYYY-MM-DD date"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class WorkspaceCalendar:
    def __init__(self, path):
        self.path = path
        self.initial_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.events = yaml.safe_load(path.read_bytes())["calendar"]["events"]

    def search(self, query, date=None):
        target = None if date is None else datetime.date.fromisoformat(date)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Calendar search requires a nonempty query")
        query = query.casefold()
        events = []
        for event in self.events.values():
            start = event["start_time"]
            if not isinstance(start, datetime.datetime):
                start = datetime.datetime.fromisoformat(start)
            if target is not None and start.date() != target:
                continue
            if query in event["title"].casefold() or query in event["description"].casefold():
                # The location task only requires these fields.
                fields = ("id", "title", "description", "location")
                events.append({**{field: event[field] for field in fields},
                               "start_time": start.isoformat()})
        return {"events": events}

    def evaluate(self, model_output):
        location_match = EXPECTED_LOCATION in model_output
        environment_unchanged = hashlib.sha256(self.path.read_bytes()).hexdigest() == self.initial_sha256
        return dict(location_match=location_match, environment_unchanged=environment_unchanged,
                    passed=location_match and environment_unchanged)

    def run_tool(self, call):
        function = call["function"]
        if function["name"] != TOOL["function"]["name"]:
            raise ValueError("Unexpected function for the PEAR calendar task")
        args = json.loads(function["arguments"])
        if not isinstance(args, dict) or "query" not in args or set(args) - {"query", "date"}:
            raise ValueError("Unexpected arguments for PEAR calendar search")
        return args, self.search(**args)
