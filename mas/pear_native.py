"""Use the pinned PEAR user-task registry and local simulation tools."""
from functools import lru_cache
from importlib import import_module
from inspect import Parameter, signature
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import get_type_hints

from pydantic import TypeAdapter
import yaml

DOMAINS = ("banking", "slack", "travel", "workspace")
DATA = Path(__file__).resolve().parents[1] / "datasets" / "pear"


@lru_cache
def load_suites():
    source = Path(os.environ["MAS_PEAR_SOURCE"]).resolve()
    if not (source / "mav" / "Tasks" / "task_suite.py").is_file():
        raise ValueError("PEAR source path is missing the official user task suite")
    sys.path.append(str(source))
    if dependency_path := os.environ.get("MAS_PEAR_DEPS"):
        sys.path.append(dependency_path)
    # The upstream task imports these names to annotate its unused native MAS runner.
    # Model execution in this project is owned by the selected framework adapter.
    framework = ModuleType("mav.MAS.framework")
    framework.MultiAgentSystem = object
    framework.MASRunResult = object
    sys.modules[framework.__name__] = framework
    suites = {}
    for domain in DOMAINS:
        module = import_module(f"mav.Tasks.{domain}.user_tasks")
        suites[domain] = module.task_suite
    return suites


def normal_value(raw, defaults):
    if isinstance(raw, str):
        return re.sub(r"\{([^{}]+)\}", lambda m: str(defaults.get(m[1], m[0])), raw)
    if isinstance(raw, dict):
        return {normal_value(k, defaults): normal_value(v, defaults) for k, v in raw.items()}
    if isinstance(raw, list):
        return [normal_value(v, defaults) for v in raw]
    return raw


@lru_cache
def load_initial_state(domain):
    suite = load_suites()[domain]
    path = DATA / domain
    raw = yaml.safe_load((path / "environment.yaml").read_text(encoding="utf-8"))
    vectors = path / "injection_vectors.yaml"
    defaults = {}
    if vectors.exists():
        defaults = {k: v["default"] for k, v in yaml.safe_load(vectors.read_text()).items()}
    return suite.environment_type.model_validate(normal_value(raw, defaults))


@lru_cache
def tools_for(domain):
    functions = {}
    schemas = []
    for function in load_suites()[domain].tools:
        params = list(signature(function).parameters.values())[1:]
        types = get_type_hints(function)
        inputs = {p.name: TypeAdapter(types[p.name]) for p in params}
        functions[function.__name__] = (function, inputs)
        schema = {"type": "object", "properties": {k: v.json_schema() for k, v in inputs.items()},
                  "required": [p.name for p in params if p.default is Parameter.empty],
                  "additionalProperties": False}
        schemas.append({"type": "function", "function": {
            "name": function.__name__,
            "description": (function.__doc__ or function.__name__).strip().splitlines()[0],
            "parameters": schema,
        }})
    return functions, schemas


class PearEnvironment:
    def __init__(self, row):
        domain = row["domain"]
        self.task = load_suites()[domain].user_tasks[row["upstream_id"]]
        self.environment = self.task.init_environment(load_initial_state(domain).model_copy(deep=True))
        self.before = self.environment.model_copy(deep=True)
        self.functions, self.tools = tools_for(domain)
        self.calls = []

    def invoke(self, name, arguments):
        from mav.Tasks.items import FunctionCall

        try:
            values = json.loads(arguments)
            function, types = self.functions[name]
            if not isinstance(values, dict) or set(values) - set(types):
                raise ValueError("Tool arguments do not match its schema")
            parsed = {k: types[k].validate_python(v) for k, v in values.items()}
            context = self.environment.user if name == "get_user_information" else self.environment
            result = function(context, **parsed)
            result_text = TypeAdapter(object).dump_json(result).decode()
            self.calls.append(FunctionCall(function=name, args=values))
            return result_text, values, False
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            return json.dumps({"error": str(exc)}), arguments, True

    def evaluate(self, output):
        score = self.task.utility_from_traces(output, self.before, self.environment, self.calls)
        return self.task.utility(output, self.before, self.environment) if score is None else score


def task_rows():
    for domain, suite in load_suites().items():
        for name, task in sorted(suite.user_tasks.items(), key=lambda item: int(item[0].rsplit("_", 1)[-1])):
            yield {"id": f"pear/{domain}/{name}", "dataset": "pear", "domain": domain,
                   "upstream_id": name, "query": task.PROMPT}
