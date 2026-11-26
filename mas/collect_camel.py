"""CAMEL RolePlaying runner with a backend that logs the actual model inputs."""
from importlib.metadata import version
import json
import time

from camel.models import BaseModelBackend
from camel.utils import BaseTokenCounter
from openai.types.chat import ChatCompletion

from mas.collect_clean import instructions
from mas.workflows import camel_chain


class SizeCounter(BaseTokenCounter):
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")

    def count_tokens_from_messages(self, messages):
        # A local conservative estimate for CAMEL's context budget.
        return len(json.dumps(messages, ensure_ascii=False))


class FlashBackend(BaseModelBackend):
    def __init__(self, name, run, known):
        super().__init__("deepseek-flash", model_config_dict={"stream": False}, token_counter=SizeCounter())
        self.name = name
        self.run_record = run
        self.known = known
        self.last_output = None

    @property
    def token_limit(self):
        return 128000

    @property
    def token_counter(self):
        return self._token_counter

    def _run(self, messages, response_format=None, tools=None):
        inputs, triggers = [], []
        for message in messages:
            item = dict(message)
            candidates = self.known.get(item["content"], [])
            producer_is_self = item["role"] == "assistant"
            matches = [mid for mid, sender in candidates if (sender == self.name) == producer_is_self]
            if matches and item["role"] in {"assistant", "user"}:
                item["message_id"] = matches[-1]
                if not producer_is_self:
                    self.run_record.trace.deliver(item["message_id"], self.name)
                    triggers.append(item["message_id"])
            inputs.append(item)
        if not triggers:
            raise ValueError(f"CAMEL did not preserve the incoming message for {self.name}")
        mid, answer = self.run_record.respond_payload(self.name, inputs, triggers[-1:],
                                                       use_tools=self.name == "assistant")
        self.known.setdefault(answer, []).append((mid, self.name))
        self.last_output = (mid, answer)
        return ChatCompletion(id=mid, created=int(time.time()), model="deepseek-flash",
                              object="chat.completion", choices=[
                                  {"index": 0, "finish_reason": "stop",
                                   "message": {"role": "assistant", "content": answer}},
                              ])

    async def _arun(self, messages, response_format=None, tools=None):
        return self._run(messages, response_format, tools)


def invoke(run):
    run.trace.event("framework", name="camel", version=version("camel-ai"), workflow="RolePlaying")
    query = run.task["query"]
    known = {query: [(run.query_id, "user")]}
    coordinator = FlashBackend("coordinator", run, known)
    assistant = FlashBackend("assistant", run, known)
    goal = (query + "\n\n" + instructions(run.task["dataset"], "Collaborate to answer this user.")
            + "\nGive a useful answer after the first instruction. In the second round, review and finalize it. "
              "Use any supplied simulator tools only when carrying out the user task.")
    session = camel_chain(goal, user_model=coordinator, assistant_model=assistant)
    message = session.init_chat(init_msg_content=query)
    for _ in range(2):
        response, _ = session.step(message)
        if response.terminated or not response.msgs:
            break
        message = response.msgs[0]
    if assistant.last_output is None:
        raise ValueError("CAMEL did not generate an assistant answer")
    return assistant.last_output
