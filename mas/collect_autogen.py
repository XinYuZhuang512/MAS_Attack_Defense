"""AutoGen GraphFlow runner for normal benchmark tasks."""
import asyncio
from autogen_agentchat.agents import BaseChatAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage

from mas.collect_clean import instructions
from mas.workflows import autogen_branch

ROLES = {
    "planner": ("user",),
    "solver": ("planner",),
    "checker": ("planner",),
    "reviewer": ("solver", "checker"),
}
PROMPTS = {
    "planner": "Propose an approach and the user's constraints for two independent workers.",
    "solver": "Work out the answer. For an interactive task, use the provided tools to carry it out.",
    "checker": "Independently review the original request and approach. Identify pitfalls or constraints.",
    "reviewer": "Combine the solver and checker results into the final answer using verified findings.",
}


class BenchmarkAgent(BaseChatAgent):
    def __init__(self, name, run):
        super().__init__(name, PROMPTS[name])
        self.run_record = run

    @property
    def produced_message_types(self):
        return (TextMessage,)

    async def on_reset(self, cancellation_token):
        pass

    async def on_messages(self, messages, cancellation_token):
        parents = ROLES[self.name]
        triggers = [m.id for m in messages if m.source in parents]
        if len(triggers) != len(parents):
            raise ValueError(f"Incomplete AutoGen inputs for {self.name}")
        out_id, text = await asyncio.to_thread(
            self.run_record.respond, self.name, instructions(self.run_record.task["dataset"], self.description),
            [(m.id, m.source, m.content) for m in messages], triggers, use_tools=self.name == "solver")
        return Response(chat_message=TextMessage(id=out_id, source=self.name, content=text))


def invoke(run):
    agents = []
    for name, parents in ROLES.items():
        sources = tuple(dict.fromkeys(("user", *parents)))
        rules = [PerSourceFilter(source=source, position="last", count=1) for source in sources]
        agents.append(MessageFilterAgent(name, BenchmarkAgent(name, run), MessageFilterConfig(per_source=rules)))
    team = autogen_branch(*agents)
    question = TextMessage(id=run.query_id, source="user", content=run.task["query"])
    result = asyncio.run(team.run(task=question))
    for message in reversed(result.messages):
        if isinstance(message, TextMessage) and message.source == "reviewer":
            return message.id, message.content
    raise ValueError(f"AutoGen returned no final answer: {result.stop_reason}")
