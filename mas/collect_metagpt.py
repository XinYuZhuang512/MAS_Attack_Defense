"""MetaGPT Team runner with published, explicitly addressed role messages."""
import asyncio
from importlib.metadata import version

from metagpt.actions import Action
from metagpt.actions.add_requirement import UserRequirement
from metagpt.config2 import Config
from metagpt.configs.llm_config import LLMConfig
from metagpt.context import Context
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team
from pydantic import Field

from mas.collect_clean import instructions


class SolveTask(Action):
    recorder: object = Field(exclude=True)
    speaker: str
    recipient: str
    task_instruction: str

    async def run(self, history):
        message = history[-1]
        original = self.recorder.task["query"]
        inputs = [(self.recorder.query_id, "user", original)]
        if message.id != self.recorder.query_id:
            inputs.append((message.id, message.role, message.content))
        mid, answer = await asyncio.to_thread(
            self.recorder.respond, self.speaker,
            instructions(self.recorder.task["dataset"], self.task_instruction),
            inputs, [message.id], use_tools=self.speaker == "solver")
        return Message(id=mid, role=self.speaker, content=answer, cause_by=self,
                       sent_from=self.speaker, send_to={self.recipient})


async def run_team(run):
    # MetaGPT role objects initialize a provider, but the custom action issues
    # the only API calls through the recorded DeepSeek client.
    context = Context(config=Config(llm=LLMConfig(
        api_key="local-recording-adapter", model="deepseek-flash",
        base_url="https://api.deepseek.com",
    )))
    team = Team(context=context)
    specifications = (
        ("planner", "solver", "Write a plan that covers the user's constraints."),
        ("solver", "reviewer", "Execute the plan and verify results using the task tools if present."),
        ("reviewer", "user", "Use the solver's results to answer the original question accurately."),
    )
    roles = []
    for name, recipient, prompt in specifications:
        action = SolveTask(recorder=run, speaker=name, recipient=recipient, task_instruction=prompt, context=context)
        role = Role(name=name, profile=name, actions=[action], context=context)
        role._set_react_mode("by_order")
        roles.append(role)
    team.hire(roles)
    team.env.publish_message(Message(id=run.query_id, role="user", content=run.task["query"],
                                     cause_by=UserRequirement, send_to={"planner"}))
    # Each round resolves the next role's mailbox using MetaGPT's native environment.
    await team.run(n_round=5, auto_archive=False)
    final = [m for m in roles[-1].get_memories() if m.role == "reviewer"]
    if len(final) != 1:
        raise ValueError(f"MetaGPT produced {len(final)} reviewer messages")
    return final[0].id, final[0].content


def invoke(run):
    run.trace.event("framework", name="metagpt", version=version("metagpt"), workflow="Team")
    return asyncio.run(run_team(run))
