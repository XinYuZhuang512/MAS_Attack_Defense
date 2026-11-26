"""Construct the two pilot topologies with caller-provided agents or model backends."""


def autogen_branch(planner, left, right, reviewer):
    from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

    agents = [planner, left, right, reviewer]
    if len({agent.name for agent in agents}) != len(agents):
        raise ValueError("Every role requires a distinct agent name")
    graph = DiGraphBuilder()
    for agent in agents:
        graph.add_node(agent)
    for worker in (left, right):
        graph.add_edge(planner, worker)
        graph.add_edge(worker, reviewer)
    return GraphFlow(participants=agents, graph=graph.build())


def camel_chain(query, *, user_model, assistant_model):
    from camel.societies import RolePlaying

    return RolePlaying(user_role_name="Task coordinator", assistant_role_name="Task assistant",
                       task_prompt=query, with_task_specify=False,
                       user_agent_kwargs={"model": user_model},
                       assistant_agent_kwargs={"model": assistant_model})
