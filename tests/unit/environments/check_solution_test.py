"""
Solution validity of the single-task AEC environments.

Sizes can be set from the command line, e.g.
``pytest tests/unit/environments/check_solution_test.py --nodes 51 --nodes 101 --agents 20 --batch 4``.
"""
import pytest

from tests.helpers import SINGLE_TASK_ENVS, make_env, nodes_for, rollout

SELECTORS = ["AgentSelector", "SmallestTimeAgentSelector", "RandomSelector"]
NO_SELECTOR_MODES = ["agent_node", "joint", "node_agent"]


def run_and_check(env, env_name, mode, sizes, **extra):
    for num_nodes in sizes.num_nodes:
        for num_agents in sizes.num_agents:
            rollout(env, mode, num_agents=num_agents, num_nodes=nodes_for(env_name, num_nodes),
                    batch_size=sizes.batch_size, device=sizes.device, **extra)
            env.check_solution_validity()


@pytest.mark.parametrize("env_name", SINGLE_TASK_ENVS)
@pytest.mark.parametrize("selector", SELECTORS)
def test_solution_with_agent_selector(env_name, selector, sizes):
    run_and_check(make_env(env_name, selector=selector), env_name, "select", sizes)


@pytest.mark.parametrize("env_name", SINGLE_TASK_ENVS)
@pytest.mark.parametrize("mode", NO_SELECTOR_MODES)
def test_solution_without_agent_selector(env_name, mode, sizes):
    run_and_check(make_env(env_name, selector=None), env_name, mode, sizes)


@pytest.mark.slow
@pytest.mark.parametrize("env_name", SINGLE_TASK_ENVS)
def test_solution_large_instances(env_name, sizes):
    env = make_env(env_name)
    for num_nodes in [101, 501]:
        for num_agents in [20, 50]:
            rollout(env, "select", num_agents=num_agents, num_nodes=nodes_for(env_name, num_nodes),
                    device=sizes.device)
            env.check_solution_validity()
