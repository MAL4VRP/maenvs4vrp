"""
Solution validity of the multi-task environments, for random variants and for every variant preset.

Sizes can be set from the command line (``--nodes``, ``--agents``, ``--batch``).
"""
import pytest

from tests.helpers import MULTITASK_ENVS, make_env, rollout

VARIANT_PRESETS = [
    'cvrp', 'ovrp', 'ovrpb', 'ovrpbl', 'ovrpbltw', 'ovrpbtw',
    'ovrpl', 'ovrpltw', 'ovrpmb', 'ovrpmbl', 'ovrpmbltw', 'ovrpmbtw',
    'ovrptw', 'vrpb', 'vrpbl', 'vrpbltw', 'vrpbtw', 'vrpl',
    'vrpltw', 'vrpmb', 'vrpmbl', 'vrpmbltw', 'vrpmbtw', 'vrptw',
]
SELECTORS = ["AgentSelector", "SmallestTimeAgentSelector", "RandomSelector"]
NO_SELECTOR_MODES = ["agent_node", "joint", "node_agent"]


def run_and_check(env, mode, sizes, **extra):
    for num_nodes in sizes.num_nodes:
        for num_agents in sizes.num_agents:
            rollout(env, mode, num_agents=num_agents, num_nodes=num_nodes,
                    batch_size=sizes.batch_size, device=sizes.device, **extra)
            env.check_solution_validity()


@pytest.mark.parametrize("env_name", MULTITASK_ENVS)
@pytest.mark.parametrize("selector", SELECTORS)
def test_solution_with_agent_selector(env_name, selector, sizes):
    run_and_check(make_env(env_name, selector=selector), "select", sizes)


@pytest.mark.parametrize("env_name", MULTITASK_ENVS)
@pytest.mark.parametrize("mode", NO_SELECTOR_MODES)
def test_solution_without_agent_selector(env_name, mode, sizes):
    run_and_check(make_env(env_name, selector=None), mode, sizes)


@pytest.mark.parametrize("env_name", MULTITASK_ENVS)
@pytest.mark.parametrize("variant", VARIANT_PRESETS)
def test_solution_variant_presets(env_name, variant, sizes):
    env = make_env(env_name)
    rollout(env, "select", num_agents=sizes.num_agents[0], num_nodes=sizes.num_nodes[0],
            batch_size=sizes.batch_size, device=sizes.device, variant_preset=variant)
    env.check_solution_validity()


@pytest.mark.slow
@pytest.mark.parametrize("env_name", MULTITASK_ENVS)
@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("variant", VARIANT_PRESETS)
def test_solution_variant_presets_all_selectors(env_name, selector, variant, sizes):
    env = make_env(env_name, selector=selector)
    run_and_check(env, "select", sizes, variant_preset=variant)
