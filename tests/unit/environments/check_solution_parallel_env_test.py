"""
Solution validity of the parallel environments.

Sizes can be set from the command line (``--nodes``, ``--agents``, ``--batch``).
"""
import pytest

from tests.helpers import PARALLEL_ENVS, make_env, rollout


@pytest.mark.parametrize("env_name", PARALLEL_ENVS)
def test_solution(env_name, sizes):
    env = make_env(env_name, group="parallel_environments")
    for num_nodes in sizes.num_nodes:
        for num_agents in sizes.num_agents:
            rollout(env, "parallel", num_agents=num_agents, num_nodes=num_nodes,
                    batch_size=sizes.batch_size, device=sizes.device)
            env.check_solution_validity()


@pytest.mark.slow
@pytest.mark.parametrize("env_name", PARALLEL_ENVS)
def test_solution_large_instances(env_name, sizes):
    env = make_env(env_name, group="parallel_environments")
    for num_nodes in [101, 1001]:
        for num_agents in [20, 50]:
            rollout(env, "parallel", num_agents=num_agents, num_nodes=num_nodes, device=sizes.device)
            env.check_solution_validity()
