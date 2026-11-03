"""Reset and episode API of every AEC environment, for every agent selection mode."""
import pytest

from tests.helpers import AEC_ENVS, make_env, nodes_for, rollout

SELECTORS = ["AgentSelector", "SmallestTimeAgentSelector", "RandomSelector"]
NO_SELECTOR_MODES = ["agent_node", "joint", "node_agent"]


@pytest.mark.parametrize("env_name", AEC_ENVS)
@pytest.mark.parametrize("reset", ["reset", "reset_observe", "reset_agent_select", "reset_agent_select_observe"])
def test_reset_methods(env_name, reset):
    env = make_env(env_name)
    td = getattr(env, reset)(num_agents=3, num_nodes=nodes_for(env_name, 11))
    assert "done" in td.keys()
    assert not td["done"].any()
    if reset.endswith("observe"):
        assert "observations" in td.keys()


@pytest.mark.parametrize("env_name", AEC_ENVS)
@pytest.mark.parametrize("selector", SELECTORS)
def test_episode_with_agent_selector(env_name, selector):
    env = make_env(env_name, selector=selector)
    td = rollout(env, "select", num_agents=3, num_nodes=nodes_for(env_name, 11))
    assert td["done"].all()


@pytest.mark.parametrize("env_name", AEC_ENVS)
@pytest.mark.parametrize("mode", NO_SELECTOR_MODES)
def test_episode_without_agent_selector(env_name, mode):
    env = make_env(env_name, selector=None)
    td = rollout(env, mode, num_agents=3, num_nodes=nodes_for(env_name, 11))
    assert td["done"].all()


@pytest.mark.parametrize("env_name", AEC_ENVS)
def test_step_agent_select_requires_selector(env_name):
    env = make_env(env_name, selector=None)
    td = env.reset_observe(num_agents=2, num_nodes=nodes_for(env_name, 7))
    td = env.sample_joint(td)
    with pytest.raises(AssertionError):
        env.step_agent_select(td)
