"""Episode-level properties: determinism and reward consistency (validators are tested in test_validators.py)."""
import pytest
import torch

from tests.helpers import AEC_ENVS, PARALLEL_ENVS, make_env, module, nodes_for, rollout

ALL_ENVS = [("environments", e) for e in AEC_ENVS] + [("parallel_environments", e) for e in PARALLEL_ENVS]


def play(env_name, reward="DenseReward", actions=None, seed=0):
    """Run one episode with the AgentSelector, sampling actions or replaying the given ones."""
    torch.manual_seed(seed)
    env = make_env(env_name, reward=reward, batch_size=2, seed=0)
    td = env.reset_agent_select_observe(num_agents=3, num_nodes=nodes_for(env_name, 11), seed=1)
    played, total = [], torch.zeros(2, 1)
    i = 0
    while not td["done"].all():
        # always sample, so that replays consume the random number generator like the original episode
        # (stochastic environments draw travel times from it)
        td = env.sample_action(td)
        if actions is not None:
            td["next_action"] = actions[i]
        played.append(td["next_action"].clone())
        td = env.step_agent_select_observe(td)
        total = total + td["reward"] + td["penalty"]
        i += 1
    return env, played, total


@pytest.mark.parametrize("env_name", AEC_ENVS)
def test_same_seed_gives_same_episode(env_name):
    env1, actions1, total1 = play(env_name, seed=3)
    env2, actions2, total2 = play(env_name, seed=3)
    assert torch.equal(env1.td_state["solution"]["actions"], env2.td_state["solution"]["actions"])
    assert torch.equal(env1.td_state["solution"]["agents"], env2.td_state["solution"]["agents"])
    assert torch.allclose(total1, total2)


@pytest.mark.parametrize("env_name", AEC_ENVS)
def test_dense_and_sparse_rewards_agree(env_name):
    """Summed over an episode, the dense reward equals the sparse (final) reward."""
    _, actions, dense_total = play(env_name, reward="DenseReward")
    _, _, sparse_total = play(env_name, reward="SparseReward", actions=actions)
    assert torch.allclose(dense_total, sparse_total, atol=1e-3), (dense_total, sparse_total)


@pytest.mark.parametrize("env_name", PARALLEL_ENVS)
def test_parallel_same_seed_gives_same_episode(env_name):
    results = []
    for _ in range(2):
        torch.manual_seed(3)
        env = make_env(env_name, group="parallel_environments", batch_size=2, seed=0)
        rollout(env, "parallel", num_agents=3, num_nodes=11, seed=1)
        results.append(env.td_state["solution"]["actions"].clone())
    assert torch.equal(results[0], results[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("group,env_name", ALL_ENVS)
@pytest.mark.parametrize("built_on_cuda", [True, False])
def test_episode_runs_on_the_environment_device(group, env_name, built_on_cuda):
    """
    An environment built on CUDA keeps its device when reset is called without one, and reset(device=...)
    moves a CPU environment to that device.
    """
    kwargs = dict(instance_generator_object=module(env_name, "instances_generator", group).InstanceGenerator(),
                  obs_builder_object=module(env_name, "observations", group).Observations(),
                  reward_evaluator=module(env_name, "env_agent_reward", group).DenseReward(), batch_size=2,
                  device="cuda" if built_on_cuda else None)
    if group == "environments":
        kwargs["agent_selector_object"] = module(env_name, "env_agent_selector", group).AgentSelector()
    env = module(env_name, "env", group).Environment(**kwargs)
    reset_kwargs = {} if built_on_cuda else {"device": "cuda"}
    rollout(env, "select" if group == "environments" else "parallel", num_agents=3,
            num_nodes=nodes_for(env_name, 11), seed=1, **reset_kwargs)
    devices = {v.device.type for v in env.td_state.values(include_nested=True, leaves_only=True)}
    assert devices == {"cuda"}, devices


@pytest.mark.parametrize("env_name", AEC_ENVS)
def test_visited_nodes_match_the_solution(env_name):
    """agents/visited_nodes marks exactly the nodes each agent visited in the solution, in its own batch row."""
    torch.manual_seed(0)
    env = make_env(env_name, batch_size=4)
    rollout(env, "select", num_agents=3, num_nodes=nodes_for(env_name, 11), seed=0)
    solution = env.td_state["solution"]
    expected = torch.zeros_like(env.td_state["agents"]["visited_nodes"])
    for b in range(expected.shape[0]):
        expected[b, solution["agents"][b], solution["actions"][b]] = True
    assert torch.equal(env.td_state["agents"]["visited_nodes"], expected)
