"""Episode-level properties: determinism and reward consistency (validators are tested in test_validators.py)."""
import pytest
import torch

from tests.helpers import AEC_ENVS, PARALLEL_ENVS, make_env, nodes_for, rollout


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
