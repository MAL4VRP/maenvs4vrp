"""
Golden rollouts: random episodes of every environment must reproduce the stored reference trajectories.

The reference file records, for fixed seeds, the action, reward, penalty and done flag of every step and the
final environment state. Performance work on the environments must leave it unchanged. Integer and boolean
tensors are compared exactly; floating point tensors with a small tolerance, so that the test does not depend
on the CPU vector instructions.

Regenerate the reference only when a change of behaviour is intended:
``python tests/unit/test_golden_rollouts.py --update``.
"""
import inspect
import io
import lzma
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.helpers import AEC_ENVS, PARALLEL_ENVS, make_env, nodes_for  # noqa: E402

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "golden_rollouts.pt.xz"

AEC_CONFIGS = {
    "select_dense": dict(mode="select", selector="AgentSelector", reward="DenseReward", force_visit=False),
    "select_stime_sparse_fv": dict(mode="select", selector="SmallestTimeAgentSelector", reward="SparseReward",
                                   force_visit=True),
    "joint_dense": dict(mode="joint", selector="AgentSelector", reward="DenseReward", force_visit=False),
}
PARALLEL_CONFIGS = {
    "parallel_dense": dict(mode="parallel", selector=None, reward="DenseReward", force_visit=False),
}
CASES = ([("environments", e, c) for e in AEC_ENVS for c in AEC_CONFIGS]
         + [("parallel_environments", e, c) for e in PARALLEL_ENVS for c in PARALLEL_CONFIGS])


def state_leaves(td, prefix: str = "") -> dict:
    """
    Flatten a TensorDict into a dictionary of CPU tensors.

    Args:
        td (TensorDict): TensorDict to flatten.
        prefix (str, optional): Key prefix. Defaults to "".

    Returns:
        dict: Flattened tensors, keyed by their path.
    """
    out = {}
    for k in td.keys():
        v = td[k]
        if hasattr(v, "keys"):
            out.update(state_leaves(v, f"{prefix}{k}/"))
        else:
            out[prefix + k] = v.detach().cpu().clone()
    return out


def play(group: str, env_name: str, config: str) -> dict:
    """
    Run one random episode of an environment.

    Args:
        group (str): "environments" or "parallel_environments".
        env_name (str): Environment name.
        config (str): Configuration name, a key of AEC_CONFIGS or PARALLEL_CONFIGS.

    Returns:
        dict: Per-step tensors stacked along the first dimension ("steps") and final state ("state").
    """
    c = {**AEC_CONFIGS, **PARALLEL_CONFIGS}[config]
    torch.manual_seed(7)
    env = make_env(env_name, group=group, selector=c["selector"], reward=c["reward"], batch_size=3, seed=0)
    kwargs = dict(num_agents=3, num_nodes=nodes_for(env_name, 11), seed=11)
    if "force_visit" in inspect.signature(env.reset).parameters:
        kwargs["force_visit"] = c["force_visit"]
    td = env.reset_agent_select_observe(**kwargs) if c["mode"] == "select" else env.reset_observe(**kwargs)
    steps = []
    while not td["done"].all():
        assert len(steps) < 1000, "episode did not finish"
        if c["mode"] == "select":
            td = env.step_agent_select_observe(env.sample_action(td))
        elif c["mode"] == "joint":
            td = env.step_observe(env.sample_joint(td))
        else:
            td = env.step_all_observe(env.sample_actions_all(td))
        action_key = "next_actions" if c["mode"] == "parallel" else "next_action"
        steps.append({k: td[k].detach().cpu().clone() for k in (action_key, "reward", "penalty", "done")})
    steps = {k: torch.stack([s[k] for s in steps]) for k in steps[0]}
    return {"steps": steps, "state": state_leaves(env.td_state)}


def assert_same(ref: torch.Tensor, new: torch.Tensor, where: str):
    """
    Compare two tensors: exactly for integers and booleans, with a tolerance for floating point.

    Args:
        ref (torch.Tensor): Reference tensor.
        new (torch.Tensor): New tensor.
        where (str): Description used in the failure message.
    """
    assert ref.dtype == new.dtype and ref.shape == new.shape, \
        f"{where}: {ref.dtype}{tuple(ref.shape)} != {new.dtype}{tuple(new.shape)}"
    if ref.is_floating_point():
        assert torch.allclose(ref, new, rtol=1e-5, atol=1e-5, equal_nan=True), f"{where}: values differ"
    else:
        assert torch.equal(ref, new), f"{where}: values differ"


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE.exists():
        pytest.skip(f"missing {REFERENCE}; run this file with --update")
    return torch.load(io.BytesIO(lzma.decompress(REFERENCE.read_bytes())))


@pytest.mark.parametrize("group,env_name,config", CASES)
def test_rollout_matches_reference(reference, group, env_name, config):
    key = f"{group}/{env_name}/{config}"
    assert key in reference, f"{key} not in the reference; regenerate it with --update"
    ref, new = reference[key], play(group, env_name, config)
    assert ref["steps"].keys() == new["steps"].keys(), f"{key}: step keys differ"
    for k in ref["steps"]:
        assert_same(ref["steps"][k], new["steps"][k], f"{key} steps {k}")
    assert ref["state"].keys() == new["state"].keys(), f"{key}: state keys differ"
    for k in ref["state"]:
        assert_same(ref["state"][k], new["state"][k], f"{key} state {k}")


if __name__ == "__main__":
    if "--update" not in sys.argv:
        sys.exit("usage: python tests/unit/test_golden_rollouts.py --update")
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save({f"{g}/{e}/{c}": play(g, e, c) for g, e, c in CASES}, buffer)
    REFERENCE.write_bytes(lzma.compress(buffer.getvalue(), preset=9))
    print(f"wrote {len(CASES)} rollouts to {REFERENCE} ({REFERENCE.stat().st_size / 1024:.0f} KiB)")
