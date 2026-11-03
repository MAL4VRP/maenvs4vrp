"""Every declared observation feature works and has the expected shape."""
import pytest
import torch

from tests.helpers import AEC_ENVS, PARALLEL_ENVS, make_env, module, nodes_for

SECTIONS = {
    "nodes_static": "POSSIBLE_NODES_STATIC_FEATURES",
    "edges_static": "POSSIBLE_EDGES_STATIC_FEATURES",
    "nodes_dynamic": "POSSIBLE_NODES_DYNAMIC_FEATURES",
    "agent": "POSSIBLE_AGENT_FEATURES",
    "other_agents": "POSSIBLE_OTHER_AGENTS_FEATURES",
    "all_agents": "POSSIBLE_ALL_AGENTS_FEATURES",
    "global": "POSSIBLE_GLOBAL_FEATURES",
}
DICT_SECTIONS = ("nodes_static", "edges_static")
B, A, N = 2, 3, 11
ENVS = [("environments", e) for e in AEC_ENVS] + [("parallel_environments", e) for e in PARALLEL_ENVS]


def feature_list(section_features: dict) -> dict:
    """Build a feature_list dict with the given features per section and every other section empty."""
    fl = {}
    for section in SECTIONS:
        feats = section_features.get(section, [])
        fl[section] = {f: {"feat": f, "norm": None} for f in feats} if section in DICT_SECTIONS else list(feats)
    return fl


def expected_shape(section, n_feats, num_nodes, num_agents):
    return {"nodes_static": (B, num_nodes, n_feats), "nodes_dynamic": (B, num_nodes, n_feats),
            "edges_static": (B, num_nodes, num_nodes, n_feats), "agent": (B, n_feats),
            "other_agents": (B, num_agents, n_feats), "all_agents": (B, num_agents, n_feats),
            "global": (B, n_feats)}[section]


def run_steps(group, env_name, observations, sections, steps=3):
    """Reset and run a few steps, returning the observations of every step and the problem size."""
    env = make_env(env_name, group=group, observations=observations, batch_size=B)
    num_nodes = nodes_for(env_name, N)
    out = []
    if group == "environments":
        td = env.reset_agent_select_observe(num_agents=A, num_nodes=num_nodes, obs_list=sections)
    else:
        td = env.reset_observe(num_agents=A, num_nodes=num_nodes, obs_list=sections)
    out.append(td["observations"])
    for _ in range(steps):
        if td["done"].all():
            break
        if group == "environments":
            td = env.step_agent_select_observe(env.sample_action(td), obs_list=sections)
        else:
            td = env.step_all_observe(env.sample_actions_all(td), obs_list=sections)
        out.append(td["observations"])
    # multi-depot environments have num_agents agents per depot
    return out, num_nodes, env.num_agents


FEATURE_CASES = [
    pytest.param(group, env_name, section, feat, id=f"{group[:3]}-{env_name}-{section}-{feat}")
    for group, env_name in ENVS
    for section, attr in SECTIONS.items()
    for feat in getattr(module(env_name, "observations", group).Observations, attr, [])
]


@pytest.mark.parametrize("group, env_name, section, feat", FEATURE_CASES)
def test_each_declared_feature(group, env_name, section, feat):
    obs_cls = module(env_name, "observations", group).Observations
    steps, num_nodes, num_agents = run_steps(group, env_name, obs_cls(feature_list({section: [feat]})), [section])
    for observations in steps:
        tensor = observations[f"{section}_obs"]
        assert tuple(tensor.shape) == expected_shape(section, 1, num_nodes, num_agents)
        assert torch.isfinite(tensor.float()).all()


@pytest.mark.parametrize("group, env_name", ENVS, ids=[f"{g[:3]}-{e}" for g, e in ENVS])
def test_all_features_together_match_declared_dims(group, env_name):
    obs_cls = module(env_name, "observations", group).Observations
    all_feats = {s: list(dict.fromkeys(getattr(obs_cls, a, []))) for s, a in SECTIONS.items()}
    observations = obs_cls(feature_list(all_feats))
    sections = [s for s, f in all_feats.items() if f]
    steps, num_nodes, num_agents = run_steps(group, env_name, observations, sections, steps=1)
    dims = {"nodes_static": observations.get_nodes_static_feat_dim(),
            "nodes_dynamic": observations.get_nodes_dynamic_feat_dim(),
            "agent": observations.get_agent_feat_dim(),
            "other_agents": observations.get_other_agents_feat_dim(),
            "all_agents": observations.get_all_agents_feat_dim(),
            "global": observations.get_global_feat_dim(),
            "edges_static": len(all_feats["edges_static"])}
    for section in sections:
        tensor = steps[-1][f"{section}_obs"]
        assert tuple(tensor.shape) == expected_shape(section, dims[section], num_nodes, num_agents), section


@pytest.mark.parametrize("group, env_name", ENVS, ids=[f"{g[:3]}-{e}" for g, e in ENVS])
def test_default_features_are_declared(group, env_name):
    observations = module(env_name, "observations", group).Observations()
    for section, attr in SECTIONS.items():
        default = observations.default_feature_list.get(section) or []
        names = [v.get("feat") for v in default.values()] if isinstance(default, dict) else list(default)
        undeclared = set(names) - set(getattr(observations, attr, []))
        assert not undeclared, f"{section}: {sorted(undeclared)}"
