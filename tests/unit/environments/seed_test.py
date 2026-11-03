"""Seeding of the random instance generators."""
import pytest

from maenvs4vrp.utils.utils import data_equivalence
from tests.helpers import AEC_ENVS, PARALLEL_ENVS, module, nodes_for

GENERATORS = [("environments", e) for e in AEC_ENVS] + [("parallel_environments", e) for e in PARALLEL_ENVS]


@pytest.fixture(params=GENERATORS, ids=[f"{g[:3]}-{e}" for g, e in GENERATORS])
def generator(request):
    group, env_name = request.param
    return env_name, module(env_name, "instances_generator", group).InstanceGenerator()


def test_different_seed_gives_different_instances(generator):
    env_name, gen = generator
    n = nodes_for(env_name, 101)
    instance1 = gen.sample_instance(num_agents=10, num_nodes=n, seed=1)
    instance2 = gen.sample_instance(num_agents=10, num_nodes=n, seed=5)
    assert not data_equivalence(instance1, instance2)


def test_same_seed_gives_same_instance(generator):
    env_name, gen = generator
    n = nodes_for(env_name, 101)
    instance1 = gen.sample_instance(num_agents=10, num_nodes=n, seed=1)
    instance2 = gen.sample_instance(num_agents=10, num_nodes=n, seed=1)
    assert data_equivalence(instance1, instance2)
