"""Seeding of the benchmark instance generators (instance sets available locally only)."""
import pytest

from maenvs4vrp.utils.utils import data_equivalence
from tests.unit.environments.reset_bench_test import BENCHMARK_ENVS, local_benchmark_generator


@pytest.fixture(params=BENCHMARK_ENVS)
def benchmark_generator(request):
    return local_benchmark_generator(request.param)


def test_different_seed_benchmark_instance_generator(benchmark_generator):
    instance1 = benchmark_generator.sample_instance(num_agents=20, num_nodes=51, seed=1)
    instance2 = benchmark_generator.sample_instance(num_agents=20, num_nodes=51, seed=5)
    assert not data_equivalence(instance1, instance2)


def test_same_seed_benchmark_instance_generator(benchmark_generator):
    instance1 = benchmark_generator.sample_instance(num_agents=20, num_nodes=51, seed=1)
    instance2 = benchmark_generator.sample_instance(num_agents=20, num_nodes=51, seed=1)
    assert data_equivalence(instance1, instance2)
