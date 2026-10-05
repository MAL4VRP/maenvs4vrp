"""Episodes on benchmark instances (only instance sets available locally are used, so no network is needed)."""
import pytest

from tests.helpers import PKG_DIR, make_env, module, rollout

BENCHMARK_ENVS = sorted(p.parent.name for p in (PKG_DIR / "environments").glob("*/benchmark_instances_generator.py"))


def local_benchmark_generator(env_name):
    """Benchmark generator for the first instance set available locally, or skip the test."""
    gen_cls = module(env_name, "benchmark_instances_generator").BenchmarkInstanceGenerator
    for instance_name, instances in gen_cls.get_list_of_instances().items():
        if instances:
            return gen_cls(instance_name=instance_name, list_of_instances=list(instances)[:2])
    pytest.skip(f"no {env_name} benchmark instances available locally")


@pytest.mark.parametrize("env_name", BENCHMARK_ENVS)
def test_benchmark_reset_and_observe(env_name):
    env = make_env(env_name, generator=local_benchmark_generator(env_name), batch_size=None)
    td = env.reset_agent_select()
    td = env.observe(td)
    assert "observations" in td.keys()


@pytest.mark.parametrize("env_name", BENCHMARK_ENVS)
@pytest.mark.parametrize("selector", ["AgentSelector", "SmallestTimeAgentSelector", "RandomSelector"])
def test_benchmark_episode(env_name, selector):
    env = make_env(env_name, selector=selector, generator=local_benchmark_generator(env_name), batch_size=None)
    td = rollout(env, "select")
    assert td["done"].all()
    env.check_solution_validity()
