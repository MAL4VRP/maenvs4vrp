"""
Every environment package follows the same structure (see AGENTS.md): same files, classes, base classes
and public API. Environments are deliberately self-contained, so these tests are what keeps them uniform.
"""
import ast
import inspect

import pytest

from maenvs4vrp.core.env import AECEnv
from maenvs4vrp.core.env_agent_reward import RewardFn
from maenvs4vrp.core.env_agent_selector import BaseSelector
from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.parallel_env import PEnv
from tests.helpers import AEC_ENVS, PARALLEL_ENVS, PKG_DIR, REPO_DIR, module

REQUIRED_FILES = ["__init__.py", "env.py", "instances_generator.py", "toy_instance_generator.py",
                  "observations.py", "env_agent_selector.py", "env_agent_reward.py"]
# Known gaps, documented in AGENTS.md. Remove an entry when the file is added.
MISSING_FILES = {
    ("environments", "top"): {"toy_instance_generator.py"},
    ("parallel_environments", "top"): {"toy_instance_generator.py"},
    ("parallel_environments", "cvrp"): {"env_agent_selector.py"},
}
REQUIRED_CLASSES = {
    "env.py": {"Environment"},
    "instances_generator.py": {"InstanceGenerator"},
    "toy_instance_generator.py": {"ToyInstanceGenerator"},
    "benchmark_instances_generator.py": {"BenchmarkInstanceGenerator"},
    "observations.py": {"Observations"},
    "env_agent_selector.py": {"AgentSelector", "RandomSelector", "SmallestTimeAgentSelector"},
    "env_agent_reward.py": {"DenseReward", "SparseReward"},
}
BASES = {
    "instances_generator.py": ("InstanceGenerator", InstanceBuilder),
    "toy_instance_generator.py": ("ToyInstanceGenerator", InstanceBuilder),
    "benchmark_instances_generator.py": ("BenchmarkInstanceGenerator", InstanceBuilder),
    "observations.py": ("Observations", ObservationBuilder),
    "env_agent_reward.py": ("DenseReward", RewardFn),
    "env_agent_selector.py": ("AgentSelector", BaseSelector),
}
AEC_API = ["reset", "reset_observe", "reset_agent_select", "reset_agent_select_observe", "observe",
           "sample_action", "sample_agent", "sample_joint", "set_cur_agent", "step", "step_observe",
           "step_agent_select", "step_agent_select_observe", "check_solution_validity"]
# methods whose full signature must be identical in every AEC environment
AEC_FIXED_SIGNATURES = ["__init__", "observe", "sample_action", "sample_agent", "sample_joint", "set_cur_agent",
                        "step", "step_observe", "step_agent_select", "check_solution_validity"]
PARALLEL_API = ["reset", "reset_observe", "observe", "sample_actions_all", "step_all", "step_all_observe",
                "check_solution_validity"]
ENVS = [("environments", e) for e in AEC_ENVS] + [("parallel_environments", e) for e in PARALLEL_ENVS]
IDS = [f"{g[:3]}-{e}" for g, e in ENVS]


@pytest.mark.parametrize("group, env_name", ENVS, ids=IDS)
def test_required_files(group, env_name):
    present = {p.name for p in (PKG_DIR / group / env_name).glob("*.py")}
    missing = set(REQUIRED_FILES) - present - MISSING_FILES.get((group, env_name), set())
    assert not missing, f"missing files: {sorted(missing)}"


@pytest.mark.parametrize("group, env_name", ENVS, ids=IDS)
def test_known_gaps_are_still_gaps(group, env_name):
    present = {p.name for p in (PKG_DIR / group / env_name).glob("*.py")}
    fixed = MISSING_FILES.get((group, env_name), set()) & present
    assert not fixed, f"{sorted(fixed)} now exist: remove them from MISSING_FILES and AGENTS.md"


@pytest.mark.parametrize("group, env_name", ENVS, ids=IDS)
def test_required_classes(group, env_name):
    for path in (PKG_DIR / group / env_name).glob("*.py"):
        required = REQUIRED_CLASSES.get(path.name)
        if not required:
            continue
        classes = {n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}
        assert required <= classes, f"{path.name}: missing {sorted(required - classes)}"


@pytest.mark.parametrize("group, env_name", ENVS, ids=IDS)
def test_base_classes(group, env_name):
    env_cls = module(env_name, "env", group).Environment
    assert issubclass(env_cls, AECEnv if group == "environments" else PEnv)
    for file_name, (cls_name, base) in BASES.items():
        if (PKG_DIR / group / env_name / file_name).exists():
            cls = getattr(module(env_name, file_name[:-3], group), cls_name)
            assert issubclass(cls, base), f"{cls_name} must extend {base.__name__}"


@pytest.mark.parametrize("env_name", AEC_ENVS)
def test_aec_public_api(env_name):
    env_cls = module(env_name, "env").Environment
    missing = [m for m in AEC_API if not callable(getattr(env_cls, m, None))]
    assert not missing, f"missing methods: {missing}"


@pytest.mark.parametrize("method", AEC_FIXED_SIGNATURES)
def test_aec_signatures_are_identical(method):
    signatures = {e: str(inspect.signature(getattr(module(e, "env").Environment, method))) for e in AEC_ENVS}
    reference = max(set(signatures.values()), key=list(signatures.values()).count)
    different = {e: s for e, s in signatures.items() if s != reference}
    assert not different, f"{method}{reference} differs in {different}"


@pytest.mark.parametrize("env_name", PARALLEL_ENVS)
def test_parallel_public_api(env_name):
    env_cls = module(env_name, "env", "parallel_environments").Environment
    missing = [m for m in PARALLEL_API if not callable(getattr(env_cls, m, None))]
    assert not missing, f"missing methods: {missing}"


@pytest.mark.parametrize("group, env_name", ENVS, ids=IDS)
def test_documentation_pages_exist(group, env_name):
    docs = REPO_DIR / "docs" / "source" / group / env_name
    assert (docs / f"{env_name}.rst").exists()
    for page in ["environment/environment.rst", "observations/observations.rst", "generation/generation.rst"]:
        assert (docs / page).exists(), page
    index = (REPO_DIR / "docs" / "source" / "index.rst").read_text()
    overview = (REPO_DIR / "docs" / "source" / "parallel_environments" / "parallel_environments.rst").read_text()
    toc = index if group == "environments" else overview
    entry = f"environments/{env_name}/{env_name}" if group == "environments" else f"{env_name}/{env_name}"
    assert entry in toc, f"{entry} is not in the documentation toctree"
