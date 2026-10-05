# AGENTS.md - MAEnvs4VRP (Multi-Agent Environments for Vehicle Routing Problems)

PyTorch/TensorDict library of multi-agent environments for vehicle routing problems,
following an Agent-Environment-Cycle (AEC) design inspired by PettingZoo.
Paper: Gama et al., *INFORMS Journal on Computing* (2026), doi:10.1287/ijoc.2025.1211.

## Repository layout
- `maenvs4vrp/core/` - abstract base classes: `AECEnv` (`env.py`), `PEnv` (`parallel_env.py`),
  `InstanceBuilder`, `ObservationBuilder`, `BaseSelector`, `RewardFn`.
- `maenvs4vrp/environments/<problem>/` - one self-contained package per problem (AEC, one agent acts per step):
  `cvrp cvrptw cvrpstw dvrptw dsvrptw hcvrp mdvrptw pcvrp pcvrptw pdptw sdvrptw top toptw mtvrp mtdvrp gmtvrp gmtdvrp`.
- `maenvs4vrp/parallel_environments/<problem>/` - parallel variants (all agents act per step): `cvrp pcvrp pcvrptw top toptw`.
- `maenvs4vrp/neuro_solvers/` - reference neural solvers (attention model, 2D pointer) and training scripts.
- `maenvs4vrp/utils/` - shared helpers (`utils.py`, `ops.py`, `plotting.py`, `augment_utils.py`).
- `maenvs4vrp/learning_notebooks/` - tutorial notebooks (source of truth) and exercise `snippets/`.
- `docs/source/` - Sphinx docs (furo theme, napoleon, nbsphinx). One folder per environment mirroring the code.
- `tests/` - pytest suite; `tests/helpers.py` discovers the environments and builds/rolls them out.

## Core design rule: isolated but uniform environments
Each environment folder is **deliberately self-contained**: code is duplicated between environments
instead of being shared, so a problem can be read, copied and modified in isolation. Do **not**
refactor duplicated code into shared helpers. Instead, keep every folder structurally identical:

| File | Required content |
|---|---|
| `__init__.py` | present (may be empty) |
| `env.py` | `Environment(AECEnv)` (or `PEnv` for parallel envs) |
| `instances_generator.py` | `InstanceGenerator(InstanceBuilder)` - random instances |
| `toy_instance_generator.py` | `ToyInstanceGenerator(InstanceBuilder)` - small hand-made instances |
| `benchmark_instances_generator.py` | `BenchmarkInstanceGenerator(InstanceBuilder)` - only when public benchmarks exist |
| `observations.py` | `Observations(ObservationBuilder)` with the `POSSIBLE_*_FEATURES` lists |
| `env_agent_selector.py` | `AgentSelector`, `RandomSelector`, `SmallestTimeAgentSelector` |
| `env_agent_reward.py` | `DenseReward`, `SparseReward` |

- Known gaps: `top` (both groups) has no `toy_instance_generator.py`; `parallel_environments/cvrp` has no
  `env_agent_selector.py`; `dvrptw`, `dsvrptw`, `cvrp` and `pcvrp` have no benchmark generator.
- Public method names and signatures (`reset`, `observe`, `step`, `sample_action`, `check_solution_validity`, ...)
  must stay identical across environments; problem-specific parameters are added as keyword arguments.
- State keys in `td_state` should use the same names in every environment (e.g. `demands`, `coords`, `tw_low`, `tw_high`).
  Known legacy exception: `hcvrp` uses `demand` / `capacity`.
- When fixing a bug or changing an API in one environment, apply the same change to **every** environment
  that has the same code (grep for it) and to the matching `docs/source/<group>/<problem>/` pages.
- Every generator (instance, toy, benchmark, GTI) must raise `ValueError` for an unknown `sample_type`
  (`tests/unit/test_generators.py`), and random generation must work offline.

## Environment & commands
- Python >= 3.11. Development conda env: `maenvs4vrp`
  (`/home/gama/miniconda3/envs/maenvs4vrp/bin/python`). It has an editable install of a *different*
  checkout (`maenvs4vrp_dev`), so run with `PYTHONPATH=$PWD` (pytest already does, see `pyproject.toml`).
- Install: `pip install -e ".[dev,docs]"` (or `uv sync --all-extras`).
- Tests: `pytest` (a few minutes; excludes the `slow` and `notebooks` markers). Run the files relevant to
  your change, e.g. `pytest tests/unit/environments/reset_test.py -k cvrptw`; `pytest -m slow` for large
  instances and `pytest -m notebooks` to execute the tutorials.
- Larger solution checks: `pytest tests/unit/environments/check_solution_test.py --nodes 101 --agents 20 --batch 4`.
- The consistency tests (`test_structure.py`, `test_docstrings.py`, `test_docs.py`, `test_observations.py`)
  enforce the rules in this file: run them after touching any environment.
- Docs: `cd docs && make html` (output in `docs/build/`, not committed). Read the Docs builds with
  `fail_on_warning: true`, so check locally with `python -m sphinx -W --keep-going -b html docs/source /tmp/docs`.
- Observation docs pages are generated from the code: `python docs/tools/generate_observations_docs.py`
  (it also reports `POSSIBLE_*_FEATURES` entries without a `get_feat_*` method).
- Notebooks: `jupyter nbconvert --to notebook --execute --output-dir /tmp/nb <nb>.ipynb`, run from
  `maenvs4vrp/learning_notebooks/`. They are the single source; `docs/source/notebooks/` is filled at docs build time.
- Bibliography: cite with `[Key]_` and add entries only to `docs/source/content/references.rst`.

## Benchmark data
Benchmark and saved instances are downloaded on demand from Hugging Face datasets into
`maenvs4vrp/<group>/<problem>/data/`:
- `ai4co/routefinder`: the `mtvrp`, `mtdvrp`, `gmtvrp` and `gmtdvrp` benchmark generators.
- `ai4co/parco`: the `hcvrp` benchmark generator.
- `maenvs4vrp/environments` (`HF_REPO_ID`): every other generator.

These folders are git-ignored; never commit data, `docs/build/`, `dist/`, `*.egg-info` or `__pycache__`.
When a dataset moves, update the generators and this section together.

## Code style
- Match the surrounding code: 4-space indentation, snake_case functions, PascalCase classes,
  imports ordered stdlib / third-party / local.
- Tensors: batch dimension first; document shapes as `[B, N]`, `[B, A, N]` (B=batch, A=agents, N=nodes).
- Use type annotations in signatures of public functions.

## Docstrings (Google style, enforced across all environments)
```python
def reset(self, num_agents: int | None = None, seed: int | None = None) -> TensorDict:
    """
    Reset the environment and load a new batch of instances.

    Args:
        num_agents (int, optional): Total number of agents. Defaults to None.
        seed (int, optional): Random number generator seed. Defaults to None.

    Returns:
        TensorDict: Environment state after reset.

    Raises:
        ValueError: If ``sample_type`` is unknown.
    """
```
- Every module, public class and public method has a docstring; module docstrings are one line describing the file.
- Section headers end with `:` (`Args:`, `Returns:`, `Raises:`). Parameters: `name (type): Description.`
  with a space before the parenthesis; optional parameters add `, optional` and end with `Defaults to X.`
- List **exactly** the parameters in the signature (no removed parameters, no missing ones).
- Omit `Args:` when there are no parameters and omit `Returns:` when nothing is returned
  (no `N/a`, no `Returns: None.`). `__init__` never has a `Returns:` section.
- The same method must have the same docstring wording in every environment, except for problem-specific details.

## Contributions
- Keep diffs minimal and focused; no unrelated reformatting.
- Preserve backwards compatibility of public APIs unless fixing a real bug; update notebooks and docs when an API changes.
- Commit messages: short imperative summary line.
