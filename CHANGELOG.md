# Changelog

## [0.3.1] - Unreleased

Maintenance release: bug fixes, documentation, tests and repository hygiene.
The changes below can alter results obtained with 0.3; check them before
reproducing published experiments.

### Changed behavior
- **PDPTW random instances differ for the same seed.** The generator now places
  time windows so that every pickup/delivery pair can be served on its own.
- **PDPTW agents behave differently.** An agent can only return to the depot after
  delivering every load it picked up, and a node is only available if the pending
  deliveries can still be served and the depot reached in time. Before, agents
  could return carrying a load, which silently dropped it. The validator now
  rejects routes that return with undelivered pickups.
- **`td_state['solution']` has new keys:** `deliveries` (SDVRPTW, quantity delivered
  at each step) and `travel_times` (DSVRPTW, sampled travel times). The SDVRPTW and
  DSVRPTW validators use them and reject solutions they accepted before
  (over-capacity routes, over-delivery, customers served before they appear,
  wrong late-arrival penalties).
- **hcvrp:** capacity is reloaded per batch element at the depot (it was broadcast
  to `[B, B]` for `batch_size > 1`), and `check_solution_validity` accepts unvisited
  nodes like every other environment.
- **Every generator raises `ValueError` for an unknown `sample_type`.** Benchmark
  generators used to return a saved instance for any value other than `"random"`
  (so a typo went unnoticed), and toy generators raised `UnboundLocalError`.
  Benchmark generators accept `"random"` and `"saved"`, toy generators `"random"`
  (hcvrp also `"saved"`), GTI generators `"saved"`.
- Random instance generation never touches the network; saved instances are
  downloaded lazily.
- Observation features: legacy `get_feat_agents_*` methods are renamed
  `get_feat_other_agents_*`. Features that had no implementation or cannot apply to
  an environment were removed from its `POSSIBLE_*_FEATURES` lists (for example
  current-agent features of the parallel environments, time features of
  environments without time, `available_load` of the multi-task environments).
  Configurations that name them now fail with an `AssertionError` when the
  observations are set up, instead of failing during a rollout.

### Fixed
- Speed/capacity broadcasting in the all-agents feasibility of cvrpstw, dvrptw,
  mdvrptw, pcvrptw, pdptw and parallel top, and in the toptw validators.
- Crash of gmtdvrp for `batch_size > 1`; debug prints in the multi-task validators.
- NaN in the `frac_backhaul_demands` and pdptw `frac_demands` features; the
  multi-depot `dist2depot` feature uses each agent's own depot.
- Docstrings of the benchmark generators list the instance sets the code accepts
  (they all said "Solomon" or "Homberger").

### Packaging and repository
- `setup.py` and `pytest.ini` are removed; all settings are in `pyproject.toml`.
  Install with `pip install -e ".[dev,docs]"`.
- New `[notebooks]` extra; `pyyaml` and `packaging` are declared dependencies;
  the `[docs]` extra and `docs/requirements.txt` include `ipython`.
- Generated files and downloaded benchmark data are no longer tracked. The
  mtvrp/mtdvrp RouteFinder instances are downloaded from `ai4co/routefinder` on
  first use of their `BenchmarkInstanceGenerator`.
- Notebooks use the current generator API (`instance_name`, `list_of_instances`).
- New test suite with consistency checks (structure, docstrings, docs,
  observations, generators, validators) and a GitHub Actions workflow.
