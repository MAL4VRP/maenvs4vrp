# Environment step performance

Study of the episode time of the maenvs4vrp environments compared with [rl4co](https://github.com/ai4co/rl4co),
of where the time of a step goes, and of what can be made faster **without changing the structure of the
environments** (self-contained folders, same `td_state` keys, same methods and signatures).
Measured in November 2026 on branch `maintenance/library-review`.

## Summary

- A step of our environments cost **2.9 ms on GPU** (CVRP/CVRPTW, batch 512), against **0.6–1.0 ms for rl4co**.
  The gap is fixed per-step overhead (Python, TensorDict access, kernel launches and GPU synchronizations),
  not computation: rl4co runs 161 PyTorch operations per CVRPTW step, we ran 410 and waited for the GPU 11 times.
- Part of the gap is structural: rl4co models a single vehicle with a flat state, our AEC environments keep a
  per-agent state (`agents`, `cur_agent`) and build observations at every step.
- Phases 0 and 1 below remove every GPU synchronization from the step of 15 of the 17 AEC environments and cut
  20–25% of the operations. On GPU the step went from 2.9 ms to **1.8–2.5 ms** (−20 to −35%). Episodes are
  **bitwise identical**, except for two bug fixes listed below.
- In a REINFORCE training iteration with the attention model, the environment took **32–44%** of the time, so
  environment work alone can speed up training by at most that much. With the changes the environment is 18–23%
  faster per step and a **training iteration is 8–11% faster** (see [Training](#training-iteration)).

## Setup

- GTX 1080 (8 GB), i7-7800X (12 threads), PyTorch 2.7.0+cu126, TensorDict 0.9.1, rl4co 0.7.0 (installed in a
  separate virtual environment, `--no-deps`, with torchrl 0.9.2).
- Random feasible actions sampled from the action mask, which isolates the environment from the policy.
  `AgentSelector`, `DenseReward`, the step observations of the attention-model training scripts
  (`agent_cur_node_idx`, `action_mask`, `agent`). Agents: 6 / 10 / 20 for 20 / 50 / 100 customers.
- Batch 512, median of 5 episodes after one warm-up episode, `torch.set_num_threads(1)`.
- The machine was shared with other jobs (a PyVRP run on 11 cores, another process on the GPU), so the numbers
  carry ±20% noise; the conclusions rely on results that repeated across runs.
- rl4co pitfall: `CVRPTWEnv(device="cuda")` still generates on the CPU; use `env.reset(env.generator(B).to("cuda"))`.

Reproduce with `python docs/tools/benchmark_step.py --rl4co` (add `--device cpu`, `--envs`, `--customers` as needed).

## Where the time goes (before the changes)

Profile of one CVRPTW step on GPU (N=100, A=20, B=512, 4.4 ms instrumented):

| Part | Share | Notes |
|---|---|---|
| `_update_curr_agent_feasibility` | 16% | distance to all nodes, time windows, capacity |
| `_update_state` | 12% | 8 separate `scatter_` calls, one per agent field |
| reward `get_reward` | 9% | penalty computed at every step, assigned with a boolean mask |
| observations | 9% | of which 7% static features recomputed at every step by the training scripts |
| `_update_cur_agent` | 6% | new TensorDict with 7 `gather` + 7 `clone` |
| `_update_solution` | 3% | `torch.concat` of a growing tensor |
| rest | ~45% | TensorDict access, `observe`/`step` glue |

GPU synchronizations per step (each one stops the CPU until the GPU queue is empty, so the environment cannot
overlap with the policy):

| Source | Pattern |
|---|---|
| `step` feasibility asserts | `assert mask.gather(1, action).all()` |
| `_update_done` | `done[former_done] = True` |
| `_update_state` | `active[active.sum(1).eq(0), 0] = True` |
| reward | `penalty[is_last_step] = ...`, `reward[is_last_step] = ...` |
| `AgentSelector`, `SmallestTimeAgentSelector` | `avail[~active] = inf` |
| all-agents mask | `nonzero` + `if done_rows.numel() > 0` |
| `visited_nodes` | `visited[r, agent, action] = True` (advanced indexing) |

## Changes

### Phase 0: no change to the environments' logic
- **Training scripts** no longer request `nodes_static` at every step: the policy only uses it at reset.
- **Device:** `reset*(device=None)` now keeps the environment device (it defaulted to `"cpu"`, so an environment
  built on CUDA generated on the CPU or crashed), and the instance data is moved to the environment device.
  The multi-task generators draw on the CPU and move the result, so they now run on CUDA with the same instances
  as on CPU.
- **Golden rollouts** (`tests/unit/test_golden_rollouts.py`): 56 random episodes (every environment, three
  interaction modes) whose actions, rewards, penalties and final state must match a stored reference.

### Phase 1: same results, cheaper step
| Change | Where |
|---|---|
| `check_actions` constructor argument (default `True`) gates the per-step asserts; `False` in the training scripts | 22 `env.py` |
| boolean-mask assignments → `torch.where`, `|`, `scatter_` | `done`, active agents, 22 reward files, selectors, all-agents mask |
| `visited_nodes` advanced indexing → one `scatter_` | 17 AEC, parallel pcvrptw |
| `clone()` after `gather` removed (`gather` already returns a new tensor) | `_update_cur_agent`, 17 AEC |
| `AgentSelector`: `argmax` over the active mask instead of `arange`/`repeat`/`.to`/`argmin` | 21 selectors |
| hcvrp depot branch without `if is_depot.any()`; sdvrptw capacity mask without boolean indexing; mtdvrp index on the state device; parallel pcvrptw per-agent loops vectorized | single environments |
| `eval` → `getattr` for the feature methods | `core/env_observation_builder.py` |

Every change was checked against 446 exhaustive rollouts (every environment, every selector, both rewards,
`force_visit`, full observation lists, CPU and CUDA, with and without `check_actions`): all bitwise identical,
except the intended fixes.

Bugs found and fixed on the way:
- **top, toptw:** `visited_nodes[r, cur_agent_idx, action] = True` used an unsqueezed agent index and marked
  the visit for the agents of every batch row. No mask, observation or reward reads this field, so episodes are
  unchanged; `test_visited_nodes_match_the_solution` now covers all AEC environments.
- **cvrp, parallel cvrp:** crash when built on CUDA and reset without a device.
- **mtvrp, mtdvrp, gmtvrp, gmtdvrp:** crash on CUDA (generator mixing CPU and CUDA tensors).

### Operations and synchronizations per step

CUDA, B=512, 50 customers, 10 agents. "Now" with `check_actions=False`.

| env | ops before | syncs before | ops now | syncs now |
|---|---|---|---|---|
| cvrp | 358 | 11 | 261 | 0 |
| cvrptw | 410 | 11 | 313 | 0 |
| cvrpstw | 458 | 12 | 375 | 0 |
| dvrptw | 410 | 11 | 313 | 0 |
| dsvrptw | 554 | 12 | 465 | 1 |
| hcvrp | 428 | 14 | 322 | 0 |
| mdvrptw | 465 | 11 | 365 | 0 |
| pcvrp | 394 | 9 | 311 | 0 |
| pcvrptw | 409 | 9 | 326 | 0 |
| pdptw | 1008 | 13 | 911 | 1 |
| sdvrptw | 458 | 14 | 361 | 0 |
| top | 376 | 9 | 283 | 0 |
| toptw | 370 | 7 | 281 | 0 |
| mtvrp, gmtvrp, mtdvrp, gmtdvrp | did not run on CUDA | | 405–481 | 0 |

(The "ops now" of cvrpstw, dsvrptw, hcvrp and sdvrptw were measured before their last fixes; the "syncs now"
column is after them.) The two remaining synchronizations are deliberate: dsvrptw draws late customers with a
CPU `bernoulli_` (moving it to the GPU would change the random stream), and pdptw loops `count.max()` times.

## Results

Per-step time in ms (episode time / steps), random actions, B=512:

| device | env | N | rl4co | before | now | now, `check_actions=True` |
|---|---|---|---|---|---|---|
| cuda | cvrp | 20 | 0.67 | 2.98 | 1.88 | 1.87 |
| cuda | cvrp | 50 | 0.64 | 2.97 | 1.88 | 1.84 |
| cuda | cvrp | 100 | 0.61 | 2.89 | 2.05 | 1.76 |
| cuda | cvrptw | 20 | 0.98 | 2.85 | 3.67* | 2.55 |
| cuda | cvrptw | 50 | 0.93 | 2.90 | 2.13 | 2.21 |
| cuda | cvrptw | 100 | 0.92 | 3.19 | 2.47 | 2.27 |
| cpu | cvrp | 100 | 1.91 | 4.93 | 3.97 | 4.01 |
| cpu | cvrptw | 50 | 1.84 | 4.01 | 3.30 | 3.15 |
| cpu | cvrptw | 100 | 3.01 | 5.98 | 5.30 | 5.03 |

\* outlier: same code as the next column, which differs only by the asserts.

- With random actions rl4co takes more steps (it returns to the depot more often: 143 against 90 steps for
  CVRPTW N=100), so episode times are closer than step times (0.22 s now vs 0.13 s rl4co, against 0.29 s before).
- On GPU, `check_actions=True` costs little *in isolation*; its cost is that the CPU cannot run ahead of the GPU,
  which matters when a policy shares the GPU.

## Training iteration

REINFORCE with the attention model (`policy_net_am.py`), policy on CUDA, CVRPTW, one iteration = rollout +
backward + Adam step. Before = code at `8afa10b` with the old training setup (`nodes_static` at every step);
now = this branch with the new setup (`check_actions=False`). Median of 3 interleaved runs (each of 5 iterations
after a warm-up) on a free GPU; the runs agreed within 5%.

| case | iteration before | iteration now | change | env ms/step before → now | env share before → now |
|---|---|---|---|---|---|
| N=50, B=512, env on CPU | 0.440 s | 0.405 s | −8% | 2.75 → 2.29 | 33% → 30% |
| N=50, B=512, env on CUDA | 0.406 s | 0.373 s | −8% | 2.45 → 1.90 | 32% → 27% |
| N=100, B=128, env on CPU | 0.545 s | 0.500 s | −8% | 2.09 → 1.71 | 40% → 36% |
| N=100, B=128, env on CUDA | 0.547 s | 0.490 s | −10% | 2.37 → 1.84 | 44% → 38% |

(N=100 with B=512 does not fit the policy in 8 GB.) An earlier prototype that also precomputed the travel-time
matrix (next steps, item 2) reached −15 to −20%. A first attempt at this measurement while another job kept the GPU
at 57–100% was discarded: repeated runs of the same code varied by up to 2×.

## Not done (next steps)

Ranked by expected gain per effort:
1. **Reward:** cache the static depot→node distances at reset instead of recomputing them at every step.
2. **Travel-time matrix** `[B, N, N]` computed at reset: the feasibility mask becomes one `gather` instead of
   distances and rounding (bitwise identical in the prototype). Memory grows with N² (≈2 GB for B=512, N=1000),
   and it does not apply to stochastic or dynamic travel times.
3. **Solution buffer:** preallocate `solution/actions` and `solution/agents` and expose views, instead of
   `torch.concat` at every step (≈3% of the step; many variants across environments).
4. **`torch.compile` of the mask function**, possible now that the step has no synchronization.
5. Behaviour changes, for a minor release: end the episode without spending one step per unused agent; pack the
   agent state into one `[B, A, F]` tensor (conflicts with the uniform `td_state` keys).
6. Beyond the environment, the policy is 62–73% of an iteration, so training speed now depends mostly on the
   model and the training loop.

Pre-existing issues seen during the study, not addressed here:
- `force_visit=True` can raise "invalid multinomial distribution" (cvrp, cvrptw, dvrptw, dsvrptw, top in the
  agent-then-node mode; cvrpstw and sdvrptw with the full observation list).
- Parallel pcvrptw has no feasibility check in `step_all`.
- Several random generators ignore the `device` argument of `sample_instance` and generate on their construction
  device (the environment now moves the data, so this no longer breaks anything).
