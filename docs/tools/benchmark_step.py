"""Measure the episode and per-step time of the AEC environments with random feasible actions."""
import argparse
import importlib
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# observations requested at every step by the attention-model training scripts
STEP_OBS = ['agent_cur_node_idx', 'action_mask', 'agent']
RESET_OBS = ['agent_cur_node_idx', 'nodes_static', 'action_mask', 'agent']


def sync(device: str):
    """
    Wait for the device to finish its queued work.

    Args:
        device (str): "cpu" or "cuda".
    """
    if device == "cuda":
        torch.cuda.synchronize()


def make_env(name: str, batch_size: int, device: str, check_actions: bool):
    """
    Build an AEC environment with its default components.

    Args:
        name (str): Environment name.
        batch_size (int): Batch size.
        device (str): "cpu" or "cuda".
        check_actions (bool): Value of the environment check_actions argument.

    Returns:
        Environment: The environment.
    """
    module = lambda m: importlib.import_module(f"maenvs4vrp.environments.{name}.{m}")
    return module("env").Environment(
        instance_generator_object=module("instances_generator").InstanceGenerator(device=device),
        obs_builder_object=module("observations").Observations(),
        agent_selector_object=module("env_agent_selector").AgentSelector(),
        reward_evaluator=module("env_agent_reward").DenseReward(),
        batch_size=batch_size, device=device, seed=0, check_actions=check_actions)


def episode(env, num_nodes: int, num_agents: int, device: str, seed: int) -> tuple[float, int]:
    """
    Run one episode with random feasible actions.

    Args:
        env (Environment): Environment.
        num_nodes (int): Number of nodes, depot included.
        num_agents (int): Number of agents.
        device (str): "cpu" or "cuda".
        seed (int): Instance seed.

    Returns:
        tuple[float, int]: Episode time in seconds and number of steps.
    """
    sync(device)
    start = time.perf_counter()
    td = env.reset_agent_select_observe(num_agents=num_agents, num_nodes=num_nodes, seed=seed, obs_list=RESET_OBS)
    steps = 0
    while not td["done"].all():
        td["next_action"] = torch.multinomial(td["observations"]["action_mask"].float(), 1)
        td = env.step_agent_select_observe(td, obs_list=STEP_OBS)
        steps += 1
    sync(device)
    return time.perf_counter() - start, steps


def rl4co_episode(env, batch_size: int, device: str) -> tuple[float, int]:
    """
    Run one rl4co episode with random feasible actions.

    Args:
        env (RL4COEnvBase): rl4co environment.
        batch_size (int): Batch size.
        device (str): "cpu" or "cuda".

    Returns:
        tuple[float, int]: Episode time in seconds and number of steps.
    """
    sync(device)
    start = time.perf_counter()
    # rl4co generates on the CPU even when the environment is built with device="cuda"
    td = env.reset(env.generator(batch_size=[batch_size]).to(device))
    steps = 0
    while not td["done"].all():
        td["action"] = torch.multinomial(td["action_mask"].float(), 1).squeeze(-1)
        td = env.step(td)["next"]
        steps += 1
    sync(device)
    return time.perf_counter() - start, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", default="cvrp,cvrptw", help="comma separated AEC environment names")
    parser.add_argument("--customers", default="20,50,100", help="comma separated numbers of customers")
    parser.add_argument("--agents", type=int, default=None, help="number of agents (default: customers // 5)")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, default=5, help="measured episodes, after one warm-up episode")
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    parser.add_argument("--check-actions", action="store_true", help="keep the per-step feasibility asserts")
    parser.add_argument("--rl4co", action="store_true", help="also time the rl4co CVRP/CVRPTW environments")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    print(f"{'library':9} {'env':8} {'N':>4} {'A':>3} {'episode s':>10} {'steps':>6} {'ms/step':>8}")
    for name in args.envs.split(","):
        for n in map(int, args.customers.split(",")):
            agents = args.agents or max(2, n // 5)
            env = make_env(name, args.batch, args.device, args.check_actions)
            runs = [episode(env, n + 1, agents, args.device, seed) for seed in range(args.episodes + 1)][1:]
            t, s = statistics.median(r[0] for r in runs), statistics.median(r[1] for r in runs)
            print(f"{'maenvs4vrp':9} {name:8} {n:4} {agents:3} {t:10.3f} {s:6.0f} {1e3 * t / s:8.2f}")
            if args.rl4co and name in ("cvrp", "cvrptw"):
                from rl4co.envs import CVRPEnv, CVRPTWEnv
                r4 = {"cvrp": CVRPEnv, "cvrptw": CVRPTWEnv}[name](generator_params={"num_loc": n}, check_solution=False)
                runs = [rl4co_episode(r4, args.batch, args.device) for _ in range(args.episodes + 1)][1:]
                t, s = statistics.median(r[0] for r in runs), statistics.median(r[1] for r in runs)
                print(f"{'rl4co':9} {name:8} {n:4} {'-':>3} {t:10.3f} {s:6.0f} {1e3 * t / s:8.2f}")


if __name__ == "__main__":
    main()
