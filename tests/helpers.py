"""Shared helpers for the test suite: environment discovery, construction and rollouts."""
import importlib
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "maenvs4vrp"
REPO_DIR = PKG_DIR.parent


def discover(group: str) -> list[str]:
    """
    List the environment packages of a group.

    Args:
        group (str): "environments" or "parallel_environments".

    Returns:
        list[str]: Sorted environment names.
    """
    return sorted(p.name for p in (PKG_DIR / group).iterdir() if (p / "env.py").exists())


AEC_ENVS = discover("environments")
PARALLEL_ENVS = discover("parallel_environments")
MULTITASK_ENVS = [e for e in AEC_ENVS if e in ("mtvrp", "gmtvrp", "mtdvrp", "gmtdvrp")]
SINGLE_TASK_ENVS = [e for e in AEC_ENVS if e not in MULTITASK_ENVS]
# environments whose instances need an odd number of nodes (depot + pickup/delivery pairs)
ODD_NODES_ENVS = {"pdptw"}


def module(env: str, name: str, group: str = "environments"):
    """
    Import a module of an environment package.

    Args:
        env (str): Environment name.
        name (str): Module name, e.g. "env" or "observations".
        group (str, optional): Environment group. Defaults to "environments".

    Returns:
        module: Imported module.
    """
    return importlib.import_module(f"maenvs4vrp.{group}.{env}.{name}")


def make_env(env: str, group: str = "environments", selector: str | None = "AgentSelector",
             generator=None, observations=None, reward: str = "DenseReward", batch_size: int = 1, seed: int = 0):
    """
    Build an environment with its default components.

    Args:
        env (str): Environment name.
        group (str, optional): Environment group. Defaults to "environments".
        selector (str | None, optional): Agent selector class name, or None for no selector
            (ignored for parallel environments). Defaults to "AgentSelector".
        generator (InstanceBuilder, optional): Instance generator. Defaults to the random InstanceGenerator.
        observations (ObservationBuilder, optional): Observation builder. Defaults to the default Observations.
        reward (str, optional): Reward class name. Defaults to "DenseReward".
        batch_size (int, optional): Batch size. Defaults to 1.
        seed (int, optional): Random number generator seed. Defaults to 0.

    Returns:
        Environment: The environment.
    """
    kwargs = dict(
        instance_generator_object=generator or module(env, "instances_generator", group).InstanceGenerator(),
        obs_builder_object=observations or module(env, "observations", group).Observations(),
        reward_evaluator=getattr(module(env, "env_agent_reward", group), reward)(),
        batch_size=batch_size,
        seed=seed,
    )
    if group == "environments":
        kwargs["agent_selector_object"] = (getattr(module(env, "env_agent_selector", group), selector)()
                                           if selector else None)
    return module(env, "env", group).Environment(**kwargs)


def nodes_for(env: str, num_nodes: int) -> int:
    """
    Adjust a node count to the constraints of an environment.

    Args:
        env (str): Environment name.
        num_nodes (int): Requested number of nodes.

    Returns:
        int: A valid number of nodes for the environment.
    """
    if env in ODD_NODES_ENVS and num_nodes % 2 == 0:
        return num_nodes + 1
    return num_nodes


def rollout(env, mode: str = "select", max_steps: int = 10_000, record: bool = False, **reset_kwargs):
    """
    Run one episode with random actions.

    Args:
        env (Environment): Environment.
        mode (str, optional): Interaction mode: "select" (agent selector), "agent_node" (sample agent then
            node), "joint" (sample both at once), "node_agent" (sample node then agent) or "parallel".
            Defaults to "select".
        max_steps (int, optional): Fail if the episode is not done after this many steps. Defaults to 10000.
        record (bool, optional): If True, also return the (action, reward, penalty) of every step.
            Defaults to False.
        **reset_kwargs: Arguments passed to the reset method.

    Returns:
        TensorDict | tuple: Final environment tensor instance, plus the list of step tensors when ``record``.
    """
    steps = []
    if mode == "select":
        td = env.reset_agent_select_observe(**reset_kwargs)
    elif mode == "parallel":
        td = env.reset_observe(**reset_kwargs)
    elif mode == "agent_node":
        td = env.reset(**reset_kwargs)
    else:
        td = env.reset_observe(**reset_kwargs)
    n = 0
    while not td["done"].all():
        assert n < max_steps, f"episode not done after {max_steps} steps"
        if mode == "select":
            td = env.sample_action(td)
            td = env.step_agent_select_observe(td)
        elif mode == "parallel":
            td = env.sample_actions_all(td)
            td = env.step_all_observe(td)
        elif mode == "agent_node":
            td = env.sample_agent(td)
            td = env.sample_action(td)
            td = env.step_observe(td)
        elif mode == "joint":
            td = env.sample_joint(td)
            td = env.step_observe(td)
        elif mode == "node_agent":
            td = env.sample_action(td, action_without_agent=True)
            td = env.sample_agent(td, agent_given_action=True)
            td = env.step_observe(td)
        else:
            raise ValueError(mode)
        if record:
            steps.append((td["next_action"].clone() if "next_action" in td.keys() else None,
                          td["reward"].clone(), td["penalty"].clone()))
        n += 1
    return (td, steps) if record else td
