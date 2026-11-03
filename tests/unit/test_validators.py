"""
The solution validators reject each kind of constraint violation, without printing, and random episodes serve
every customer when the fleet is large enough.

Each mutation test takes a valid episode, breaks exactly one constraint and checks the validator's error message,
so that a test cannot pass because a different, unrelated check fired.
"""
import pytest
import torch

from tests.helpers import AEC_ENVS, make_env, nodes_for, rollout

# split deliveries legitimately visit a node more than once
REVISITS_ALLOWED = {"sdvrptw"}
# prize collecting problems do not have to serve every customer, and dynamic customers may appear too late
PARTIAL_SERVICE_ENVS = {"pcvrp", "pcvrptw", "top", "toptw", "dvrptw", "dsvrptw"}

CAPACITY_ENVS = [e for e in AEC_ENVS if e not in {"top", "toptw"}]
TIME_WINDOW_ENVS = [e for e in AEC_ENVS if e not in {"cvrp", "hcvrp", "pcvrp", "top"}]
DISTANCE_LIMIT_ENVS = ["mtvrp", "gmtvrp", "mtdvrp", "gmtdvrp"]


def served_episode(env_name: str, num_agents: int = 3):
    """
    Run a valid episode (batch of one) that serves at least one customer.

    Args:
        env_name (str): Environment name.
        num_agents (int, optional): Number of agents. Defaults to 3.

    Returns:
        tuple: The environment and the served customers.
    """
    env = make_env(env_name, batch_size=1)
    for seed in range(20):
        torch.manual_seed(seed)
        rollout(env, "select", num_agents=num_agents, num_nodes=nodes_for(env_name, 11), seed=seed)
        actions = env.td_state["solution"]["actions"][0]
        customers = actions[actions >= getattr(env, "num_depots", 1)]
        if customers.numel():
            env.check_solution_validity()
            return env, customers
    pytest.fail(f"no {env_name} episode served a customer")


def assert_rejected(env, match: str, capsys):
    """
    Check that the validator rejects the solution with the expected message and prints nothing.

    Args:
        env (Environment): Environment with a mutated state or solution.
        match (str): Regular expression the error message must match.
        capsys (pytest.CaptureFixture): Pytest output capture.
    """
    with pytest.raises(AssertionError, match=match):
        env.check_solution_validity()
    assert capsys.readouterr().out == "", "check_solution_validity must not print"


def customers_mask(env) -> torch.Tensor:
    """Boolean mask [N] of the customer nodes."""
    num_nodes = env.td_state["coords"].shape[1]
    return torch.arange(num_nodes) >= getattr(env, "num_depots", 1)


@pytest.mark.parametrize("env_name", [e for e in AEC_ENVS if e not in REVISITS_ALLOWED])
def test_validator_rejects_repeated_visits(env_name, capsys):
    # an unused agent serves an already served customer (and its delivery, for pickups) and returns to its depot:
    # only the repeated visit breaks a constraint
    env, customers = served_episode(env_name, num_agents=6)
    state, solution = env.td_state, env.td_state["solution"]
    agents = solution["agents"]
    num_agents = state["agents"]["cur_node_idx"].shape[1]
    used = set(agents[0][solution["actions"][0] >= getattr(env, "num_depots", 1)].tolist())
    idle = [a for a in range(num_agents) if a not in used]
    assert idle, "the episode has no unused agent"
    agent = idle[-1]
    depot = int(state["agents"]["depot_idx"][0, agent]) if "depot_idx" in state["agents"].keys() else 0
    if "appear_time" in state.keys():  # dynamic customers: pick one known from the start
        customers = customers[state["appear_time"][0, customers] <= 0]
        assert customers.numel(), "no served customer appeared at the start"
    customer = int(customers[0])
    route = [customer]
    if "is_pickup" in state.keys():
        customer = int(customers[state["is_pickup"][0, customers]][0])
        route = [customer, int(state["delivery_idx"][0, customer])]
    route.append(depot)
    num_steps = solution["actions"].shape[-1]
    for key in list(solution.keys()):  # other per-step records (e.g. travel times) get zeros
        if key not in ("actions", "agents") and solution[key].shape[-1] == num_steps:
            solution[key] = torch.cat([solution[key], torch.zeros((1, len(route)), dtype=solution[key].dtype)], dim=-1)
    solution["actions"] = torch.cat([solution["actions"], torch.tensor([route])], dim=-1)
    solution["agents"] = torch.cat([agents, torch.full((1, len(route)), agent)], dim=-1)
    assert_rejected(env, "more than once", capsys)


@pytest.mark.parametrize("env_name", CAPACITY_ENVS)
def test_validator_rejects_capacity_violation(env_name, capsys):
    env, _ = served_episode(env_name)
    env.td_state["capacity"] = env.td_state["capacity"] * 1e-3
    assert_rejected(env, "(?i)capacity", capsys)


@pytest.mark.parametrize("env_name", TIME_WINDOW_ENVS)
def test_validator_rejects_time_window_violation(env_name, capsys):
    # dsvrptw has soft time windows: lateness is allowed, but the recorded penalties no longer match
    env, _ = served_episode(env_name)
    state, customers = env.td_state, customers_mask(env)
    if "time_windows" in state.keys():
        state["time_windows"][..., 1] = torch.where(customers, -1.0, state["time_windows"][..., 1])
    else:
        key = "tw_high_limit" if "tw_high_limit" in state.keys() else "tw_high"
        state[key] = torch.where(customers, -1.0, state[key])
    assert_rejected(env, "(?i)time window", capsys)


@pytest.mark.parametrize("env_name", DISTANCE_LIMIT_ENVS)
def test_validator_rejects_distance_limit_violation(env_name, capsys):
    env, _ = served_episode(env_name)
    env.td_state["distance_limits"] = torch.full_like(env.td_state["distance_limits"], 1e-3)
    assert_rejected(env, "distance limit", capsys)


def test_pdptw_validator_rejects_undelivered_pickup(capsys):
    env, customers = served_episode("pdptw")
    state, solution = env.td_state, env.td_state["solution"]
    pickup = customers[state["is_pickup"][0, customers]][0]
    keep = solution["actions"][0] != state["delivery_idx"][0, pickup]
    solution["actions"] = solution["actions"][:, keep]
    solution["agents"] = solution["agents"][:, keep]
    assert_rejected(env, "without delivering a pickup", capsys)


def test_sdvrptw_validator_rejects_delivery_above_demand(capsys):
    env, customers = served_episode("sdvrptw")
    state = env.td_state
    customer = customers[0]
    state["demands"][0, customer] = state["solution"]["deliveries"][0, state["solution"]["actions"][0] == customer].sum() / 2
    assert_rejected(env, "more than the demand", capsys)


def test_sdvrptw_deliveries_match_served_demand():
    env = make_env("sdvrptw", batch_size=4)
    torch.manual_seed(0)
    rollout(env, "select", num_agents=3, num_nodes=11, seed=0)
    state, solution = env.td_state, env.td_state["solution"]
    delivered = torch.zeros_like(state["demands"]).scatter_add_(1, solution["actions"], solution["deliveries"])
    assert torch.allclose(delivered + state["nodes"]["cur_demands"], state["demands"])


def test_dsvrptw_validator_rejects_customer_before_appearance(capsys):
    env, customers = served_episode("dsvrptw")
    env.td_state["appear_time"][0, customers[0]] = 1e6
    assert_rejected(env, "before it appeared", capsys)


@pytest.mark.parametrize("mode", ["select", "agent_node", "joint", "node_agent"])
def test_pdptw_pickups_are_always_delivered(mode):
    """Agents never return to the depot carrying load, in every interaction mode."""
    env = make_env("pdptw", selector="AgentSelector" if mode == "select" else None, batch_size=8)
    torch.manual_seed(0)
    rollout(env, mode, num_agents=3, num_nodes=21, seed=0)
    env.check_solution_validity()
    state, actions = env.td_state, env.td_state["solution"]["actions"]
    served = torch.zeros_like(state["is_pickup"]).scatter_(1, actions, True)
    picked = served & state["is_pickup"]
    assert torch.equal(picked, picked & served.gather(1, state["delivery_idx"]))


def unservable_pdptw_customers(env) -> torch.Tensor:
    """
    Customers of pickup and delivery pairs that no vehicle can serve, even alone, within the depot time window.

    Args:
        env (Environment): PDPTW environment after reset.

    Returns:
        torch.Tensor: Boolean mask [B, N] of the unservable customers.
    """
    state = env.td_state
    coords, delivery = state["coords"], state["delivery_idx"]
    delivery_coords = coords.gather(1, delivery.unsqueeze(-1).expand(-1, -1, 2))
    depot = coords[:, :1]
    tour = ((coords - depot).norm(dim=-1) + (delivery_coords - coords).norm(dim=-1)
            + (depot - delivery_coords).norm(dim=-1)) / state["speed"]
    service = state["service_time"] + state["service_time"].gather(1, delivery)
    pickups = state["is_pickup"] & (tour + service > state["end_time"].unsqueeze(-1))
    return pickups.scatter(1, torch.where(pickups, delivery, 0), pickups) & customers_mask(env)


@pytest.mark.parametrize("env_name", [pytest.param(e, marks=pytest.mark.xfail(
    strict=True, reason="no waiting action: agents leave the depot at time 0 and cannot reach customers whose "
                        "earliest arrival time is later")) if e == "cvrpstw" else e
    for e in AEC_ENVS if e not in PARTIAL_SERVICE_ENVS])
def test_every_customer_is_served_with_a_large_fleet(env_name):
    num_nodes = nodes_for(env_name, 21)
    env = make_env(env_name, batch_size=4)
    torch.manual_seed(0)
    rollout(env, "select", num_agents=num_nodes, num_nodes=num_nodes, seed=0)
    env.check_solution_validity()
    actions = env.td_state["solution"]["actions"]
    served = torch.zeros(actions.shape[0], env.td_state["coords"].shape[1], dtype=torch.bool).scatter_(1, actions, True)
    expected = customers_mask(env).expand_as(served)
    if env_name == "pdptw":
        expected = expected & ~unservable_pdptw_customers(env)
    unserved = (expected & ~served).nonzero().tolist()
    assert not unserved, f"unserved (instance, node): {unserved}"
