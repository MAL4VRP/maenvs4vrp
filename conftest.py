"""Pytest configuration: command line options for the solution validity tests."""
from dataclasses import dataclass

import pytest
import torch

DEFAULT_NUM_AGENTS = [2, 5]
DEFAULT_NUM_NODES = [11, 21]


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu",
                     help='Device used by the environments: "cpu" or "cuda" ("gpu" is accepted as an alias).')
    parser.addoption("--batch", "--batch_size", action="store", default=None,
                     help="Batch size used by the solution validity tests.")
    parser.addoption("--agents", "--num_agents", action="append", default=None,
                     help="Number of agents. Repeat the option to test several values.")
    parser.addoption("--nodes", "--num_nodes", action="append", default=None,
                     help="Number of nodes. Repeat the option to test several values.")


@dataclass
class Sizes:
    """Problem sizes used by the solution validity tests."""
    device: torch.device
    batch_size: int
    num_agents: list
    num_nodes: list


@pytest.fixture(scope="session")
def sizes(request) -> Sizes:
    """
    Problem sizes from the command line, with small defaults.

    Returns:
        Sizes: Device, batch size, numbers of agents and numbers of nodes.
    """
    device = request.config.getoption("--device")
    batch = request.config.getoption("--batch")
    agents = request.config.getoption("--agents")
    nodes = request.config.getoption("--nodes")
    return Sizes(device=torch.device("cuda" if device in ("cuda", "gpu") else "cpu"),
                 batch_size=int(batch) if batch is not None else 2,
                 num_agents=[int(a) for a in agents] if agents else DEFAULT_NUM_AGENTS,
                 num_nodes=[int(n) for n in nodes] if nodes else DEFAULT_NUM_NODES)
