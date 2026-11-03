"""Benchmark instance generator for the HCVRP environment."""
import os
from os import path
import shutil
import logging
import warnings
import torch
import numpy as np
from tensordict import TensorDict
from typing import Optional, Dict, Set

from huggingface_hub import snapshot_download
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

log = logging.getLogger(__name__)

BENCHMARK_INSTANCES_PATH = "hcvrp/data/benchmark"

HF_REPO_ID = "ai4co/parco"


class BenchmarkInstanceGenerator(InstanceBuilder):

    """
    HCVRP Benchmark Instance Generator class.
    """

    @classmethod
    def get_list_of_instances(cls):
        """
        Get list of possible instances from benchmark files.

        Returns:
            dict: Keys 'instances'; values are lists of instance
                  name strings, or empty lists when data is not available locally.
        """

        try:
            cls.download_and_copy_instances()
        except Exception as e:  # offline or Hugging Face unreachable: list what is available locally
            warnings.warn(f"Benchmark instances could not be downloaded ({type(e).__name__}); "
                          "only the instances available locally are listed.", RuntimeWarning)

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        full_dir = path.join(base_dir, BENCHMARK_INSTANCES_PATH)

        files = [f for f in os.listdir(full_dir) if f.endswith(".npz")] if path.isdir(full_dir) else []

        return {
            "instances": [
                f"{BENCHMARK_INSTANCES_PATH}/{fname.split('.')[0]}"
                for fname in files
            ]
        }

    @classmethod
    def download_and_copy_instances(cls):
        """
        Download benchmark instances from HuggingFace if they are not locally present.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        target_dir = path.join(base_dir, BENCHMARK_INSTANCES_PATH)


        if os.path.isdir(target_dir):
            return


        local_snapshot = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )


        os.makedirs(target_dir, exist_ok=True)

        for root, _, files in os.walk(local_snapshot):
            if "data{}hcvrp".format(os.sep) not in root:
                continue

            for fname in files:
                if not fname.endswith(".npz"):
                    continue

                src = path.join(root, fname)
                dst = path.join(target_dir, fname)

                if path.exists(dst):
                    log.info(f"Ignorando duplicado: {fname}")
                    continue

                shutil.copy(src, dst)
                log.info(f"Copiado: {fname}")

        log.warning("Download concluído.")


    def __init__(self,
                num_agents: Optional[int] = None,
                num_nodes: Optional[int] = None,
                min_nodes: Optional[float] = None,
                max_nodes: Optional[float] = None,
                min_demand: Optional[int] = None,
                max_demand: Optional[int] = None,
                min_capacity: Optional[float] = None,
                max_capacity: Optional[float] = None,
                min_speed: Optional[float] = None,
                max_speed: Optional[float] = None,
                instance_name: Optional[str] = None,
                list_of_instances: Optional[Set[str]] = None,
                device: str = "cpu",
                batch_size: Optional[int] = 1,
                seed: Optional[int] = None,
            ):
        """
        Initialize the BenchmarkInstanceGenerator.

        Args:
            num_agents (int, optional): Total number of agents. Defaults to None.
            num_nodes (int, optional): Total number of nodes. Defaults to None.
            min_nodes (float, optional): Minimum coordinate value for node locations. Defaults to None.
            max_nodes (float, optional): Maximum coordinate value for node locations. Defaults to None.
            min_demand (int, optional): Minimum customer demand. Defaults to None.
            max_demand (int, optional): Maximum customer demand. Defaults to None.
            min_capacity (float, optional): Minimum vehicle capacity. Defaults to None.
            max_capacity (float, optional): Maximum vehicle capacity. Defaults to None.
            min_speed (float, optional): Minimum vehicle speed. Defaults to None.
            max_speed (float, optional): Maximum vehicle speed. Defaults to None.
            instance_name (str, optional): Instance set name. It can only be "instances" (every benchmark instance); ignored when ``list_of_instances`` is given. Defaults to None.
            list_of_instances (Set[str], optional): List of instances file names. Defaults to None.
            device (str, optional): Device for tensor operations, e.g. "cpu" or "cuda". Defaults to "cpu".
            batch_size (int, optional): Batch size. Defaults to 1.
            seed (int, optional): Random number generator seed. Defaults to None.
        """


        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        if num_agents is None:
            self.num_agents = 3
        else:
            self.num_agents = num_agents

        if num_nodes is None:
            self.num_nodes = 40
        else:
            self.num_nodes = num_nodes

        if min_nodes is None:
            self.min_nodes = 0.0
        else:
            self.min_nodes = min_nodes
        if max_nodes is None:
            self.max_nodes = 1.0
        else:
            self.max_nodes = max_nodes

        if min_demand is None:
            self.min_demand = 1
        else:
            self.min_demand = min_demand
        if max_demand is None:
            self.max_demand = 10
        else:
            self.max_demand = max_demand

        if min_capacity is None:
            self.min_capacity = 20.0
        else:
            self.min_capacity = min_capacity

        if max_capacity is None:
            self.max_capacity = 50.0
        else:
            self.max_capacity = max_capacity

        if min_speed is None:
            self.min_speed = 0.5
        else:
            self.min_speed = min_speed

        if max_speed is None:
            self.max_speed = 1.5
        else:
            self.max_speed = max_speed

        self.device = device
        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)

        self.instance_name = instance_name

        if list_of_instances is not None:
            self.list_of_instances = list_of_instances
        else:
            self.list_of_instances = self.get_list_of_instances().get(instance_name, [])

        self.load_list_of_instances()

    def load_list_of_instances(self, list_of_instances=None):
        """
        Load every instance on list_of_instances list.

        Args:
            list_of_instances (list, optional): List of instances file names. Defaults to None.
        """

        if list_of_instances:
            self.list_of_instances = list_of_instances

        self.instances_data = dict()

        for instance_name in self.list_of_instances:
            instance = self.read_parse_instance_data(instance_name)
            self.instances_data[instance_name] = instance

    def read_parse_instance_data(self, instance_name: str) -> Dict:
        """
        Read instance data from file. Benchmark's instance keys are translated into our keys.

        Args:
            instance_name (str): Instance path.

        Returns:
            Dict: Instance data.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        file_path = f"{base_dir}/{instance_name}.npz"

        loaded = np.load(file_path)

        depot = loaded["depot"]
        locs = loaded["locs"]
        demand = loaded["demand"]
        capacity = loaded["capacity"]
        speed = loaded["speed"]

        num_instances = locs.shape[0]
        num_nodes = locs.shape[1]
        num_agents = capacity.shape[1]

        locs_all = locs.copy()
        locs_all[:, 0, :] = depot

        batch = num_instances
        data = TensorDict({}, batch_size=[batch], device=self.device)

        data["depot"] = torch.zeros((batch, 1), dtype=torch.long, device=self.device)
        data["coords"] = torch.from_numpy(locs_all).float().to(self.device)
        data["demand"] = torch.from_numpy(demand).float().to(self.device)
        data["capacity"] = torch.from_numpy(capacity).float().to(self.device)
        data["speed"] = torch.from_numpy(speed).float().to(self.device)

        # Depot mask
        is_depot = torch.zeros((batch, num_nodes), dtype=torch.bool, device=self.device)
        is_depot[:, 0] = True
        data["is_depot"] = is_depot

        return {
            "name": instance_name.split("/")[-1],
            "num_nodes": num_nodes,
            "num_agents": num_agents,
            "data": data,
        }

    def get_instance(self, instance_name: str) -> Dict:
        """
        Get an instance with custom number of agents.

        Args:
            instance_name (str): Instance file name.

        Returns:
            Dict: Instance data.
        """

        if not hasattr(self, "instances_data"):
            raise RuntimeError("No instances loaded. Call load_set_of_instances() first.")

        if instance_name not in self.instances_data:
            raise KeyError(f"Instance '{instance_name}' not found.")

        return self.instances_data[instance_name]



    def random_sample_instance(self,
                            batch_size: Optional[torch.Size] = None,
                            seed: Optional[int] = None,
                            device: Optional[str] = "cpu",
                        ) -> Dict:
        """
        Generate a random HCVRP instance with specified parameters.

        Args:
            batch_size (torch.Size, optional): Batch size. Defaults to None.
            seed (int, optional): Random number generator seed. Defaults to None.
            device (str, optional): Device for tensor operations, e.g. "cpu" or "cuda". Defaults to "cpu".

        Returns:
            Dict: Instance data.
        """

        if seed is not None:
            self._set_seed(seed)


        batch = self.batch_size[0]
        data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        # Depot
        depot = torch.rand((batch, 1, 2)) * (self.max_nodes - self.min_nodes) + self.min_nodes
        data["depot"] = torch.zeros((batch, 1), dtype=torch.long, device=self.device)

        # Coords
        coords = torch.rand((batch, self.num_nodes, 2)) * (self.max_nodes - self.min_nodes) + self.min_nodes
        coords[:, 0:1, :] = depot

        data["coords"] = coords

        # Demands
        demand = torch.randint(self.min_demand, self.max_demand, (batch, self.num_nodes)).float()
        demand[:, 0] = 0.0
        data["demand"] = demand

        # Capacity
        capacity = torch.rand((batch, self.num_agents)) * (self.max_capacity - self.min_capacity) + self.min_capacity
        data["capacity"] = capacity

        # Speed
        speed = torch.rand((batch, self.num_agents)) * (self.max_speed - self.min_speed) + self.min_speed
        data["speed"] = speed

        # Depot mask
        is_depot = torch.zeros((batch, self.num_nodes), dtype=torch.bool)
        is_depot[:, 0] = True
        data["is_depot"] = is_depot

        instance = {
            "name": "random_hcvrp_instance",
            "num_nodes": self.num_nodes,
            "num_agents": self.num_agents,
            "data": data.to(self.device),
        }

        return instance

    def sample_name_from_list(self, seed=None):
        """
        Sample one instance from instance list.

        Args:
            seed (int, optional): Random number generator seed. Defaults to None.

        Returns:
            str: Instance name.
        """
        if seed is not None:
            self._set_seed(seed)
        inst = list(self.list_of_instances)
        return inst[torch.randint(0, len(inst), (1,)).item()]

    def sample_instance(
                        self,
                        num_agents: Optional[int] = None,
                        num_nodes: Optional[int] = None,
                        min_nodes: Optional[float] = None,
                        max_nodes: Optional[float] = None,
                        min_demand: Optional[int] = None,
                        max_demand: Optional[int] = None,
                        min_capacity: Optional[float] = None,
                        max_capacity: Optional[float] = None,
                        min_speed: Optional[float] = None,
                        max_speed: Optional[float] = None,
                        instance_name: Optional[str] = None,
                        sample_type: Optional[str] = "random",
                        batch_size: Optional[torch.Size] = None,
                        seed: Optional[int] = None,
                        n_augment: Optional[int] = None,
                        device: Optional[str] = "cpu",
                    ) -> Dict:
        """
        Sample or generate an HCVRP instance.

        Depending on `sample_type`, this method either generates a new random
        instance (delegating to `random_sample_instance`) or returns a saved
        instance previously loaded into the generator's `instances_data`.

        Args:
            num_agents (int, optional): Total number of agents. Defaults to None.
            num_nodes (int, optional): Total number of nodes. Defaults to None.
            min_nodes (float, optional): Minimum coordinate value for locations. Defaults to None.
            max_nodes (float, optional): Maximum coordinate value for locations. Defaults to None.
            min_demand (int, optional): Minimum customer demand (inclusive). Defaults to None.
            max_demand (int, optional): Maximum customer demand (exclusive). Defaults to None.
            min_capacity (float, optional): Minimum vehicle capacity. Defaults to None.
            max_capacity (float, optional): Maximum vehicle capacity. Defaults to None.
            min_speed (float, optional): Minimum vehicle speed. Defaults to None.
            max_speed (float, optional): Maximum vehicle speed. Defaults to None.
            instance_name (str, optional): Instance name. Defaults to None.
            sample_type (str, optional): Sample type. It can be "random" or "saved". Defaults to "random".
            batch_size (torch.Size, optional): Batch size. Defaults to None.
            seed (int, optional): Random number generator seed. Defaults to None.
            n_augment (int, optional): Number of augmented copies of each instance (``batch_size`` must be divisible by it). Defaults to None.
            device (str, optional): Device for tensor operations, e.g. "cpu" or "cuda". Defaults to "cpu".

        Returns:
            Dict: Instance data.

        Raises:
            ValueError: If ``sample_type`` is not "random" or "saved".
        """

        if seed is not None:
            self._set_seed(seed)

        if instance_name is None:
            instance_name = self.sample_name_from_list(seed=seed)
        else:
            instance_name = instance_name


        if num_agents is not None:
            self.num_agents = num_agents

        if num_nodes is not None:
            self.num_nodes = num_nodes


        if min_nodes is not None:
            self.min_nodes = min_nodes

        if max_nodes is not None:
            self.max_nodes = max_nodes

        if min_demand is not None:
            self.min_demand = min_demand

        if max_demand is not None:
            self.max_demand = max_demand

        if min_capacity is not None:
            self.min_capacity = min_capacity

        if max_capacity is not None:
            self.max_capacity = max_capacity

        if min_speed is not None:
            self.min_speed = min_speed

        if max_speed is not None:
            self.max_speed = max_speed

        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)

        if device is not None:
            self.device = device


        if sample_type == "random":
            instance = self.random_sample_instance(batch_size=self.batch_size,
                                                seed=seed,
                                                device=device,
                                            )

        elif sample_type == "saved":
            instance = self.get_instance(instance_name, num_agents=num_agents)

        else:
            raise ValueError(f"Unknown sample_type '{sample_type}'. Expected 'random' or 'saved'.")

        return instance
