"""Toy instance generator for the GMTVRP environment."""
import torch
from tensordict import TensorDict

import os
import pickle

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

VARIANT_PROBS_PRESETS = { #Variant Probabilities
        "all": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
        "single_feat": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
        "single_feat_otw": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5, "OTW": 0.5},
        "cvrp": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 0.0},
        "ovrp": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 0.0},
        "vrpb": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0},
        "vrpl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 0.0},
        "vrptw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 0.0},
        "ovrptw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 0.0},
        "ovrpb": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0},
        "ovrpl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 0.0},
        "vrpbl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0},
        "vrpbtw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0},
        "vrpltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 0.0},
        "ovrpbl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0},
        "ovrpbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0},
        "ovrpltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 0.0},
        "vrpbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0},
        "ovrpbltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0},
}

class ToyInstanceGenerator(InstanceBuilder):

    """
    GMTVRP toy instance generation class.
    """

    def __init__(
        self,
        device: Optional[str] = "cpu",
        batch_size: Optional[torch.Size] = None,
        seed: int = None
    ) -> None:

        """
        Constructor. Toy instance generator for testing.

        Args:
            device (str, optional): Device for tensor operations, e.g. "cpu" or "cuda". Defaults to "cpu".
            batch_size (torch.Size, optional): Batch size. Defaults to None.
            seed (int, optional): Random number generator seed. Defaults to None.
        """

        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        self.device = device

        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)

        # defaults used when the environment passes None for a parameter
        self.num_agents = 2
        self.num_nodes = 15
        self.capacity = 50
        self.service_time = 0.2
        self.subsample = True
        self.variant_preset = None
        self.use_combinations = False
        self.force_visit = True

    def subsample_variant(
        self,
        prob_open_routes: float = 0.5,
        prob_time_windows: float = 0.5,
        prob_limit: float = 0.5,
        prob_backhaul: float = 0.5,
        td: TensorDict = None,
        variant_preset = None,
    ) -> torch.Tensor:

        """
        Subsample variant. If variant_preset is specified, it loads that variant. Otherwise it samples variant's parameters across batches based on probabilities.

        Args:
            prob_open_routes (float, optional): Probability of open routes. Defaults to 0.5.
            prob_time_windows (float, optional): Probability of time windows. Defaults to 0.5.
            prob_limit (float, optional): Probability of distance limits. Defaults to 0.5.
            prob_backhaul (float, optional): Probability of backhaul. Defaults to 0.5.
            td (TensorDict, optional): Environment tensor instance. Defaults to None.
            variant_preset (TensorDict, optional): Variant preset to be sampled. Defaults to None.

        Returns:
            torch.Tensor: Environment instance tensor.
        """

        if variant_preset is not None:
            variant_probs = VARIANT_PROBS_PRESETS.get(variant_preset)
            assert variant_probs is not None, f"Variant preset {variant_preset} not found! \
                                                Avaliable presets are {self.VARIANT_PROBS_PRESETS.keys()} with probabilities {self.VARIANT_PROBS_PRESETS.values()}"
            print("Using preset", variant_preset)
        else:
            variant_probs = {
                "O": prob_open_routes,
                "TW": prob_time_windows,
                "L": prob_limit,
                "B": prob_backhaul
            }

        for key, prob in variant_probs.items():
            assert 0 <= prob <= 1, f"Probability {key} must be between 0 and 1"

        self.variant_probs = variant_probs
        self.variant_preset = variant_preset

        variant_probs = torch.Tensor(list(self.variant_probs.values())) #Convert dict into tensor

        if self.use_combinations:
            keep_mask = torch.rand(*self.batch_size, 4) >= variant_probs #O, TW, L, B
        else:
            if self.variant_preset in list(VARIANT_PROBS_PRESETS.keys()) and self.variant_preset not in ("all", "cvrp", "single_feat", "single_feat_otw"):
                cvrp_prob = 0
            else:
                cvrp_prob = 0.5

            if self.variant_preset in ("all", "cvrp", "single_feat", "single_feat_otw"):
                indexes = torch.distributions.Categorical(
                    torch.Tensor(list(self.variant_probs.values()) + [cvrp_prob])[
                        None
                    ].repeat(*self.batch_size, 1)
                ).sample()

                if self.variant_preset == "single_feat_otw":
                    keep_mask = torch.zeros((*self.batch_size, 6), dtype=torch.bool)
                    keep_mask[torch.arange(*self.batch_size), indexes] = True

                    keep_mask[:, :2] |= keep_mask[:, 4:5]
                else:
                    keep_mask = torch.zeros((*self.batch_size, 5), dtype=torch.bool)
                    keep_mask[torch.arange(*self.batch_size), indexes] = True

            else:

                keep_mask = torch.zeros((*self.batch_size, 4), dtype=torch.bool)
                indexes = torch.nonzero(variant_probs).squeeze()
                keep_mask[:, indexes] = True

        td = self._default_open(td, ~keep_mask[:, 0])
        td = self._default_time_windows(td, ~keep_mask[:, 1])
        td = self._default_distance_limit(td, ~keep_mask[:, 2])
        td = self._default_backhaul(td, ~keep_mask[:, 3])

        self.keep_mask = keep_mask

        return td

    def sample_instance(
        self,
        num_agents: int = 2,
        num_nodes: int = 15,
        capacity: int = 50,
        service_time: float = 0.2,
        subsample: bool = True,
        variant_preset=None,
        use_combinations: bool = False,
        force_visit: bool = True,
        batch_size: int = 1,
        seed: int = None,
        device: Optional[str] = "cpu",
        sample_type: str = "random",
        **kwargs
    ) -> TensorDict:

        """
        Sample one instance from instance space.

        Args:
            num_agents (int, optional): Total number of agents. Defaults to 2.
            num_nodes (int, optional): Total number of nodes. Defaults to 15.
            capacity (int, optional): Capacity of each agent. Defaults to 50.
            service_time (float, optional): Service time. Defaults to 0.2.
            subsample (bool, optional): If problem variants are to be sampled. Defaults to True.
            variant_preset (str, optional): Variant preset to be sampled. Defaults to None.
            use_combinations (bool, optional): It considers combinations for which sampling mask the instance is defined. Defaults to False.
            force_visit (bool, optional): If True, agents must visit all feasible nodes before returning to the depot. Defaults to True.
            batch_size (int, optional): Batch size. Defaults to 1.
            seed (int, optional): Random number generator seed. Defaults to None.
            device (str, optional): Device for tensor operations, e.g. "cpu" or "cuda". Defaults to "cpu".
            sample_type (str, optional): Sample type. It can only be "random". Defaults to "random".
            **kwargs: Extra keyword arguments accepted for API compatibility; ignored.

        Returns:
            TensorDict: Instance data.

        Raises:
            ValueError: If ``sample_type`` is not "random".
        """

        if sample_type != "random":
            raise ValueError(f"Unknown sample_type '{sample_type}'. Expected 'random'.")

        if seed is not None:
            self._set_seed(seed)

        if num_agents is not None:
            assert num_agents>0, f"Number of agents must be greater than 0!"
        if num_nodes is not None:
            assert num_nodes>0, f"Number of nodes must be greater than 0!"
        if capacity is not None:
            assert capacity>0, f"Capacity must be greater than 0!"
        if service_time is not None:
            assert service_time>0, f"Service times must be greater than 0!"

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        if device is None:
            device = "cpu"
        self.device = device

        if service_time is not None:
            self.service_time = service_time

        if num_nodes is not None:
            self.num_nodes = num_nodes

        if num_agents is not None:

            self.num_agents = num_agents

        if capacity is not None:
            self.capacity = capacity

        if subsample is not None:
            self.subsample = subsample

        if use_combinations is not None:
            self.use_combinations = use_combinations

        if force_visit is not None:
            self.force_visit = force_visit

        self.variant_preset = variant_preset

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        self.depot_idx = 0

        #Depot generation. All 0.

        instance['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)

        #Coords unfiform generation

        coords = torch.tensor([[[0, 0],
                          [1, 2],
                          [2, 3],
                          [3, 2],
                          [-1, 2],
                          [-2, 3],
                          [-3, 2],
                          [-1, -2],
                          [-2, -3],
                          [-3, -2],
                          [1, -2],
                          [2, -3],
                          [3, -2]]], device=self.device)

        instance['coords'] = coords
        self.coords = coords

        #Capacity

        vehicle_capacity = torch.full((*batch_size, 1), self.capacity, dtype=torch.float32)

        #Demands: linehaul and backhaul

        linehaul_demand = torch.tensor([[[0.], [5.], [6.], [4.], [7.], [3.], [4.], [6.], [5.], [3.], [6.], [0.], [0.]]], device=self.device)
        backhaul_demand = torch.tensor([[[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [5.], [4.]]], device=self.device)

        linehaul_demand[:, self.depot_idx] = 0
        backhaul_demand[:, self.depot_idx] = 0

        #Linehaul, backhaul and capacity assignment

        instance['linehaul_demands'] = linehaul_demand.squeeze(-1)
        instance['backhaul_demands'] = backhaul_demand.squeeze(-1)
        instance['capacity'] = vehicle_capacity

        #Unscaled capacity

        instance['original_capacity'] = torch.full((*batch_size, 1), self.capacity, dtype=torch.float32)

        #Open routes

        instance['open_routes'] = torch.tensor([False])

        #Time windows and service times

        time_windows = torch.tensor([[[0., 15.],
                                [1., 7.],
                                [1., 2.],
                                [1., 9.],
                                [3., 9.],
                                [4., 8.],
                                [5., 9.],
                                [3., 6.],
                                [4., 14.],
                                [5., 9.],
                                [3., 12.],
                                [4., 8.],
                                [5., 9.]]], device=self.device)

        service_time = torch.tensor([[0.,
                                3.,
                                3.,
                                3.,
                                3.,
                                3.,
                                3.,
                                3.,
                                3.,
                                3.,
                                3.,
                                3.,
                                3.]], device=self.device)

        instance['time_windows'] = time_windows
        instance['service_time'] = service_time

        #TW low and high
        instance['tw_low'] = time_windows[:, :, 0]
        instance['tw_high'] = time_windows[:, :, 1]

        #Is depot
        instance['is_depot'] = torch.zeros((*self.batch_size, coords.shape[1]), dtype=torch.bool, device=self.device)
        instance['is_depot'][:, self.depot_idx] = True

        #Start time and end time

        instance['start_time'] = time_windows[:, :, 0].gather(1, torch.zeros((*self.batch_size, 1),
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        instance['end_time'] = time_windows[:, :, 1].gather(1, torch.zeros((*self.batch_size, 1),
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)

        #Distance limits

        distance_limits = torch.tensor([20.], device=self.device)

        instance['distance_limits'] = distance_limits

        instance_info = {'name': 'random_instance',
                         'num_nodes': coords.shape[1],
                         'num_agents': self.num_agents,
                         'data': instance}

        if self.subsample:
            instance_info = self.subsample_variant(td=instance_info, variant_preset=self.variant_preset)
            return self._complete_instance(instance_info)
        else:
            return self._complete_instance(instance_info)

    @staticmethod
    def _default_open(td, remove):
        td['data']['open_routes'][remove] = False
        return td

    @staticmethod
    def _default_time_windows(td, remove):
        default_tw = torch.zeros_like(td['data']['time_windows'])
        default_tw[..., 1] = float('inf')
        td['data']['time_windows'][remove] = default_tw[remove]
        td['data']['service_time'][remove] = torch.zeros_like(td['data']['service_time'][remove])
        return td

    @staticmethod
    def _default_distance_limit(td, remove):
        td['data']['distance_limits'][remove] = float('inf')
        return td

    @staticmethod
    def _default_backhaul(td, remove):
        td['data']['linehaul_demands'][remove] = (
            td['data']['linehaul_demands'][remove] + td['data']['backhaul_demands'][remove]
        )
        td['data']['backhaul_demands'][remove] = 0
        return td

    def _complete_instance(self, instance_info: Dict) -> Dict:
        """
        Add the fields the environment expects that the hand-written toy instance does not define.

        Args:
            instance_info (Dict): Toy instance.

        Returns:
            Dict: Instance with floating point coordinates and times and a unit speed.
        """
        data = instance_info['data']
        for key in ('coords', 'tw_low', 'tw_high', 'start_time', 'end_time', 'time_windows'):
            if key in data.keys() and not torch.is_floating_point(data[key]):
                data[key] = data[key].float()
        if 'speed' not in data.keys():
            data['speed'] = torch.ones((*data.batch_size, 1), dtype=torch.float32, device=data.device)
        # toy instances have a fixed size
        instance_info['num_nodes'] = data['coords'].shape[1]
        # multi-task fields
        if 'backhaul_class' not in data.keys():
            data['backhaul_class'] = torch.ones((*data.batch_size, 1), dtype=torch.int64, device=data.device)
        if 'initial_load' not in data.keys():
            data['initial_load'] = data['capacity'].float().expand(*data.batch_size, instance_info['num_agents']).clone()
        return instance_info
