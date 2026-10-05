"""Reward functions for the MDVRPTW environment."""
import torch
from tensordict import TensorDict
from maenvs4vrp.core.env_agent_reward import RewardFn

from typing import Optional, List


class DenseReward(RewardFn):
    """
    MDVRPTW dense reward class.
    """

    def __init__(self):
        """
        Initialize the reward function.
        """
        self.env = None
        self.pending_penalty = -2

    def set_env(self, env):
        """
        Set environment.

        Args:
            env (AECEnv): Environment.
        """
        self.env = env

    def get_reward(self, action):
        """
        Get reward and penalty.

        Args:
            action (torch.Tensor): [B, A] tensor with all agents' moves.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Reward and penalty, with shape [B, 1] each.
        """

        reward = -self.env.td_state['cur_agent']['cur_travel_time'].clone()
        penalty = torch.zeros_like(action, dtype = torch.float, device=self.env.device)

        # compute penalty if env has unvisited nodes
        is_last_step = self.env.td_state['is_last_step']

        depot2nodes = torch.cdist(self.env.td_state['depot_loc'], self.env.td_state['coords'])
        if self.env.n_digits is not None:
            depot2nodes = torch.floor(self.env.n_digits * depot2nodes) / self.env.n_digits

        penalty[is_last_step] = self.pending_penalty * ((depot2nodes.sum(1).scatter_(1, self.env.td_state['depot_idx'], 0) * self.env.td_state['nodes']['active_nodes_mask']).sum(-1, keepdim = True).float()[is_last_step])

        return reward, penalty



class SparseReward(RewardFn):
    """
    MDVRPTW sparse reward class.
    """

    def __init__(self):
        """
        Initialize the reward function.
        """
        self.env = None
        self.pending_penalty = -2

    def set_env(self, env):
        """
        Set environment.

        Args:
            env (Environment): Environment.
        """
        self.env = env

    def get_reward(self, action):
        """
        Get reward and penalty.

        Args:
            action (torch.Tensor): Tensor with agent moves.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Reward and penalty, with shape [B, 1] each.
        """

        reward = torch.zeros_like(action, dtype = torch.float, device=self.env.device)
        penalty = torch.zeros_like(action, dtype = torch.float, device=self.env.device)

        # compute penalty if env has unvisited nodes
        is_last_step = self.env.td_state['is_last_step']

        depot2nodes = torch.cdist(self.env.td_state['depot_loc'], self.env.td_state['coords'])
        if self.env.n_digits is not None:
            depot2nodes = torch.floor(self.env.n_digits * depot2nodes) / self.env.n_digits

        final_reward = -self.env.td_state['agents']['cum_travel_time'].sum(1, keepdim = True)
        penalty[is_last_step] = self.pending_penalty * ((depot2nodes.sum(1).scatter_(1, self.env.td_state['depot_idx'], 0) * self.env.td_state['nodes']['active_nodes_mask']).sum(-1, keepdim = True).float()[is_last_step])

        reward[is_last_step] = final_reward[is_last_step]
        return reward, penalty
