"""Base class for reward functions."""

class RewardFn:
    """Agent rewards base class.
    """

    def __init__(self):
        """
        Initialize the reward function.
        """
        self.env = None

    def set_env(self, env):
        """
        Set Environment.

        Args:
            env (AECEnv): Environment.
        """
        self.env = env

    def get_reward(self):
        """
        Get Reward.
        """

        raise NotImplementedError()
