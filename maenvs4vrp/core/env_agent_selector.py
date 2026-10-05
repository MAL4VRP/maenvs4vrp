"""Base class for agent selectors."""

class BaseSelector():
    """ Agent iterator base class.
    """

    def __init__(self):
        """
        Initialize the agent selector.
        """


    def set_env(self, env):
        """
        Set environment.

        Args:
            env (AECEnv): Environment.
        """
        self.env = env

    def _next_agent(self):
        """
        Return the next agent.
        """
        raise NotImplementedError()
