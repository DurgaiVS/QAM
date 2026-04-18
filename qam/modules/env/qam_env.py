import gymnasium as gym

from ...constants import DATA_DIR, MAX_SEQ_LEN, SAMPLE_DIM
from ...utils import QAMAction


class QAMEnv(gym.Env):
    def __init__(self, env_config):
        super(QAMEnv, self).__init__()

        self._holding_state = False
        self.observation_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(MAX_SEQ_LEN, SAMPLE_DIM),
        )
        # Initialize your environment here using env_config
        self.action_space = gym.spaces.Discrete(len(QAMAction))

    def reset(self):
        # Reset the state of the environment to an initial state
        self._holding_state = False

    def step(self, action):
        # Execute one time step within the environment
        pass

    def render(self, mode="human"):
        # Render the environment to the screen
        pass

    def close(self):
        # Clean up resources when closing the environment
        pass
