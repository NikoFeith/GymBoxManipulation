# physics_block_rearrangement_env/envs/task_interface.py
from abc import ABC, abstractmethod
import numpy as np

class BaseTask(ABC):
    """Abstract Base Class for defining tasks within the rearrangement environment."""

    def __init__(self, env_instance, task_config: dict):
        """
        Initializes the Task.

        Args:
            env_instance: A reference to the main PhysicsBlockRearrangementEnv instance.
            task_config (dict): A dictionary containing parameters specific to this task,
                                loaded from its configuration file.
        """
        self.env = env_instance # Reference to the main environment
        self.config = task_config
        # --- Core Task Attributes (to be set in _load_task_params) ---
        self.num_blocks = 0
        self.num_dump_locations = 0 # Task might override base config
        # Add other common task attributes initialized to defaults here if needed
        # --- Load specific parameters ---
        self._load_task_params()

    @abstractmethod
    def _load_task_params(self):
        """Load task-specific parameters from self.config into instance variables."""
        # Example: self.num_blocks = self.config.get("num_blocks", 3)
        # Example: self.target_pattern = self.config.get("target_pattern", "line_y")
        raise NotImplementedError

    @abstractmethod
    def reset_task_scenario(self):
        """
        Set up the specific task scenario in the PyBullet simulation during env.reset().

        This method should:
        1. Define target locations (e.g., calculate coordinates based on a pattern).
           Store the results in self.env.target_locations_pos.
        2. Define the goal configuration (e.g., which block goes to which target).
           Store the result in self.env.goal_config.
        3. Trigger the spawning of blocks and target visuals by calling appropriate
           helper methods on self.env (e.g., self.env._spawn_blocks, self.env._place_target_visuals).
           The number of blocks comes from self.num_blocks loaded from config.

        Returns:
            dict: Initial task-specific info to be added to the env's info dict, if any.
        """
        raise NotImplementedError

    @abstractmethod
    def check_goal(self) -> bool:
        """
        Check if the current state meets this task's specific goal condition.

        Accesses the simulation state via self.env (e.g., self.env.block_ids,
        self.env.target_locations_pos, self.env.goal_config, self.env.client).

        Returns:
            bool: True if the goal is met, False otherwise.
        """
        raise NotImplementedError

    # --- Optional Methods for more advanced tasks ---
    # def compute_reward(self, previous_state_info, current_state_info, action_info) -> float:
    #     """Calculate task-specific rewards beyond simple step penalty/goal reward."""
    #     # Default could just return self.env.step_penalty or self.env.goal_reward
    #     pass

    # def get_task_observation(self) -> np.ndarray | dict:
    #     """Provide task-specific elements to add to the environment observation."""
    #     # Could return goal information, relative poses etc.
    #     pass

    # def step_curriculum(self, episode_success_rate: float):
    #     """Adjust task difficulty based on performance (e.g., increase num_blocks)."""
    #     pass