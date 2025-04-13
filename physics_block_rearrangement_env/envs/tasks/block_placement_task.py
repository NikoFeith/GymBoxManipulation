# physics_block_rearrangement_env/envs/tasks/block_placement_task.py
import numpy as np
import pybullet as p
import math
from ..task_interface import BaseTask
from physics_block_rearrangement_env.utils.logging_utils import *

log_level = logging.ERROR # Or logging.DEBUG
logger = setup_logger(__name__, level=log_level)



class BlockPlacementTask(BaseTask):
    """
    Task: Place N blocks, initially spawned randomly, onto N target locations
          arranged according to a specified pattern ('line_y', 'circle', 'random_scatter').
          Goal is block i on target i.
    """
    def _load_task_params(self):
        """Load parameters for the placement task."""
        self.num_blocks = self.config.get("num_blocks", 2)
        self.target_spacing = self.config.get("target_spacing", 0.15)
        self.line_x_offset = self.config.get("line_x_offset", 0.20)
        self.circle_radius = self.config.get("circle_radius", 0.18)
        self.circle_center_offset = self.config.get("circle_center_offset", [0.15, 0.0])
        self.target_scatter_bounds = self.config.get("target_scatter_bounds", [0.05, 0.25, -0.2, 0.2])
        self.goal_dist_threshold = self.config.get("goal_dist_threshold", 0.04)

        self.spawn_bounds_relative = self.config.get("spawn_bounds_relative", [0.0, 0.25, -0.20, -0.05]) # [minX, maxX, minY, maxY] relative to table center

        self.num_locations = self.num_blocks
        self.num_dump_locations = self.config.get("num_dump_locations", 1)

    def define_spawn_area(self) -> list[float]:
        """Defines the block spawn area based on task config."""
        # Get table center X from the environment instance
        table_center_x = self.env.table_start_pos[0]
        table_center_y = self.env.table_start_pos[1] # Usually 0

        # Calculate absolute bounds based on relative config and table center
        min_x = table_center_x + self.spawn_bounds_relative[0]
        max_x = table_center_x + self.spawn_bounds_relative[1]
        min_y = table_center_y + self.spawn_bounds_relative[2]
        max_y = table_center_y + self.spawn_bounds_relative[3]

        spawn_bounds = [min_x, max_x, min_y, max_y]
        logger.info(f"  Task defined spawn area: {np.round(spawn_bounds, 2)}")
        return spawn_bounds

    def reset_task_scenario(self):
        """
        Sets up the task scenario:
        - Defines target and dump locations.
        - Spawns target visuals.
        - Spawns blocks.
        - Defines the goal configuration.
        Returns a task_info dict (can be empty).
        """
        env = self.env  # shorthand

        # --- 1. Define target positions ---
        layout = env.target_layout  # Must be loaded from config earlier
        env.target_locations_pos = env._generate_target_positions(layout)

        # --- 2. Define dump positions ---
        dump_base = env.config.get("task", {}).get("dump_base_pos", [0.05, 0.0])
        if len(dump_base) < 2:
            dump_base = [0.05, 0.0]

        # Compute Z position: flush with table top, add small offset for clearance
        base = np.array(env.table_start_pos[:2])
        dump_z = env.table_height + 0.005

        # Create multiple dump positions along vertical axis
        env.dump_location_pos = [
            [base[0] + dump_base[0], base[1] + dump_base[1] - i * 0.08, dump_z]
            for i in range(env.num_dump_locations)
        ]

        # --- 3. Define goal configuration ---
        mapping_type = env.config.get("mapping_type", "random")
        env.goal_config = self._generate_goal_config(mapping_type)
        logger.debug(f'Goal config updated with: {env.goal_config}')

        # --- 4. Spawn target visuals ---
        env._place_target_visuals()  # You must implement this if not already there

        # --- 5. Spawn blocks ---
        env._spawn_blocks_random_xy()  # Existing method

        return {"goal_mapping": env.goal_config}

    def check_goal(self) -> bool:
        env = self.env
        if env.held_object_id is not None: return False
        if not env.goal_config: return False
        on_target_count = 0
        for target_loc_idx, required_block_idx in env.goal_config.items():
            if required_block_idx >= len(env.block_ids) or target_loc_idx >= len(env.target_locations_pos): continue
            block_id = env.block_ids[required_block_idx]
            target_pos = env.target_locations_pos[target_loc_idx]
            try:
                current_pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=env.client)
                dist_xy = np.linalg.norm(np.array(current_pos[:2]) - np.array(target_pos[:2]))
                on_surface = abs(current_pos[2] - (env.table_height + env.block_half_extents[2])) < 0.02
                if dist_xy < self.goal_dist_threshold and on_surface:
                    on_target_count += 1
            except Exception: return False
        return on_target_count == len(env.goal_config)

    def _generate_goal_config(self, mapping_type: str = "random") -> dict:
        """
        Generate a mapping from block indices to target indices.

        Args:
            mapping_type (str): Either 'ordered' or 'random'.
                - 'ordered' maps block i → target i.
                - 'random' shuffles target assignments.

        Returns:
            dict: goal_config where keys are block indices and values are target indices.
        """
        num_blocks = self.env.num_blocks
        block_indices = list(range(num_blocks))

        if mapping_type == "random":
            target_indices = block_indices.copy()
            self.env.np_random.shuffle(target_indices)
        else:  # fallback to ordered
            target_indices = block_indices

        return {t: b for b, t in zip(block_indices, target_indices)}
