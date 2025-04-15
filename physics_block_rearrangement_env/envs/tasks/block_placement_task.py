# physics_block_rearrangement_env/envs/tasks/block_placement_task.py
import numpy as np
import pybullet as p
import math
import random
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
        self.num_targets = self.config.get("num_targets", 2)
        self.num_locations = self.num_blocks
        self.goal_dist_threshold = self.config.get("goal_dist_threshold", 0.04)
        self.spawn_bounds_relative = self.config.get("spawn_bounds_relative", [0.0, 0.25, -0.20, -0.05]) # [minX, maxX, minY, maxY] relative to table center


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

    def _generate_non_overlapping_dump_positions(self, num_dumps=3) -> list:
        target_positions = self.env.target_locations_pos
        spawn_bounds = self.define_spawn_area()
        min_x, max_x, min_y, max_y = spawn_bounds

        dump_candidates = [
            [min_x, min_y],  # bottom-left
            [max_x, min_y],  # bottom-right
            [min_x, max_y],  # top-left
            [max_x, max_y],  # top-right
            [0.0, 0.0],  # table center as backup
        ]

        final_dumps = []
        dump_z = self.env.table_height + 0.005
        min_dist = 0.1  # minimum distance from target to consider it "clear"

        random.shuffle(dump_candidates)

        for pos in dump_candidates:
            pos_3d = [pos[0], pos[1], dump_z]
            too_close = any(
                np.linalg.norm(np.array(pos[:2]) - np.array(t[:2])) < min_dist
                for t in target_positions
            )
            if not too_close:
                final_dumps.append(pos_3d)
            if len(final_dumps) >= num_dumps:
                break

        # Fallback: if not enough space found, fill remaining with default offset positions
        while len(final_dumps) < num_dumps:
            offset = len(final_dumps) * 0.1
            final_dumps.append([0.0 + offset, 0.0 + offset, dump_z])

        return final_dumps

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
        env.dump_location_pos = self._generate_non_overlapping_dump_positions(num_dumps=env.num_dump_locations)

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
                - 'ordered' maps target i → block i.
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
