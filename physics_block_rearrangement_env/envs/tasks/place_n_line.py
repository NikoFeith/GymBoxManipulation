# physics_block_rearrangement_env/envs/tasks/place_n_line.py
import numpy as np
import pybullet as p # May need pybullet if doing calculations involving it
import math
from base_task import BaseTask # Import from parent directory's file

class PlaceNLineTask(BaseTask):
    """
    Task: Place N blocks, initially spawned randomly, onto N target locations
          arranged in a vertical line. Goal is block i on target i.
    """
    def _load_task_params(self):
        """Load parameters specific to placing N blocks in a line."""
        self.num_blocks = self.config.get("num_blocks", 2)
        # Get line parameters from config or use defaults
        self.target_spacing = self.config.get("target_spacing", 0.15)
        self.line_x_offset = self.config.get("line_x_offset", 0.20) # Offset from table center X
        self.goal_dist_threshold = self.config.get("goal_dist_threshold", 0.04)
        print(f"  PlaceNLineTask: Loaded params - num_blocks={self.num_blocks}, spacing={self.target_spacing}")

    def reset_task_scenario(self):
        """Set up the line placement task."""
        env = self.env # Shortcut

        # 1. Define Target Locations for a line
        target_locs = []
        center_x = env.table_start_pos[0]
        center_y = env.table_start_pos[1] # Usually 0
        z_pos = env.table_height # Base Z is table height
        line_x = center_x + self.line_x_offset
        y_start = center_y - self.target_spacing * (self.num_blocks - 1) / 2.0 # Center the line
        for i in range(self.num_blocks):
            target_locs.append([line_x, y_start + i * self.target_spacing, z_pos])
        env.target_locations_pos = target_locs # Set target locations in main env

        # 2. Define Goal Config (Simple: Block i must go to Target i)
        # Target locations are indexed 0 to N-1 in env.target_locations_pos
        # Blocks are indexed 0 to N-1 in env.block_ids
        # goal_config maps target_location_index -> required_block_index
        env.goal_config = {i: i for i in range(self.num_blocks)}

        # 3. Trigger Spawning of Blocks and Target Visuals in Env
        # These helpers need to exist in the main Env class
        env._place_target_visuals() # Tell env to draw targets based on env.target_locations_pos
        env._spawn_blocks(self.num_blocks) # Tell env to spawn blocks

        print(f"  Task Scenario Reset: Place {self.num_blocks} blocks in a line.")
        return {"task_type": "PlaceNLine"} # Example task-specific info

    def check_goal(self) -> bool:
        """Check if Block i is on Target i for all i."""
        env = self.env
        # Cannot be goal state while holding an object
        if env.held_object_id is not None:
            return False

        on_target_count = 0
        # Check each target location defined by the goal config
        for target_loc_idx, required_block_idx in env.goal_config.items():
            # Ensure indices are valid for the current state
            if required_block_idx >= len(env.block_ids) or target_loc_idx >= len(env.target_locations_pos):
                # This case should ideally not happen if reset is correct
                print(f"Warning: Mismatch in goal check indices. TargetIdx:{target_loc_idx}, BlockIdx:{required_block_idx}")
                continue

            block_id = env.block_ids[required_block_idx]
            target_pos = env.target_locations_pos[target_loc_idx] # Get the [x,y,z] for this target index

            try:
                current_pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=env.client)
                # Check XY distance between block center and target base position
                dist_xy = np.linalg.norm(np.array(current_pos[:2]) - np.array(target_pos[:2]))
                # Check if block is resting near the surface (Z pos check)
                on_surface = abs(current_pos[2] - (env.table_height + env.block_half_extents[2])) < 0.02

                if dist_xy < self.goal_dist_threshold and on_surface:
                    on_target_count += 1
            except Exception as e:
                 # Block might have been removed or invalid ID
                 print(f"Error checking goal for block {block_id} at target {target_loc_idx}: {e}")
                 return False # Cannot reach goal if expected block is missing

        # Goal met if all required blocks are on their respective targets
        return on_target_count == len(env.goal_config)