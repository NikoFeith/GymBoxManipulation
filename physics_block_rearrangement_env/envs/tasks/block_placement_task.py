# physics_block_rearrangement_env/envs/tasks/block_placement_task.py
import numpy as np
import pybullet as p
import math
from ..task_interface import BaseTask


class BlockPlacementTask(BaseTask):
    """
    Task: Place N blocks, initially spawned randomly, onto N target locations
          arranged according to a specified pattern ('line_y', 'circle', 'random_scatter').
          Goal is block i on target i.
    """
    def _load_task_params(self):
        """Load parameters for the placement task."""
        self.num_blocks = self.config.get("num_blocks", 2)
        # Get the target pattern type for this task instance
        self.target_pattern = self.config.get("target_pattern", "line_y")
        # Load parameters potentially used by different patterns
        self.target_spacing = self.config.get("target_spacing", 0.15) # For line
        self.line_x_offset = self.config.get("line_x_offset", 0.20)  # For line
        self.circle_radius = self.config.get("circle_radius", 0.18) # For circle
        self.circle_center_offset = self.config.get("circle_center_offset", [0.15, 0.0]) # For circle [x,y]
        # Bounds for random scatter relative to table center [minX, maxX, minY, maxY]
        self.target_scatter_bounds = self.config.get("target_scatter_bounds", [0.05, 0.25, -0.2, 0.2])
        # Goal checking parameter
        self.goal_dist_threshold = self.config.get("goal_dist_threshold", 0.04)

        # Set num_locations based on num_blocks for this task type
        self.num_locations = self.num_blocks
        # Get num_dump locations from task config or default
        self.num_dump_locations = self.config.get("num_dump_locations", 1) # Allow overriding dump num

        print(f"  BlockPlacementTask: Loaded params - num_blocks={self.num_blocks}, pattern='{self.target_pattern}'")

    def reset_task_scenario(self):
        """Set up the placement task scenario based on the chosen pattern."""
        env = self.env # Shortcut

        # 1. Define Target Locations based on self.target_pattern
        target_locs = []
        center_x = env.table_start_pos[0]
        center_y = env.table_start_pos[1] # Usually 0
        z_pos = env.table_height # Base Z is table height
        min_target_dist_sq = (env.block_scale * 2.0)**2 # Use env's block_scale

        print(f"  Defining {self.num_locations} target locations with pattern: {self.target_pattern}")

        if self.target_pattern == 'line_y':
            line_x = center_x + self.line_x_offset
            y_start = center_y - self.target_spacing * (self.num_locations - 1) / 2.0
            for i in range(self.num_locations):
                target_locs.append([line_x, y_start + i * self.target_spacing, z_pos])

        elif self.target_pattern == 'circle':
            radius = self.circle_radius
            circle_center_x = center_x + self.circle_center_offset[0]
            circle_center_y = center_y + self.circle_center_offset[1]
            angle_offset = env.np_random.uniform(0, math.pi / self.num_locations) if self.num_locations > 0 else 0
            for i in range(self.num_locations):
                angle = angle_offset + 2 * math.pi * i / self.num_locations
                x = circle_center_x + radius * math.cos(angle)
                y = circle_center_y + radius * math.sin(angle)
                # Optional: Add bounds check
                target_locs.append([x, y, z_pos])

        elif self.target_pattern == 'random_scatter':
             bounds = [ # Calculate absolute bounds
                 center_x + self.target_scatter_bounds[0], center_x + self.target_scatter_bounds[1],
                 center_y + self.target_scatter_bounds[2], center_y + self.target_scatter_bounds[3]
             ]
             print(f"    Scattering targets within: {np.round(bounds, 2)}")
             attempts = 0; max_attempts = self.num_locations * 50
             while len(target_locs) < self.num_locations and attempts < max_attempts:
                attempts += 1
                x = env.np_random.uniform(bounds[0], bounds[1])
                y = env.np_random.uniform(bounds[2], bounds[3])
                too_close = any(((loc[0]-x)**2 + (loc[1]-y)**2 < min_target_dist_sq) for loc in target_locs)
                if not too_close: target_locs.append([x, y, z_pos])
             if len(target_locs) < self.num_locations: print(f"Warning: Only placed {len(target_locs)}/{self.num_locations} random targets.")

        else: # Fallback or error for unknown pattern
            print(f"Warning: Unknown target pattern '{self.target_pattern}'. Using default line_y.")
            # Fallback to line_y logic...
            target_spacing = 0.15; line_x = center_x + 0.20
            y_start = center_y - target_spacing * (self.num_locations - 1) / 2.0
            for i in range(self.num_locations): target_locs.append([line_x, y_start + i * target_spacing, z_pos])

        env.target_locations_pos = target_locs # Set target locations in main env

        # 2. Define Goal Config (Simple: Block i must go to Target i)
        # Locations are indexed 0..M-1 in env.target_locations_pos
        # Blocks are indexed 0..N-1 in env.block_ids (N=M for this task)
        # goal_config maps target_location_index -> required_block_index
        env.goal_config = {i: i for i in range(self.num_locations)} # Block i -> Target i

        # 3. Trigger Spawning of Blocks and Target Visuals in Env
        # Ensure color lists are ready based on self.num_blocks in main env
        env._load_colors() # Reload/slice colors based on the task's num_blocks
        env._place_target_visuals()
        env._spawn_blocks(self.num_blocks)

        print(f"  Task Scenario Reset: {self.config.get('name', 'BlockPlacementTask')} - {self.num_blocks} blocks, pattern '{self.target_pattern}'.")
        return {"task_type": "BlockPlacement", "pattern": self.target_pattern}

    def check_goal(self) -> bool:
        """Check if Block i is on Target i for all i."""
        env = self.env
        if env.held_object_id is not None: return False

        # Need to check against the goal config defined in reset
        if not env.goal_config: return False # No goal defined

        on_target_count = 0
        for target_loc_idx, required_block_idx in env.goal_config.items():
            # Ensure indices are valid
            if required_block_idx >= len(env.block_ids) or target_loc_idx >= len(env.target_locations_pos):
                continue

            block_id = env.block_ids[required_block_idx]
            target_pos = env.target_locations_pos[target_loc_idx]

            try:
                current_pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=env.client)
                dist_xy = np.linalg.norm(np.array(current_pos[:2]) - np.array(target_pos[:2]))
                on_surface = abs(current_pos[2] - (env.table_height + env.block_half_extents[2])) < 0.02
                if dist_xy < self.goal_dist_threshold and on_surface:
                    on_target_count += 1
            except Exception: return False # Error reading state

        # Goal met if all required blocks are on their assigned targets
        return on_target_count == len(env.goal_config)