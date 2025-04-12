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
        self.target_pattern = self.config.get("target_pattern", "line_y")
        self.target_spacing = self.config.get("target_spacing", 0.15)
        self.line_x_offset = self.config.get("line_x_offset", 0.20)
        self.circle_radius = self.config.get("circle_radius", 0.18)
        self.circle_center_offset = self.config.get("circle_center_offset", [0.15, 0.0])
        self.target_scatter_bounds = self.config.get("target_scatter_bounds", [0.05, 0.25, -0.2, 0.2])
        self.goal_dist_threshold = self.config.get("goal_dist_threshold", 0.04)

        self.spawn_bounds_relative = self.config.get("spawn_bounds_relative", [0.0, 0.25, -0.20, -0.05]) # [minX, maxX, minY, maxY] relative to table center

        self.num_locations = self.num_blocks
        self.num_dump_locations = self.config.get("num_dump_locations", 1)

        print(f"  BlockPlacementTask: Loaded params - num_blocks={self.num_blocks}, pattern='{self.target_pattern}'")

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
        print(f"  Task defined spawn area: {np.round(spawn_bounds, 2)}")
        return spawn_bounds

    def reset_task_scenario(self):
        """Set up the placement task scenario based on the chosen pattern."""
        env = self.env # Shortcut

        # 1. Define Target Locations (using self.target_pattern)
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
        # ... (elif 'circle', elif 'random_scatter', else) ...
        elif self.target_pattern == 'circle':
            radius = self.circle_radius
            circle_center_x = center_x + self.circle_center_offset[0]
            circle_center_y = center_y + self.circle_center_offset[1]
            angle_offset = env.np_random.uniform(0, math.pi / self.num_locations) if self.num_locations > 0 else 0
            for i in range(self.num_locations):
                angle = angle_offset + 2 * math.pi * i / self.num_locations
                x = circle_center_x + radius * math.cos(angle)
                y = circle_center_y + radius * math.sin(angle)
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
            target_spacing = 0.15; line_x = center_x + 0.20
            y_start = center_y - target_spacing * (self.num_locations - 1) / 2.0
            for i in range(self.num_locations): target_locs.append([line_x, y_start + i * target_spacing, z_pos])


        env.target_locations_pos = target_locs # Set target locations in main env

        # --- Define Dump Location (if needed) ---
        # Although BlockPlacement doesn't use dump, we should still define the pos
        # if num_dump_locations > 0, based on env's _dump_location_base_pos
        env.dump_location_pos = []
        if self.num_dump_locations > 0:
             # Use the base position stored in the env, potentially generating multiple
             dump_base_x = env._dump_location_base_pos[0]
             dump_base_y = env._dump_location_base_pos[1]
             for i in range(self.num_dump_locations):
                 # Simple fixed position for the first dump location, stagger others
                 env.dump_location_pos.append([dump_base_x, dump_base_y + i*0.1, z_pos])


        # 2. Define Goal Config
        env.goal_config = {i: i for i in range(self.num_locations)}

        # 3. Trigger Spawning of Blocks and Target Visuals in Env
        env._load_colors()
        env._place_target_visuals()
        env._spawn_blocks(self.num_blocks) # This now uses env.spawn_area_bounds set in env.__init__

        print(f"  Task Scenario Reset: {self.config.get('name', 'BlockPlacementTask')} - {self.num_blocks} blocks, pattern '{self.target_pattern}'.")
        return {"task_type": "BlockPlacement", "pattern": self.target_pattern}

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