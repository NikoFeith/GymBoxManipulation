import logging

from ..task_interface import BaseTask
import numpy as np
import random

from physics_block_rearrangement_env.utils.logging_utils import *

DEFAULT_INIT_LOG_LEVEL = logging.ERROR
logger = setup_logger(__name__, level=DEFAULT_INIT_LOG_LEVEL)


class GridFieldMovementTask(BaseTask):
    def __init__(self, env_instance, task_config: dict):
        super().__init__(env_instance, task_config)
        self.fields = {}
        self.grid_size = self.config.get("grid_size", [3, 3])
        self.spacing = self.config.get("field_spacing", 0.08)
        self.allow_occupied = self.config.get("allow_occupied", False)
        self.max_occupied = self.config.get("max_occupied", 1)
        self.base_xy = np.array(self.env.table_start_pos[:2])
        self.grid_rotation_rad = 0
        self.z = self.env.table_height + self.env.block_half_extents[2] + 0.01
        self.target_id_to_field = {}

    def _load_task_params(self):
        """Load parameters specific to the grid-based task."""
        self.num_blocks = self.config.get("num_blocks", 3)
        self.num_targets = self.config.get("num_targets", 3)

    def setup_field_grid(self):
        """Create a rotated and translated grid of fields."""
        self.fields.clear()

        # Grid layout
        spacing = self.config.get("field_spacing", 0.08)
        block_size = self.env.block_scale  # Assuming square blocks
        stride = block_size + spacing

        grid_size = self.grid_size
        base_grid = []

        # Step 1: Generate unrotated local grid positions (centered at origin)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x = (i - (grid_size[0] - 1) / 2) * stride
                y = (j - (grid_size[1] - 1) / 2) * stride
                base_grid.append(np.array([x, y]))

        # Step 2: Apply grid-wide rotation and translation
        rotation_deg = np.random.uniform(-5, 5)
        self.grid_rotation_rad = np.radians(rotation_deg)
        rotation_matrix = np.array([
            [np.cos(self.grid_rotation_rad), -np.sin(self.grid_rotation_rad)],
            [np.sin(self.grid_rotation_rad), np.cos(self.grid_rotation_rad)],
        ])
        translation_xy = self.base_xy + np.random.uniform(-0.05, 0.05, size=2)

        z = self.env.table_height + self.env.block_half_extents[2] + 0.01

        for field_id, local_xy in enumerate(base_grid):
            rotated_xy = rotation_matrix @ local_xy
            final_xy = rotated_xy + translation_xy
            self.fields[field_id] = {
                "position": (final_xy[0], final_xy[1], z),
                "target_id": None,
                "block_id": None,
            }

    def reset_task_scenario(self):
        """Set up fields, assign target/block IDs, and define goal_config as target_id → block_id."""
        self.setup_field_grid()

        all_field_ids = list(self.fields.keys())

        # === Safety checks based on config flags ===
        if not self.allow_occupied:
            if self.num_targets + self.num_blocks > len(all_field_ids):
                raise ValueError("Too few fields for chosen number of targets + blocks.")
        else:
            if self.num_blocks > len(all_field_ids):
                raise ValueError("Too few fields for blocks when target fields can be reused.")

        # === Sample target fields ===
        target_fields = random.sample(all_field_ids, self.num_targets)

        # === Assign target IDs ===
        for i, fid in enumerate(target_fields):
            self.fields[fid]["target_id"] = i

        self.target_id_to_field = {
            self.fields[fid]["target_id"]: fid
            for fid in target_fields
        }

        # === Clear all block IDs ===
        for f in self.fields.values():
            f["block_id"] = None

        # === Recursive block placement ===
        blocks_to_place = list(range(self.num_blocks))
        available_fields = list(self.fields.keys())
        self._recursive_place_blocks(blocks_to_place, available_fields)

        # === Define goal_config: target_id i → block_id i ===
        goal_config = {i: i for i in range(min(self.num_targets, self.num_blocks))}

        return {
            "fields": self.fields.copy(),
            "goal_config": goal_config,
            "target_field_ids": target_fields,
            "block_field_ids": [fid for fid in self.fields if self.fields[fid].get("block_id") is not None],
        }

    def move_block_between_fields(self, block_id, field_from, field_to):
        """Move a block from one field to another, with internal consistency checks."""
        if self.fields[field_from]["block_id"] != block_id:
            raise ValueError(f"Field {field_from} does not contain block {block_id}")
        if self.fields[field_to]["block_id"] is not None:
            raise ValueError(f"Field {field_to} is already occupied.")

        self.fields[field_to]["block_id"] = block_id
        self.fields[field_from]["block_id"] = None

    def check_goal(self):
        block_pos = self.env.get_all_block_positions()
        body_to_block = {v: k for k, v in self.env.block_ids.items()}

        matched = 0
        total = len(self.env.goal_config)

        for target_id, expected_block_id in self.env.goal_config.items():
            target_fid = self.target_id_to_field[target_id]  # precomputed once at reset
            target_pos = self.fields[target_fid]["position"]

            found = None
            for body_id, pos in block_pos.items():
                if np.linalg.norm(np.array(pos[:2]) - np.array(target_pos[:2])) <= 0.03:
                    block_id = body_to_block[body_id]
                    if block_id == expected_block_id:
                        matched += 1
                    break  # only check the closest match

        correct_ratio = matched / total if total > 0 else 0.0
        return matched == total, correct_ratio

    # Helper Functions
    def _recursive_place_blocks(self, blocks_to_place, available_fields, used_fields=None):
        if used_fields is None:
            used_fields = set()

        if not blocks_to_place:
            return  # Base case: done placing

        block_id = blocks_to_place[0]
        random.shuffle(available_fields)

        for fid in available_fields:
            if fid in used_fields:
                continue

            target_id = self.fields[fid].get("target_id")

            if not self.allow_occupied:
                if target_id is not None:
                    continue  # Target fields are reserved
            else:
                if target_id == block_id:
                    continue  # Don't pre-place block on its correct target

            # Place block
            self.fields[fid]["block_id"] = block_id
            used_fields.add(fid)

            try:
                self._recursive_place_blocks(blocks_to_place[1:], available_fields, used_fields)
                return  # Success
            except RuntimeError:
                # Backtrack
                self.fields[fid]["block_id"] = None
                used_fields.remove(fid)

        raise RuntimeError(f"Failed to place block {block_id} with current constraints.")

