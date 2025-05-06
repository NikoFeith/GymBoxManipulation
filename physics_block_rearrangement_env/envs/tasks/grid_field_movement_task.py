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
        self.base_xy = np.array(self.env.table_start_pos[:2])
        self.grid_rotation_rad = 0
        self.z = self.env.table_height + self.env.block_half_extents[2] + 0.01

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
        if self.num_targets + self.num_blocks > len(all_field_ids):
            raise ValueError("Too few fields for chosen number of targets + blocks.")

        # === Sample fields ===
        target_fields = random.sample(all_field_ids, self.num_targets)
        remaining_fields = [fid for fid in all_field_ids if fid not in target_fields]
        block_fields = random.sample(remaining_fields, self.num_blocks)

        # === Assign stable randomized target IDs (for visual/color consistency)
        shuffled_target_ids = list(range(self.num_targets))
        random.shuffle(shuffled_target_ids)
        for fid, tid in zip(target_fields, shuffled_target_ids):
            self.fields[fid]["target_id"] = tid

        # === Assign block IDs
        for idx, fid in enumerate(block_fields):
            self.fields[fid]["block_id"] = idx

        # === Create stable goal_config {target_id: block_id}
        # only for matching pairs
        goal_config = {}
        matching_count = min(len(target_fields), len(block_fields))
        matching_targets = random.sample(target_fields, matching_count)
        matching_blocks = random.sample(block_fields, matching_count)

        for t_fid, b_fid in zip(matching_targets, matching_blocks):
            target_id = self.fields[t_fid]["target_id"]
            block_id = self.fields[b_fid]["block_id"]
            goal_config[target_id] = block_id

        return {
            "fields": self.fields.copy(),
            "goal_config": goal_config,
            "target_field_ids": target_fields,
            "block_field_ids": block_fields,
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
        """Check if each goal-mapped target ID is occupied by the correct block ID."""
        for target_id, expected_block_id in self.env.goal_config.items():
            # Find the field with this target ID
            matched_fields = [fid for fid, data in self.fields.items() if data.get("target_id") == target_id]
            if not matched_fields:
                logger.warning(f"No field found with target_id {target_id}")
                return False

            target_field = matched_fields[0]
            target_pos = self.fields[target_field]["position"]
            actual_block_id = self.env.get_block_at_position(target_pos, threshold=0.03)

            logger.debug(
                f"[GOAL CHECK] target_id {target_id} → expected block {expected_block_id}, found {actual_block_id}")

            if actual_block_id != expected_block_id:
                return False

        return True

