import random

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import os
import re
from pathlib import Path
import yaml
import cv2

from physics_block_rearrangement_env.utils.robot_utils import *
from physics_block_rearrangement_env.utils.logging_utils import *
from physics_block_rearrangement_env.envs.task_interface import BaseTask
from physics_block_rearrangement_env.envs import tasks

DEFAULT_INIT_LOG_LEVEL = logging.ERROR
logger = setup_logger(__name__, level=DEFAULT_INIT_LOG_LEVEL)

class PhysicsBlockRearrangementEnv(gym.Env):
    """
    PyBullet Gymnasium environment for block rearrangement using a Panda robot arm.
    Supports modular task loading and reusable robot control.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}


    def __init__(self, render_mode=None, use_gui=False,
                 task_config_file="place_easy.yaml",
                 base_config_file="base_config.yaml"):
        super().__init__()

        # --- Render settings ---
        self.render_mode = render_mode
        self.use_gui = use_gui or (render_mode == "human")  # GUI if rendering in human mode

        # --- Load configuration files ---
        self._load_and_merge_configs(base_config_file, task_config_file)
        self.config = self.config or {}  # fallback to empty dict
        self._configure_logging()

        # --- Connect to PyBullet (GUI or headless) ---
        self.client = -1
        self._setup_pybullet()

        # --- Load static scene elements and robot model ---
        self._load_scene_assets()
        self._load_robot_parameters()
        self._load_robot()
        self._initialize_robot_joints()

        # --- Load task-specific parameters ---
        self._load_task_settings()
        self._load_colors()

        # --- Load task logic (e.g. placement) ---
        self._load_and_instantiate_task(tasks, BaseTask)

        # --- Setup Gym RL interface ---
        self._setup_action_space()
        self._setup_observation_space()

        # --- Internal runtime state ---
        self.current_steps = 0
        self.held_object_id = None
        self.held_object_idx = None
        self.grasp_constraint_id = None
        self.goal_config = {}
        self.block_ids = {}            # IDs of active block objects
        self.target_ids = {}           # IDs of active target visuals
        self.fields = {}

        logger.info("Environment initialized.")

    # region CONFIGURATION LOADING + LOGGING

    def _load_and_merge_configs(self, base_config_file, task_config_file):
        """
        Loads and merges two YAML config files: base + task-specific.
        Result is stored in self.config.
        """
        config_dir = Path(__file__).parent / "configs"
        base_path = config_dir / base_config_file
        task_path = config_dir / "tasks" / task_config_file

        base_config = {}
        task_config = {}

        if base_path.exists():
            with open(base_path, 'r') as f:
                base_config = yaml.safe_load(f) or {}

        if task_path.exists():
            with open(task_path, 'r') as f:
                task_config = yaml.safe_load(f) or {}
        else:
            # Optional fallback: look for task config outside 'tasks/' subfolder
            alt_path = config_dir / task_config_file
            if alt_path.exists():
                with open(alt_path, 'r') as f:
                    task_config = yaml.safe_load(f) or {}

        def deep_merge(a, b):
            # Recursive dict merge: values in b overwrite those in a
            for k, v in b.items():
                if isinstance(v, dict):
                    a[k] = deep_merge(a.get(k, {}), v)
                else:
                    a[k] = v
            return a

        self.config = deep_merge(base_config.copy(), task_config)

    def _configure_logging(self):
        """
        Applies the logging level from the config.
        """
        log_cfg = self.config.get("logging", {})
        level_str = log_cfg.get("level", "INFO").upper()
        level = get_level_from_string(level_str)

        # Set logger level + handler levels
        logger.setLevel(level)
        for h in logger.handlers:
            h.setLevel(level)

        logger.info(f"Log level set to {level_str}")

    # endregion

    # region BULLET SETUP

    def _setup_pybullet(self):
        """
        Connects to PyBullet (GUI or DIRECT mode) and configures basic physics and camera.
        """
        phys_cfg = self.config.get("physics", {})
        cam_cfg = self.config.get("camera", {})

        # Extract physics params
        gravity = phys_cfg.get("gravity", [0, 0, -9.81])
        self.timestep = phys_cfg.get("timestep", 1.0 / 240)
        solver_iters = phys_cfg.get("num_solver_iterations", 150)

        # Extract camera params
        self.camera_target_pos = cam_cfg.get("target_pos", [0.55, 0.0, 0.65])
        self.camera_distance = cam_cfg.get("distance", 1.0)
        self.camera_yaw = cam_cfg.get("yaw", 75)
        self.camera_pitch = cam_cfg.get("pitch", -45)

        # Choose connection type
        if self.use_gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2, cameraYaw=90, cameraPitch=-40,
                cameraTargetPosition=self.camera_target_pos, physicsClientId=self.client
            )
        else:
            self.client = p.connect(p.DIRECT)

        # Set global physics params
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*gravity, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep,
            numSolverIterations=solver_iters,
            physicsClientId=self.client
        )
        p.setRealTimeSimulation(0, physicsClientId=self.client)

        logger.info("PyBullet connection established and configured.")

    def _load_scene_assets(self):
        """
        Loads static URDFs like the ground plane and table.
        Calculates table height from bounding box.
        """
        assets = self.config.get("assets", {})
        plane_path = assets.get("plane", "plane.urdf")
        table_path = assets.get("table", "table/table.urdf")
        self.table_start_pos = assets.get("table_start_pos", [0, 0, 0])

        # Load plane and table
        self.plane_id = p.loadURDF(plane_path, physicsClientId=self.client)
        self.table_id = p.loadURDF(
            table_path, basePosition=self.table_start_pos, useFixedBase=True, physicsClientId=self.client
        )

        # Compute height of table (used for positioning blocks, robot, etc.)
        aabb_min, aabb_max = p.getAABB(self.table_id, -1, physicsClientId=self.client)
        self.table_height = aabb_max[2]
        logger.info(f"Scene assets loaded. Table height = {self.table_height:.3f}")

    # endregion

    # region ROBOT SETUP

    def _load_robot_parameters(self):
        """
        Loads robot joint names, control gains, and grasp parameters from config.
        """
        robot_cfg = self.config.get("robot", {})
        panda_cfg = robot_cfg.get("panda", {})

        self.robot_type = robot_cfg.get("type", "panda")
        self.arm_joint_names = panda_cfg.get("arm_joint_names", [f"panda_joint{i+1}" for i in range(7)])
        self.finger_joint_names = panda_cfg.get("finger_joint_names", ["panda_finger_joint1", "panda_finger_joint2"])

        # End-effector link fallbacks
        self.preferred_ee_link_name = panda_cfg.get("preferred_ee_link_name", "panda_hand")
        self.fallback_ee_link_name_1 = panda_cfg.get("fallback_ee_link_name_1", "panda_link7")
        self.fallback_ee_link_name_2 = panda_cfg.get("fallback_ee_link_name_2", "panda_link8")

        # Arm control parameters
        self.arm_max_forces = panda_cfg.get("arm_max_forces", [100.0] * 7)
        self.arm_kp = panda_cfg.get("arm_kp", [0.05] * 7)
        self.arm_kd = panda_cfg.get("arm_kd", [1.0] * 7)

        # Gripper control parameters
        self.gripper_open_value = panda_cfg.get("gripper_open_value", 0.04)
        self.gripper_closed_value = panda_cfg.get("gripper_closed_value", 0.025)
        self.gripper_max_force = panda_cfg.get("gripper_max_force", 40)
        self.gripper_kp = panda_cfg.get("gripper_kp", 0.2)
        self.gripper_kd = panda_cfg.get("gripper_kd", 1.0)

        # IK settings
        ik_solver_name = panda_cfg.get("ik_solver", "IK_DLS")
        self.ik_solver = getattr(p, ik_solver_name, p.IK_DLS)
        self.failed_ik_counter = 0
        self.max_failed_ik_retries = panda_cfg.get("max_failed_ik_retries", 3)

        # Default joint pose (home) + IK rest pose
        self.home_pose_joints = panda_cfg.get("home_pose_joints", [
            0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4
        ])
        self.rest_poses_for_ik = self.home_pose_joints

        # Grasp offset (Z lift between hand frame and block)
        self.grasp_offset_in_hand_frame = panda_cfg.get("grasp_offset_in_hand_frame", [0.0, 0.0, 0.065])

        logger.info("Robot parameters loaded.")

    def _load_robot(self):
        """
        Loads the robot URDF using parameters from config.
        """
        urdf_path = self.config.get("robot", {}).get("urdf_path", "franka_panda/panda.urdf")
        start_pos = [0, 0, self.table_height]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])

        try:
            self.robot_id = p.loadURDF(
                urdf_path, start_pos, start_ori, useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self.client
            )
            logger.info(f"Robot loaded from {urdf_path} with ID {self.robot_id}")
        except Exception as e:
            logger.critical(f"Robot load failed: {e}")
            self.close()
            raise e

    def _initialize_robot_joints(self):
        """
        Finds joint indices, resets joints to home pose, and enables motors.
        """
        logger.info("Initializing robot joints and gripper...")

        # Lookup indices from joint names
        self.arm_joint_indices = find_joint_indices(self.robot_id, self.arm_joint_names, self.client)
        self.finger_indices = find_joint_indices(self.robot_id, self.finger_joint_names, self.client)

        # Try preferred EE link, fallback if not available
        preferred = find_link_index_safely(self.robot_id, self.preferred_ee_link_name, self.client)
        fallback_1 = find_link_index_safely(self.robot_id, self.fallback_ee_link_name_1, self.client)
        fallback_2 = find_link_index_safely(self.robot_id, self.fallback_ee_link_name_2, self.client)

        if preferred is not None:
            self.ee_link_index = preferred
        elif fallback_1 is not None:
            self.ee_link_index = fallback_1
        elif fallback_2 is not None:
            self.ee_link_index = fallback_2
        else:
            raise ValueError("Could not find any valid EE link index.")

        # Store EE name for reference
        self.ee_link_name = self.arm_joint_names[self.ee_link_index] if self.ee_link_index < len(self.arm_joint_names) else "unknown"

        # Get joint limits
        self.arm_limits = get_arm_kinematic_limits_and_ranges(self.robot_id, self.arm_joint_indices, self.client)

        # Reset joints to home pose
        for i, joint_idx in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, self.home_pose_joints[i], targetVelocity=0.0, physicsClientId=self.client)
            p.setJointMotorControl2(
                self.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=self.home_pose_joints[i],
                force=self.arm_max_forces[i],
                positionGain=self.arm_kp[i], velocityGain=self.arm_kd[i],
                physicsClientId=self.client
            )

        # Reset gripper
        for idx in self.finger_indices:
            p.resetJointState(self.robot_id, idx, self.gripper_open_value, targetVelocity=0.0, physicsClientId=self.client)
        self._set_gripper_state("open", wait=False)

        wait_steps(100, client=self.client)  # let it settle
        logger.info("Robot joints initialized.")

    # endregion

    # region TASK SETUP

    def _load_and_instantiate_task(self, task_package, base_class):
        """
        Dynamically loads the task class and instantiates it.
        """
        task_cfg = self.config.get("task", {})
        class_name = task_cfg.get("task_class_name", "BlockPlacementTask")

        if not task_cfg.get("task_class_name"):
            logger.warning(f"task_class_name missing in config, defaulting to {class_name}")

        # Infer module name (e.g., BlockPlacementTask → block_placement_task)
        module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        if not module_name.endswith("_task"):
            module_name += "_task"

        task_class = None
        try:
            task_module = getattr(task_package, module_name, None)
            if task_module:
                task_class = getattr(task_module, class_name, None)
        except Exception as e:
            logger.error(f"Dynamic task loading error: {e}", exc_info=True)

        if not task_class or not issubclass(task_class, base_class):
            raise ValueError(f"Invalid task class '{class_name}' or not subclass of {base_class.__name__}.")

        # Create task instance
        self.task = task_class(self, task_cfg)
        logger.info(f"Loaded task class: {self.task.__class__.__name__}")

        # Pull some task-defined values into env
        self.num_blocks = self.task.num_blocks
        self.num_targets = getattr(self.task, "num_targets", self.num_blocks)

    def _load_task_settings(self):
        """
        Loads environment + reward parameters related to task setup and limits.
        """
        task_cfg = self.config.get("task", {})
        sim_cfg = self.config.get("simulation", {})
        reward_cfg = self.config.get("reward", {})

        self.target_layout = task_cfg.get("target_layout", "line")
        self.num_blocks = task_cfg.get("num_blocks", 1)
        self.num_targets = task_cfg.get("num_targets", 1)

        self.block_scale = task_cfg.get("block_scale", 0.05)
        self.block_half_extents = [self.block_scale / 2.0] * 3

        self.z_hover_offset = sim_cfg.get("z_hover_offset", 0.15)
        self.grasp_clearance_above_top = sim_cfg.get("grasp_clearance_above_top", 0.1)
        self.place_clearance_above_top = sim_cfg.get("place_clearance_above_top", 0.1)

        self.primitive_max_steps = sim_cfg.get("primitive_max_steps", 400)
        self.max_steps = sim_cfg.get("max_episode_steps", 50 * self.num_blocks)
        self.gripper_wait_steps = sim_cfg.get("gripper_wait_steps", 120)

        # Pose tolerances
        self.pose_reached_threshold = sim_cfg.get("pose_reached_threshold", 0.01)
        self.orientation_reached_threshold = sim_cfg.get("orientation_reached_threshold", 0.1)
        self.goal_dist_threshold = self.config.get("goal_dist_threshold", 0.04)

        # Rewards
        self.goal_reward = reward_cfg.get("goal_reward", 1.0)
        self.step_penalty = reward_cfg.get("step_penalty", -0.01)
        self.move_fail_penalty = reward_cfg.get("move_fail_penalty", 0.005)

    def _load_colors(self):
        """
        Assigns RGBA colors to blocks and targets from config.
        Falls back to defaults if not specified.
        """
        colors = self.config.get("colors", {})

        default_block_colors = [
            [0.9, 0.1, 0.1, 1.0], [0.1, 0.8, 0.1, 1.0], [0.1, 0.1, 0.9, 1.0],
            [0.9, 0.9, 0.1, 1.0], [0.9, 0.1, 0.9, 1.0]
        ]
        default_target_colors = [
            [0.9, 0.5, 0.5, 1.0], [0.5, 0.9, 0.5, 1.0], [0.5, 0.5, 0.9, 1.0],
            [0.9, 0.9, 0.5, 1.0], [0.9, 0.5, 0.9, 1.0]
        ]

        all_block = colors.get("block_rgba", default_block_colors)
        all_target = colors.get("target_rgba", default_target_colors)

        if self.num_blocks <= len(all_block):
            self.block_colors_rgba = all_block
            self.target_colors_rgba = all_target
        else:
            logger.warning("Block count exceeds color list. Repeating colors.")
            self.block_colors_rgba = (all_block * ((self.num_blocks // len(all_block)) + 1))
            self.target_colors_rgba = (all_target * ((self.num_blocks // len(all_target)) + 1))

        logger.info("Colors loaded and sliced.")

    # endregion

    # region TASK HELPERS
    def get_block_at_position(self, position, threshold=0.03):
        """
        Returns the block ID at a given (x, y, z) position, or None if no match is found.
        """
        for block_id in self.block_ids.values():
            pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self.client)
            if np.linalg.norm(np.array(pos[:2]) - np.array(position[:2])) <= threshold:
                return block_id
        return None

    def _place_target_visuals(self, target_field_ids):
        """Places visual markers (plates) at the field locations assigned to targets."""
        self.target_ids = {}

        if not hasattr(self, "fields") or not self.fields:
            logger.warning("Warning: Field data not available, cannot place target visuals.")
            return

        if not hasattr(self, "goal_config") or not self.goal_config:
            logger.warning("Warning: Goal config not defined, cannot place visuals.")
            self.close()
            return

        logger.debug(f"Placing {len(target_field_ids)} target visuals...")
        logger.debug(f"Goal Config for this episode: {self.goal_config}")

        plate_half_extents = [0.05, 0.05, 0.0005]
        plate_center_z = self.table_height + plate_half_extents[2] + 0.0001

        for target_idx, field_id in enumerate(target_field_ids):
            if field_id not in self.fields:
                logger.warning(f"Field ID {field_id} not found in fields. Skipping.")
                continue

            target_pos = self.fields[field_id]["position"]
            target_ori = p.getQuaternionFromEuler([0, 0, self.task.grid_rotation_rad])

            # Default to gray if no color can be assigned
            rgba = self.target_colors_rgba[target_idx % len(self.target_colors_rgba)] if target_idx != -1 else [0.5,
                                                                                                                0.5,
                                                                                                                0.5,
                                                                                                                0.5]

            try:
                vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=plate_half_extents,
                                             rgbaColor=rgba, physicsClientId=self.client)
                plate_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[target_pos[0], target_pos[1], plate_center_z],
                    baseOrientation=target_ori,
                    physicsClientId=self.client
                )
                if plate_id >= 0:
                    self.target_ids[target_idx] = {'plate_id': plate_id, 'vis_id': vis_id}
                else:
                    logger.warning(f"Failed to create visual for target field {field_id} (plate_id < 0)")
            except Exception as e:
                logger.exception(f"Error creating target plate at field {field_id}: {e}")

    def _spawn_blocks(self, block_field_ids):
        """Spawns blocks at the field positions assigned during reset, coloring them to match their goal target."""
        if not hasattr(self, "fields") or not self.fields:
            logger.warning("Warning: Field data not available, cannot spawn blocks.")
            return

        self.block_ids = {}
        logger.debug(f"Spawning {len(block_field_ids)} blocks...")

        # Build mapping: block_id → target_id (from goal_config)
        block_to_target_id = {block_id: target_id for target_id, block_id in self.goal_config.items()}

        # Track used and unused target_ids
        used_target_ids = set(block_to_target_id.values())
        all_target_ids = set(range(len(self.block_colors_rgba)))
        unused_target_ids = list(all_target_ids - used_target_ids)
        unused_color_cycle = iter(unused_target_ids) if unused_target_ids else iter([0])

        for field_id in block_field_ids:
            if field_id not in self.fields:
                logger.warning(f"Field ID {field_id} not found in fields. Skipping block spawn.")
                continue

            block_idx = self.fields[field_id]["block_id"]
            pos = self.fields[field_id]["position"]
            z = self.table_height + self.block_half_extents[2] + 0.001
            pos_with_z = (pos[0], pos[1], z)
            orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2 * np.pi)])

            try:
                body_id = p.loadURDF("cube.urdf", pos_with_z, orn,
                                     globalScaling=self.block_scale,
                                     physicsClientId=self.client)
                if body_id >= 0:
                    self.block_ids[block_idx] = body_id

                    # Determine color: goal-matched or fallback from unused target colors
                    target_id = block_to_target_id.get(block_idx, None)
                    if target_id is not None:
                        rgba = self.block_colors_rgba[target_id % len(self.block_colors_rgba)]
                    else:
                        try:
                            fallback_id = next(unused_color_cycle)
                        except StopIteration:
                            fallback_id = 0
                        rgba = self.block_colors_rgba[fallback_id % len(self.block_colors_rgba)]
                        logger.debug(f"Assigned fallback color {rgba} to distractor block {block_idx}")

                    p.changeVisualShape(body_id, -1, rgbaColor=rgba, physicsClientId=self.client)
                    logger.debug(f"Spawned block {block_idx} at field {field_id}, color={rgba}")
                else:
                    logger.warning(f"Failed to spawn block {block_idx} at field {field_id} (body_id < 0)")
            except Exception as e:
                logger.exception(f"Error spawning block {block_idx} at field {field_id}: {e}")

    # endregion

    # region RL INTERFACE SETUP
    def _setup_action_space(self):
        """Defines the discrete action space for pick/place actions."""
        num_fields = self.task.grid_size[0] * self.task.grid_size[1]
        self.action_space = spaces.MultiDiscrete([num_fields, num_fields])
        logger.info(f"Action space = MultiDiscrete({self.action_space.shape})")

    def _setup_observation_space(self):
        """Defines the observation space as a camera-based RGB image."""
        cam_cfg = self.config.get("camera", {})
        self.image_size = cam_cfg.get("image_size", 84)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.image_size, self.image_size, 3),
            dtype=np.uint8
        )
        logger.info(f"Observation space = Box({self.image_size}, {self.image_size}, 3)")
    # endregion

    #region RESET / STEP
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_steps = 0
        self.held_object_id = None
        self.held_object_idx = None
        self.grasp_constraint_id = None
        self.goal_config = {}

        self._clear_sim_objects()
        self._reset_robot_home()

        # --- Ask task for scenario setup ---
        if self.task is None:
            raise RuntimeError("Task was not initialized before reset.")

        task_info = self.task.reset_task_scenario()
        self.fields = task_info["fields"]
        self.goal_config = task_info["goal_config"]

        # Let objects settle
        wait_steps(150, client=self.client, timestep=self.timestep, use_gui=self.use_gui)

        # --- Visualize ---
        self._place_target_visuals(task_info["target_field_ids"])
        self._spawn_blocks(task_info["block_field_ids"])

        # Initial observation and info
        obs = self._get_obs()
        info = self._get_info()
        if isinstance(task_info, dict):
            info.update(task_info)

        return obs, info

    def _clear_sim_objects(self):
        """Removes blocks, targets, and grasp constraints from the simulation."""
        if self.grasp_constraint_id is not None:
            try:
                p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
            except Exception:
                pass
            self.grasp_constraint_id = None

        for block_id in self.block_ids.values():
            try:
                p.removeBody(block_id, physicsClientId=self.client)
            except Exception:
                pass

        for visual in self.target_ids.values():
            try:
                p.removeBody(visual["plate_id"], physicsClientId=self.client)
            except Exception:
                pass

        self.block_ids = {}
        self.target_ids = {}

    def _reset_robot_home(self):
        """Resets robot to home joint configuration and opens gripper."""
        if not hasattr(self, 'arm_joint_indices') or not self.arm_joint_indices:
            logger.warning("Skipping robot reset: joint indices not initialized.")
            return

        # Reset arm joints
        for i, idx in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, idx, self.home_pose_joints[i], 0.0, self.client)
            p.setJointMotorControl2(
                self.robot_id, idx, p.POSITION_CONTROL,
                targetPosition=self.home_pose_joints[i],
                force=self.arm_max_forces[i],
                positionGain=self.arm_kp[i],
                velocityGain=self.arm_kd[i],
                physicsClientId=self.client
            )

        # Open gripper and wait for stabilization
        self._set_gripper_state("open", wait=False)
        wait_steps(50, client=self.client, timestep=self.timestep, use_gui=self.use_gui)

    def step(self, action):
        """Executes one action using (field_from_id, field_to_id) instead of (block_id, target_id)."""
        field_from_id, field_to_id = action
        self.current_steps += 1

        # Reset any grasped object from previous step
        if self.held_object_id is not None:
            self._reset_grasp_state()

        # === Access field info ===
        fields = self.task.fields
        if field_from_id not in fields or field_to_id not in fields:
            logger.warning(f"Invalid field ids: {field_from_id} → {field_to_id}")
            return self._get_obs(), self.step_penalty, False, False, {"error": "invalid_fields"}

        block_id = fields[field_from_id]["block_id"]
        if block_id is None:
            logger.warning(f"Pick failed: no block at field {field_from_id}")
            return self._get_obs(), self.step_penalty, False, False, {"error": "empty_pick"}

        if fields[field_to_id]["block_id"] is not None:
            logger.warning(f"Place failed: target field {field_to_id} already occupied")
            return self._get_obs(), self.step_penalty + self.move_fail_penalty, False, False, {
                "error": "occupied_target"}

        # === Try pick ===
        success_pick = self._primitive_pick(block_id)

        # === Try place if pick succeeded ===
        success_place = False
        if success_pick:
            success_place = self._primitive_place(field_to_id)

            if success_place:
                self.task.move_block_between_fields(block_id, field_from_id, field_to_id)

        # === Success & termination ===
        success = success_pick and success_place
        terminated = self.task.check_goal()
        truncated = False  # Add max_steps logic if needed

        # === Reward logic ===
        reward = self.goal_reward if terminated else self.step_penalty
        if not success:
            reward += self.move_fail_penalty

        info = {
            "primitive_success": success,
            "pick_success": success_pick,
            "place_success": success_place,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _goal_blocks_fell(self):
        """Returns True if a goal-relevant block is below the table."""
        if not self.goal_config:
            return False

        task_cfg = self.config.get("task")

        required_blocks = set(self.goal_config.values())

        min_x = task_cfg.get("min_x", -0.3) + self.table_start_pos[0]
        max_x = task_cfg.get("max_x", 0.3) + self.table_start_pos[0]
        min_y = task_cfg.get("min_y", -0.3) + self.table_start_pos[1]
        max_y = task_cfg.get("max_y", 0.3) + self.table_start_pos[1]

        min_z = self.table_height * 0.8  # Threshold well below table

        for block_idx in required_blocks:
            if block_idx is None:
                continue
            elif block_idx < len(self.block_ids):
                try:
                    x, y, z = p.getBasePositionAndOrientation(self.block_ids[block_idx], self.client)[0]
                    if z < min_z:
                        logger.warning(f"Goal block {block_idx} fell below Z={z:.3f}.")
                        return True
                    if not (min_x <= x <= max_x):
                        logger.warning(f"Goal block {block_idx} fell out of x-boundary  X={x:.3f}.")
                        return True
                    if not (min_y <= y <= max_y):
                        logger.warning(f"Goal block {block_idx} fell out of y-boundary  Y={y:.3f}.")
                        return True

                except Exception as e:
                    logger.error(f"Error checking Z of block {block_idx}: {e}")
                    return True
        return False

    # endregion

    # region PRIMITIVES
    def _primitive_pick(self, block_idx):
        """Tries to pick up the block at the given index using dual-orientation fallback."""
        if self.held_object_id is not None:
            logger.warning("Pick failed: already holding an object.")
            self.last_failure_reason = "already_holding"
            return False

        if block_idx >= len(self.block_ids):
            logger.warning(f"Pick failed: invalid block index {block_idx}")
            self.last_failure_reason = "invalid_index"
            return False

        body_id = self.block_ids[block_idx]
        try:
            block_pos, block_orn = p.getBasePositionAndOrientation(body_id, self.client)
        except Exception as e:
            logger.error(f"Failed to get block pose: {e}")
            self.last_failure_reason = "pose_fetch_failed"
            return False

        goal_orn = p.getQuaternionFromEuler([0, 0, 0])  # Can be replaced with target-specific logic
        preferred, alternative = self._calculate_preferred_grasp_orientation(block_orn, goal_orn)

        if preferred is None:
            logger.error("Grasp orientation calculation failed.")
            self.last_failure_reason = "grasp_calc_failed"
            return False

        logger.info("Trying preferred grasp orientation")
        if self._attempt_pick_sequence(body_id, block_idx, preferred):
            return True

        logger.info("Trying alternative grasp orientation")
        if self._attempt_pick_sequence(body_id, block_idx, alternative):
            return True

        self.last_failure_reason = "both_orientations_failed"
        return False

    def _primitive_place(self, field_id):
        """
        Places the currently held object at the position of the given field ID.

        Args:
            field_id: ID of the field in self.task.fields to place the object onto.

        Returns:
            True if placement succeeded, False otherwise.
        """
        if self.held_object_id is None:
            logger.warning("Place failed: nothing is held.")
            return False

        if field_id not in self.task.fields:
            logger.warning(f"Place failed: field ID {field_id} not found.")
            return False

        place_position = self.task.fields[field_id]["position"]
        pre_z = self.table_height + self.z_hover_offset
        place_z = self.table_height + self.place_clearance_above_top
        target_ori = p.getQuaternionFromEuler([np.pi, 0, 0])  # standard downward grip

        pre_pose = [place_position[0], place_position[1], pre_z]
        place_pose = [place_position[0], place_position[1], place_z]

        # Move above the target field
        if not self._move_ee_to_pose(pre_pose, target_ori):
            return False

        # Move down to place
        self._move_ee_to_pose(place_pose, target_ori)

        # Release grasp
        if self.grasp_constraint_id is not None:
            try:
                p.removeConstraint(self.grasp_constraint_id, self.client)
            except Exception:
                pass
            self.grasp_constraint_id = None

        self._set_gripper_state("open", wait=True)
        self.held_object_id = None
        self.held_object_idx = None

        # Lift after placing
        self._move_ee_to_pose(pre_pose, target_ori)
        return True

    def _check_target_clear(self, target_idx, target_pos):
        """Checks if another block is already occupying the target location."""
        target_xy = np.array(target_pos[:2])
        radius_sq = (self.block_scale * 0.8) ** 2

        for i, body_id in self.block_ids.items():
            if body_id == self.held_object_id:
                continue
            try:
                other_xy = np.array(p.getBasePositionAndOrientation(body_id, self.client)[0][:2])
                if np.sum((target_xy - other_xy) ** 2) < radius_sq:
                    logger.warning(f"Target {target_idx} occupied by block {i}")
                    return False
            except Exception:
                continue
        return True

    def _place_sequence(self, loc_type, loc_idx, target_pos):
        """Executes the placement motion and releases the object."""
        pre_z = self._clamp_hover_z(self.table_height + self.z_hover_offset)
        place_z = self._clamp_table_z(self.table_height + self.place_clearance_above_top)


        ori = p.getQuaternionFromEuler([np.pi, 0, 0])
        pre = [target_pos[0], target_pos[1], pre_z]
        place = [target_pos[0], target_pos[1], place_z]

        if not self._move_ee_to_pose(pre, ori):
            if self.failed_ik_counter >= self.max_failed_ik_retries:
                logger.warning(f"Too many IK failures. Resetting to home.")
                self._move_to_home_pose()
                return False
            return False

        self._move_ee_to_pose(place, ori)  # Allow soft failures

        if self.grasp_constraint_id is not None:
            try:
                p.removeConstraint(self.grasp_constraint_id, self.client)
            except Exception:
                pass
            self.grasp_constraint_id = None

        self._set_gripper_state("open", wait=True)
        self.held_object_id = None
        self.held_object_idx = None

        wait_steps(50, client=self.client, timestep=self.timestep, use_gui=self.use_gui)

        # Try moving back up after release
        if not self._move_ee_to_pose(pre, ori):
            logger.warning(f"Post-place lift failed for {loc_type} {loc_idx}")
            if self.failed_ik_counter >= self.max_failed_ik_retries:
                logger.warning(f"Too many IK failures. Resetting to home.")
                self._move_to_home_pose()
                return False
        return True

    # endregion

    # region CONTROL HELPERS
    def _attempt_pick_sequence(self, block_id, block_idx, grasp_ori):
        """
        Full sequence to pick a block:
        1. Open gripper
        2. Move above -> down -> grasp
        3. Close gripper
        4. Attach block with constraint
        5. Lift
        """
        try:
            # Get current block position
            pos, _ = p.getBasePositionAndOrientation(block_id, self.client)

            pre = [pos[0], pos[1], pos[2] + self.z_hover_offset]
            grasp = [pos[0], pos[1], pos[2] + self.grasp_clearance_above_top]
            lift = [pos[0], pos[1], pos[2] + self.z_hover_offset + 0.05]

            if not self._set_gripper_state("open", wait=True):
                return False

            if not self._move_ee_to_pose(pre, grasp_ori):
                if self.failed_ik_counter >= self.max_failed_ik_retries:
                    logger.warning(f"Too many IK failures. Resetting to home.")
                    self._move_to_home_pose()
                    return False
                return False

            if not self._move_ee_to_pose(grasp, grasp_ori):
                self._recover_from_grasp_failure(grasp_ori)
                return False

            if not self._set_gripper_state("close", wait=True):
                self._move_ee_to_pose(pre, grasp_ori)
                return False

            if not self._check_gripper_closed():
                self._recover_from_grasp_failure(grasp_ori)
                return False

            if not self._attach_constraint(block_id):
                self._recover_from_grasp_failure(grasp_ori)
                return False

            if not self._move_ee_to_pose(lift, grasp_ori):
                self._reset_grasp_state()
                if self.failed_ik_counter >= self.max_failed_ik_retries:
                    logger.warning(f"Too many IK failures. Resetting to home.")
                    self._move_to_home_pose()
                    return False
                return False

            self.held_object_id = block_id
            self.held_object_idx = block_idx
            return True

        except Exception as e:
            logger.exception(f"Pick sequence error for block {block_idx}: {e}")
            self._reset_grasp_state()
            return False

    def _recover_from_grasp_failure(self, ori):
        """Attempts a vertical lift if pick fails."""
        try:
            pos, current_ori = self._get_ee_pose()
            if pos:
                up = [pos[0], pos[1], self.table_height + self.z_hover_offset]
                self._move_ee_to_pose(up, current_ori or ori)
            self._set_gripper_state("open", wait=False)
        except Exception as e:
            logger.warning(f"Grasp recovery failed: {e}")

    def _reset_grasp_state(self):
        """Removes grasp constraint and clears held object state."""
        if self.grasp_constraint_id is not None:
            try:
                p.removeConstraint(self.grasp_constraint_id, self.client)
            except Exception:
                pass
            self.grasp_constraint_id = None

        self.held_object_id = None
        self.held_object_idx = None
        self._set_gripper_state("open", wait=True)
        self._move_to_home_pose()

    def _attach_constraint(self, block_id):
        """Creates a fixed joint between EE and block."""
        try:
            ee_pos, ee_ori = self._get_ee_pose()
            if ee_pos is None:
                return False

            block_pos, block_ori = p.getBasePositionAndOrientation(block_id, self.client)
            inv_pos, inv_ori = p.invertTransform(ee_pos, ee_ori)
            _, block_rel_ori = p.multiplyTransforms(inv_pos, inv_ori, block_pos, block_ori)

            self.grasp_constraint_id = p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=self.ee_link_index,
                childBodyUniqueId=block_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=self.grasp_offset_in_hand_frame,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=block_rel_ori,
                childFrameOrientation=[0, 0, 0, 1],
                physicsClientId=self.client
            )
            return self.grasp_constraint_id >= 0

        except Exception as e:
            logger.error(f"Constraint creation failed: {e}")
            return False

    def _check_gripper_closed(self):
        """Checks if fingers actually closed tight (heuristic)."""
        if not self.finger_indices:
            return False

        states = p.getJointStates(self.robot_id, self.finger_indices, self.client)
        f1, f2 = states[0][0], states[1][0]
        close_threshold = self.gripper_closed_value + 0.01

        if f1 > close_threshold or f2 > close_threshold:
            logger.warning("Gripper did not close fully. Likely no object inside.")
            return False
        return True

    def _get_ee_pose(self):
        """Gets the current world pose of the end-effector link."""
        try:
            link_state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True,
                                        physicsClientId=self.client)
            return link_state[4], link_state[5]  # world pos, world orn
        except Exception as e:
            logger.exception(f"Error getting EE pose: {e}")
            return None, None

    def _move_to_home_pose(self):
        for i, joint_index in enumerate(self.arm_joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.rest_poses_for_ik[i],  # or define `self.home_joint_angles`
                force=self.arm_max_forces[i],
                positionGain=self.arm_kp[i],
                velocityGain=self.arm_kd[i],
                physicsClientId=self.client
            )
            self.failed_ik_counter = 0
            self.last_failure_reason = "home pose recovery"

        wait_steps(50, self.client, timestep=self.timestep, use_gui=self.use_gui)

    def _move_ee_to_pose(self, target_pos, target_ori, max_steps_override=None):
        """Computes IK and moves arm to the target pose. Returns True if pose reached."""
        self._last_ik_failure = False

        if self.arm_limits is None:
            logger.error("Arm limits not set.")
            return False

        ll, ul, jr = self.arm_limits
        start_time = time.time()

        joint_poses  = p.calculateInverseKinematics(
            self.robot_id, self.ee_link_index, target_pos, target_ori,
            lowerLimits=ll, upperLimits=ul, jointRanges=jr,
            restPoses=self.rest_poses_for_ik,
            solver=self.ik_solver,
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=self.client
        )
        ik_time = time.time() - start_time

        if not joint_poses or len(joint_poses) < len(self.arm_joint_indices):
            logger.warning(f"IK failed (took {ik_time:.3f}s)")
            self._last_ik_failure = True
            self.failed_ik_counter += 1
            self.last_failure_reason = "ik_calculation_failed"
            return False

        self.failed_ik_counter = 0  # reset on success
        poses = joint_poses[:len(self.arm_joint_indices)]
        p.setJointMotorControlArray(self.robot_id, self.arm_joint_indices, p.POSITION_CONTROL,
                                    targetPositions=poses,
                                    forces=self.arm_max_forces,
                                    positionGains=self.arm_kp,
                                    velocityGains=self.arm_kd,
                                    physicsClientId=self.client)

        max_steps = max_steps_override or self.primitive_max_steps
        wait_steps(max_steps, self.client, timestep=self.timestep, use_gui=self.use_gui)

        # Final pose check
        try:
            final_pos, final_ori = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True,
                                                  physicsClientId=self.client)[4:6]
            dist = np.linalg.norm(np.array(final_pos) - np.array(target_pos))
            ori_diff = p.getDifferenceQuaternion(target_ori, final_ori)
            _, angle = p.getAxisAngleFromQuaternion(ori_diff)

            if dist < self.pose_reached_threshold and abs(angle) < self.orientation_reached_threshold:
                return True
            else:
                logger.debug(f"Did not reach pose. Dist: {dist:.4f}, Angle error: {angle:.4f}")
                return False
        except Exception as e:
            logger.exception("Pose verification failed")
            return False

    def _set_gripper_state(self, state: str, wait: bool = True):
        """
        Sets gripper state to 'open' or 'close'.

        Args:
            state (str): 'open' or 'close'
            wait (bool): Wait for the motion to complete
        """
        logger.debug(f"Commanding gripper {state.upper()}")
        if state == 'open':
            target_value = self.gripper_open_value
        elif state == 'close':
            target_value = self.gripper_closed_value
        else:
            logger.error(f"Invalid gripper state '{state}'")
            return False

        if not self.finger_indices or len(self.finger_indices) != 2:
            logger.error("Gripper finger indices not set properly.")
            return False

        try:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.finger_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[target_value] * 2,
                forces=[self.gripper_max_force] * 2,
                positionGains=[self.gripper_kp] * 2,
                velocityGains=[self.gripper_kd] * 2,
                physicsClientId=self.client
            )
            if wait:
                wait_steps(self.gripper_wait_steps, self.client, timestep=self.timestep, use_gui=self.use_gui)
            return True
        except Exception as e:
            logger.exception(f"Gripper command failed: {e}")
            return False

    def _calculate_preferred_grasp_orientation(self, current_block_orn_quat, target_block_orn_quat=None):
        """
        Computes two possible grasp orientations (±90° from block yaw) and selects the one
        that requires the least yaw rotation to reach the target orientation.

        Tie-breakers:
        1. Prefer lower absolute delta yaw to target
        2. Prefer smaller signed delta
        3. Prefer positive rotation
        """
        if target_block_orn_quat is None:
            target_block_orn_quat = p.getQuaternionFromEuler([0, 0, 0])

        # Extract only yaw (Z axis) from current and target
        current_yaw = p.getEulerFromQuaternion(current_block_orn_quat)[2]
        target_yaw = p.getEulerFromQuaternion(target_block_orn_quat)[2]

        # Candidate grasp yaws (0 / ±90° from block orientation)
        candidate_yaws = [current_yaw + np.pi, current_yaw + np.pi / 2, current_yaw, current_yaw - np.pi / 2]
        labels = ["+180°","+90", "0", "-90"]
        candidates = []

        for label, cand_yaw in zip(labels, candidate_yaws):
            # Only compare yaw difference
            delta = self._normalize_angle(target_yaw - cand_yaw)
            quat = p.getQuaternionFromEuler([np.pi, 0, cand_yaw])  # gripper flipped over, rotated by yaw

            candidates.append({
                "label": label,
                "quat": quat,
                "delta": delta,
                "abs_delta": abs(delta)
            })

        # Sort with tie-breakers
        candidates.sort(key=lambda c: (c["abs_delta"], c["delta"], -c["delta"] > 0))

        preferred = candidates[0]
        alternative = candidates[1]

        logger.debug(
            f"Grasp choice → Preferred: {preferred['label']} (Δ={preferred['delta']:.2f}), "
            f"Alternative: {alternative['label']} (Δ={alternative['delta']:.2f})"
        )

        return preferred["quat"], alternative["quat"]

    def _clamp_hover_z(self, z):
        return max(z, self.z_hover_offset + self.table_height)

    def _clamp_table_z(self, z):
        return max(z, self.table_height)

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to be within [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # endregion

    # region RL HELPER
    def get_state(self) -> dict:
        """Returns a mapping from block index to the field it's currently occupying (or None)."""
        state = {}
        for fid, field in self.fields.items():
            block_idx = field.get("block_id")
            if block_idx is not None:
                state[block_idx] = fid
        return state

    def _get_obs(self):
        """Returns an RGB image observation with the robot hidden from view."""
        # === Hide robot visually ===
        num_links = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        p.changeVisualShape(self.robot_id, -1, rgbaColor=[1, 1, 1, 0], physicsClientId=self.client)  # base
        for link_idx in range(num_links):
            p.changeVisualShape(self.robot_id, link_idx, rgbaColor=[1, 1, 1, 0], physicsClientId=self.client)

        # === Capture image ===
        width = height = self.image_size
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target_pos,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.config['camera']['fov'],  # Lower FOV for less distortion
            aspect=1.0,
            nearVal=0.1,
            farVal=2.0
        )
        _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix, physicsClientId=self.client)
        rgb_array = np.array(rgb).astype(np.uint8)[:, :, :3]  # Drop alpha if needed

        # === Restore robot visibility ===
        p.changeVisualShape(self.robot_id, -1, rgbaColor=[1, 1, 1, 1], physicsClientId=self.client)
        for link_idx in range(num_links):
            p.changeVisualShape(self.robot_id, link_idx, rgbaColor=[1, 1, 1, 1], physicsClientId=self.client)

        logger.debug("Observation taken!")

        return rgb_array

    def _get_info(self):
        """
        Returns extra diagnostic info as a dictionary.
        Extend this with anything useful like:
        - distance to goal
        - block positions
        - grasp status
        """
        return {"held_object_idx": self.held_object_idx}

    def debug_save_if_multi_blocks_moving(self, episode: int = 0, threshold: float = 0.005,
                                          save_path: str = "./debug_saves"):
        """
        Manually call this to check if multiple blocks are moving and save a .bullet file if so.
        Args:
            episode (int): Current episode number (for filename)
            threshold (float): Linear velocity magnitude threshold
            save_path (str): Where to store the .bullet file
        """
        if not hasattr(self, "block_ids") or not self.block_ids:
            return

        moving_blocks = 0
        for block_id in self.block_ids.values():
            try:
                lin_vel, _ = p.getBaseVelocity(block_id, self.client)
                if np.linalg.norm(lin_vel) > threshold:
                    moving_blocks += 1
            except Exception:
                continue

        if moving_blocks > 1:
            logger.warning(f"[DEBUG] {moving_blocks} blocks are moving! Saving state.")
            os.makedirs(save_path, exist_ok=True)
            filename = f"anomaly_ep{episode:03d}.bullet"
            full_path = os.path.join(save_path, filename)
            p.saveBullet(full_path, physicsClientId=self.client)
            logger.info(f"[DEBUG] Saved .bullet snapshot: {full_path}")

    # endregion
