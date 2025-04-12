import logging

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import os
import re
from pathlib import Path
import yaml
import math

from physics_block_rearrangement_env.utils.robot_utils import *
from physics_block_rearrangement_env.utils.logging_utils import *
from physics_block_rearrangement_env.envs.task_interface import BaseTask
from physics_block_rearrangement_env.envs import tasks

log_level = logging.WARNING # Or logging.DEBUG
logger = setup_logger(__name__, level=log_level)


class PhysicsBlockRearrangementEnv(gym.Env):
    """
    Gymnasium environment for block rearrangement using PyBullet.
    Integrates robust Panda IK/Gripper control.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, use_gui=False,
                 task_config_file="place_3_line.yaml", # Default task name
                 base_config_file="base_config.yaml"):
        super().__init__()
        self.render_mode = render_mode
        self.use_gui = use_gui or (render_mode == 'human')

        # --- Internal State Variables ---
        self.assets_path = os.path.join(os.path.dirname(__file__), '..', 'assets')  # For local assets if needed
        self.block_ids = []
        self.target_ids = []
        self.goal_config = {}
        self.current_steps = 0
        self.held_object_id = None
        self.held_object_idx = None
        self.grasp_constraint_id = None

        # --- Load Configuration ---
        self._load_and_merge_configs(base_config_file, task_config_file)

        # --- Load and Instantiate Task using the helper method ---
        try:
            # Call the refactored method, passing the necessary imports
            self._load_and_instantiate_task(tasks, BaseTask)
        except ValueError as e:
            logger.critical(f": Environment initialization failed during task loading: {e}")
            raise e


        # --- Setup Physics, Scene, Robot (using base config) ---
        self.client = -1
        self._setup_pybullet_connection() # Connect and set basic physics params
        self._load_scene_assets() # Uses self.config for paths
        self._load_robot_parameters_from_config()
        self._load_robot() # Load the robot URDF


        # --- Task & RL Parameters ---
        self._load_task_parameters() # num_blocks, clearances, patterns etc.
        self._load_colors()          # Load and slice colors based on num_blocks


        # --- Position Initialization ---
        self._initialize_robot_info_and_state() # Finds indices, limits, resets pose
        self._setup_rl_interface()
        self.target_locations_pos = []  # Will be populated in reset
        self.dump_location_pos = []  # Will be populated in reset
        self._dump_location_base_pos = self.config.get('task', {}).get('dump_base_pos', [self.table_start_pos[0] - 0.15,
                                                                                         0.0])  # Store base XY for dump
        self.spawn_area_bounds = self.task.define_spawn_area()  # Define spawn area

        logger.info(f"Environment Initialized.")

    # ==========================================
    # --- Config Helper Methods ---
    # ==========================================
    def _load_and_merge_configs(self, base_config_file, task_config_file):
        """Loads base and task-specific config files and merges them."""
        self.config = {}
        config_dir = Path(__file__).parent / "configs"  # Assumes configs/ dir is sibling to envs/ dir
        base_config_path = config_dir / base_config_file
        # Construct task path using the provided filename
        task_config_path = config_dir / "tasks" / task_config_file  # Assumes task files are in 'tasks' subdir

        # Load Base Config
        base_config = {}
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                try:
                    base_config = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    logger.error(f"Error loading base config {base_config_path}: {e}")
        else:
            logger.warning(f"Base config file not found: {base_config_path}")

        # Load Task Config using the provided filename
        task_config = {}
        if task_config_path.exists():
            with open(task_config_path, 'r') as f:
                try:
                    task_config = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    logger.error(f"Error loading task config {task_config_path}: {e}")
        else:
            # Try looking in the main config dir as a fallback? Optional.
            task_config_path_alt = config_dir / task_config_file
            if task_config_path_alt.exists():
                logger.info(f" Task config '{task_config_file}' not found in tasks/, loading from {config_dir}")
                task_config_path = task_config_path_alt
                with open(task_config_path, 'r') as f:
                    try:
                        task_config = yaml.safe_load(f) or {}
                    except yaml.YAMLError as e:
                        logger.error(f"Error loading task config {task_config_path}: {e}")
            else:
                logger.warning(f"Task config file not found: {task_config_path} or {task_config_path_alt}")

        # Merge Configs (Simple Recursive Update: Task overrides Base)
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    # Get the existing dict or create a new one
                    # Then update it recursively
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v  # Assign the value, overwriting if key exists
            return d

        self.config = base_config.copy()  # Start with base
        self.config = deep_update(self.config, task_config)  # Merge task config in
        logger.info(f"Configuration loaded and merged using task file: '{task_config_file}'.")

    def _setup_pybullet_connection(self):
        """Connects to PyBullet and sets physics parameters from config."""
        _phys_cfg = self.config.get('physics', {})
        _cam_cfg = self.config.get('camera', {})  # Need camera params early for GUI reset

        gravity = _phys_cfg.get('gravity', [0, 0, -9.81])
        self.timestep = _phys_cfg.get('timestep', 1. / 240.)
        num_solver_iterations = _phys_cfg.get('num_solver_iterations', 150)

        self.camera_target_pos = _cam_cfg.get('target_pos', [0.55, 0.0, 0.65])
        self.camera_distance = _cam_cfg.get('distance', 1.0)
        self.camera_yaw = _cam_cfg.get('yaw', 75)
        self.camera_pitch = _cam_cfg.get('pitch', -45)

        if self.use_gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=-40,
                                         cameraTargetPosition=[0.5, 0.0, 0.65], physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(gravity[0], gravity[1], gravity[2], physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=self.timestep, numSolverIterations=num_solver_iterations,
                                    physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        logger.info("PyBullet connection established.")

    def _load_scene_assets(self):
        """Loads static scene assets like plane and table from config paths."""
        _assets_cfg = self.config.get('assets', {})
        plane_urdf_path = _assets_cfg.get('plane', "plane.urdf")
        table_urdf_path = _assets_cfg.get('table', "table/table.urdf")
        self.table_start_pos = _assets_cfg.get('table_start_pos', [0, 0, 0])

        self.plane_id = p.loadURDF(plane_urdf_path, physicsClientId=self.client)

        self.table_id = p.loadURDF(table_urdf_path, basePosition=self.table_start_pos, useFixedBase=True,
                                   physicsClientId=self.client)
        aabb = p.getAABB(self.table_id, -1, physicsClientId=self.client)
        self.table_height = aabb[1][2]
        logger.info(f"Scene assets loaded. Table height: {self.table_height:.3f}")

    def _load_robot(self):
        """Loads the robot URDF specified in the config."""
        _robot_cfg = self.config.get('robot', {})
        robot_urdf_path = _robot_cfg.get('urdf_path', "franka_panda/panda.urdf")
        robot_start_pos = [0, 0, self.table_height]
        robot_start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = -1
        try:
            self.robot_id = p.loadURDF(robot_urdf_path, robot_start_pos, robot_start_ori,
                                       useFixedBase=True,
                                       flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
                                       physicsClientId=self.client)
            logger.info(f"Loaded robot with ID: {self.robot_id} from {robot_urdf_path}")

        except Exception as e:
            logger.critical(f"LOAD FAILED: {e}"); self.close(); raise e

    def _load_robot_parameters_from_config(self):
        """Loads robot parameters from config and stores essential ones on self."""
        logger.info("Loading robot parameters from config...")
        _robot_cfg = self.config.get('robot', {})
        self.robot_type = _robot_cfg.get('type', 'panda')
        # ... (rest of robot param loading onto self attributes: joint names, link names, gains, gripper values, home pose, grasp offset etc.) ...
        _panda_cfg = _robot_cfg.get('panda', {})

        self.arm_joint_names = _panda_cfg.get('arm_joint_names', [f"panda_joint{i + 1}" for i in range(7)])
        self.finger_joint_names = _panda_cfg.get('finger_joint_names', ["panda_finger_joint1", "panda_finger_joint2"])
        self.preferred_ee_link_name = _panda_cfg.get('preferred_ee_link_name', "panda_hand")
        self.fallback_ee_link_name_1 = _panda_cfg.get('fallback_ee_link_name_1', "panda_link7")
        self.fallback_ee_link_name_2 = _panda_cfg.get('fallback_ee_link_name_2', "panda_link8")

        self.arm_max_forces = _panda_cfg.get('arm_max_forces', [100.0] * 7)
        self.arm_kp = _panda_cfg.get('arm_kp', [0.05] * 7)
        self.arm_kd = _panda_cfg.get('arm_kd', [1.0] * 7)

        self.gripper_open_value = _panda_cfg.get('gripper_open_value', 0.04)
        self.gripper_closed_value = _panda_cfg.get('gripper_closed_value', 0.025)
        self.gripper_max_force = _panda_cfg.get('gripper_max_force', 40)
        self.gripper_kp = _panda_cfg.get('gripper_kp', 0.2)
        self.gripper_kd = _panda_cfg.get('gripper_kd', 1.0)

        ik_solver_cfg = _panda_cfg.get('ik_solver', "IK_DLS")
        self.ik_solver = getattr(p, ik_solver_cfg, p.IK_DLS)  # Correct assignment
        self.home_pose_joints = _panda_cfg.get('home_pose_joints',
                                               [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])

        self.rest_poses_for_ik = self.home_pose_joints
        self.grasp_offset_in_hand_frame = _panda_cfg.get('grasp_offset_in_hand_frame', [0.0, 0.0, 0.065])

    def _initialize_robot_info_and_state(self):
        """ Finds indices, limits, resets robot to home pose, enables motors. """

        logger.info("Initializing robot info and state...")
        # 1. Find Indices using imported functions
        self.arm_joint_indices = find_joint_indices(self.robot_id, self.arm_joint_names, self.client)
        self.finger_indices = find_joint_indices(self.robot_id, self.finger_joint_names, self.client)

        # Find EE Link Index with fallbacks
        preferred_link_idx = find_link_index_safely(self.robot_id, self.preferred_ee_link_name, self.client)
        if preferred_link_idx is not None:
            self.ee_link_index, self.ee_link_name = preferred_link_idx, self.preferred_ee_link_name
        else:
            fallback_link_idx_1 = find_link_index_safely(self.robot_id,self.fallback_ee_link_name_1, self.client)
            if fallback_link_idx_1 is not None:
                self.ee_link_index, self.ee_link_name = fallback_link_idx_1, self.fallback_ee_link_name_1
            else:
                fallback_link_idx_2 = find_link_index_safely(self.robot_id, self.fallback_ee_link_name_2, self.client)
                if fallback_link_idx_2 is not None:
                    self.ee_link_index, self.ee_link_name = fallback_link_idx_2, self.fallback_ee_link_name_2
                else:
                    raise ValueError("Could not find suitable EE Link.")
        logger.info(f"  Using EE Link '{self.ee_link_name}' at index: {self.ee_link_index}")

        # 2. Get Arm Limits and Gripper Limits using imported function
        self.arm_limits = get_arm_kinematic_limits_and_ranges(self.robot_id, self.arm_joint_indices, self.client)
        logger.debug(f"  Arm Limits Extracted.")

        # 3. Reset Arm Joints State & Enable Motors (using self.home_pose_joints etc.)
        logger.info("  Initializing arm joints to home pose...")
        for i, idx in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, idx, self.home_pose_joints[i], 0.0, self.client)
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=self.home_pose_joints[i],
                                    force=self.arm_max_forces[i], positionGain=self.arm_kp[i],
                                    velocityGain=self.arm_kd[i], physicsClientId=self.client)



        # 4. Reset Gripper State & Enable Motors (using self attributes)
        if self.finger_indices:
            p.resetJointState(self.robot_id, self.finger_indices[0], self.gripper_open_value, 0.0, self.client)
            p.resetJointState(self.robot_id, self.finger_indices[1], self.gripper_open_value, 0.0, self.client)
            self._set_gripper_state('open', wait=False)  # Use internal method

        # 5. Settle Physics
        logger.info("  Stabilizing robot in home pose...")
        wait_steps(100,client=self.client)
        logger.info("Robot Initialized.")

    def _load_task_parameters(self):
        """Loads task parameters from config and stores essential ones."""
        _task_cfg = self.config.get('task', {})

        self.num_blocks = _task_cfg.get('num_blocks', 1)
        self.num_targets = _task_cfg.get('num_targets', 1)
        self.num_dump_locations = _task_cfg.get('num_dump_locations', 1)

        self.block_scale = _task_cfg.get('block_scale', 0.05)
        self.block_half_extents = [self.block_scale / 2.0] * 3
        self.z_hover_offset = _task_cfg.get('z_hover_offset', 0.15)
        self.grasp_clearance_above_top = _task_cfg.get('grasp_clearance_above_top', 0.08)
        self.place_clearance_above_top = _task_cfg.get('place_clearance_above_top', 0.09)
        self.grasp_offset_in_hand_frame = self.config.get('robot', {}).get('panda', {}).get(
            'grasp_offset_in_hand_frame', [0.0, 0.0, 0.065])

        _sim_cfg = self.config.get('simulation', {})
        self.primitive_max_steps = _sim_cfg.get('primitive_max_steps', 400)
        self.max_steps = _sim_cfg.get('max_episode_steps', 50 * self.num_blocks)
        self.gripper_wait_steps = _sim_cfg.get('gripper_wait_steps', 120)

        # Grasping and IK Thresholds
        self.pose_reached_threshold = _sim_cfg.get('pose_reached_threshold', 0.01)
        self.orientation_reached_threshold = _sim_cfg.get('orientation_reached_threshold', 0.1)

        # Reward Parameters
        self.goal_reward = _task_cfg.get('goal_reward', 1.0)
        self.step_penalty = _task_cfg.get('step_penalty', -0.01)
        self.move_fail_penalty = _task_cfg.get('move_fail_penalty', 0.005)

    def _load_and_instantiate_task(self, tasks_package, base_task_class=BaseTask):
        """
        Loads the task class specified in the config, validates it,
        instantiates it, and sets related environment parameters.

        Args:
            tasks_package: The imported 'tasks' package object.
            base_task_class: The BaseTask class for validation.

        Raises:
            ValueError: If the task cannot be loaded, validated, or instantiated.
        """
        if not tasks_package:
            raise ValueError("Task loading failed: 'tasks' package is not available or failed to import.")

        # --- 1. Get Task Configuration ---
        task_config_dict = self.config.get('task', {})
        task_class_name = task_config_dict.get('task_class_name', "BlockPlacementTask")  # Default

        # Add warning only if the default was used because the key was missing
        if task_class_name == "BlockPlacementTask" and not task_config_dict.get('task_class_name'):
            logger.warning(f"Warning: task_class_name not found in config, defaulting to {task_class_name}")

        # --- 2. Load Task Class Dynamically ---
        task_class = None
        # Heuristic: derive module name (e.g., BlockPlacementTask -> block_placement_task)
        module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', task_class_name).lower()
        # Append '_task' if needed (adjust based on your file naming convention)
        if not module_name.endswith("_task"): module_name += "_task"

        logger.debug(f"DEBUG: Attempting to load class '{task_class_name}' from module '{module_name}'.")
        try:
            # Get the specific task *module* from the tasks *package*
            task_module = getattr(tasks_package, module_name, None)
            if task_module:
                # Get the *class* from the task *module*
                task_class = getattr(task_module, task_class_name, None)

        except Exception as e:
            # Catch potential errors during getattr calls
            logger.error(f" An error occurred during dynamic task class loading: {e}")
            task_class = None  # Ensure class is None on error

        # --- 3. Validate and Instantiate ---
        if task_class and issubclass(task_class, base_task_class):
            logging.debug(f"DEBUG: Successfully retrieved and validated class: {task_class}")
            try:
                # Instantiate the task, passing the env instance and task-specific config
                self.task = task_class(self, task_config_dict)  # Pass the whole task dict
                logging.info(f"Instantiated Task Object: {self.task.__class__.__name__}")

                # --- 4. Set Environment Parameters from Task ---
                # These are determined *by the task* based on its config
                self.num_blocks = self.task.num_blocks
                self.num_locations = getattr(self.task, 'num_locations', self.num_blocks)
                self.num_dump_locations = getattr(self.task, 'num_dump_locations', 1)
                logging.debug(
                    f"DEBUG: Task parameters set: num_blocks={self.num_blocks}, num_locations={self.num_locations}")

            except Exception as e:
                # Catch errors during task __init__ or accessing its attributes
                raise ValueError(f"Failed to instantiate task '{task_class_name}' or access its parameters: {e}")

        else:
            # Construct error message for loading/validation failure
            error_msg = f"Could not find or validate task class '{task_class_name}'. "
            if not task_class:
                error_msg += f"Class not found (checked module '{module_name}' in 'tasks'). Check name and tasks/__init__.py. "
            else:
                error_msg += f"Class found but does not inherit from {base_task_class}. Check {base_task_class} import consistency. "
            raise ValueError(error_msg)


        logging.info("Task parameters loaded/updated.")

    def _load_colors(self):
        """Loads color definitions from config and slices based on num_blocks."""
        _color_cfg = self.config.get('colors', {})
        default_block_colors = [[0.9, 0.1, 0.1, 1.0], [0.1, 0.8, 0.1, 1.0], [0.1, 0.1, 0.9, 1.0], [0.9, 0.9, 0.1, 1.0],
                                [0.9, 0.1, 0.9, 1.0]]
        default_target_colors = [[0.9, 0.5, 0.5, 1.0], [0.5, 0.9, 0.5, 1.0], [0.5, 0.5, 0.9, 1.0], [0.9, 0.9, 0.5, 1.0],
                                 [0.9, 0.5, 0.9, 1.0]]
        all_block_colors = _color_cfg.get('block_rgba', default_block_colors)
        all_target_colors = _color_cfg.get('target_rgba', default_target_colors)

        if self.num_blocks <= len(all_block_colors):
            self.block_colors_rgba = all_block_colors[:self.num_blocks]
            self.target_colors_rgba = all_target_colors[:self.num_blocks]
        else:
            logging.warning(f"Warning: num_blocks > defined colors. Colors will repeat.")
            self.block_colors_rgba = (all_block_colors * (self.num_blocks // len(all_block_colors) + 1))[
                                     :self.num_blocks]
            self.target_colors_rgba = (all_target_colors * (self.num_blocks // len(all_target_colors) + 1))[
                                      :self.num_blocks]
        logging.info("Colors loaded and sliced.")

    def _setup_rl_interface(self):
        """Sets up action space and observation space based on loaded task config."""
        logging.info("Setting up RL interface...")
        # --- Get Task Dimensions from Config ---
        # These should have been loaded into self.config by _load_and_merge_configs
        _task_cfg = self.config.get('task', {})

        # Get num_blocks (should ideally already be on self from _load_task_parameters)
        # If not, load it here with a default.
        if not hasattr(self, 'num_blocks'):
            self.num_blocks = _task_cfg.get('num_blocks', 3)  # Default needed if not set before

        # Get num_locations from task config, default to num_blocks if not specified
        self.num_locations = _task_cfg.get('num_locations', self.num_blocks)

        # Get num_dump_locations from task config
        if not hasattr(self, 'num_dump_locations'):  # Check if already set
            self.num_dump_locations = _task_cfg.get('num_dump_locations', 1)

        # --- Action Space ---
        # Calculate total actions based on loaded dimensions
        self.num_actions = self.num_blocks + self.num_locations + self.num_dump_locations
        self.action_space = spaces.Discrete(self.num_actions)
        logging.info(
            f"  Action Space: Discrete({self.num_actions}) ({self.num_blocks} Pick, {self.num_locations} PlaceTarget, {self.num_dump_locations} PlaceDump)")

        # --- Observation Space ---
        # Load camera parameters for image size
        _cam_cfg = self.config.get('camera', {})
        self.image_size = _cam_cfg.get('image_size', 84)  # Default image size
        # Define observation space (currently image-based)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.image_size, self.image_size, 3),
            dtype=np.uint8
        )
        logging.info(f"  Observation Space: Box(shape=({self.image_size}, {self.image_size}, 3))")

        logging.info("RL interface setup complete.")

    # ==================================================================
    # --- Reset ---
    # ==================================================================


    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new initial state.

        - Removes old objects and constraints.
        - Resets robot to home pose with gripper open.
        - Calls the task's reset_task_scenario() method to set up the specific task
          (define targets, goals, spawn blocks/visuals).
        - Lets the simulation settle.
        - Returns the initial observation and info dict.
        """
        super().reset(seed=seed)  # Handles seeding via Gymnasium wrapper
        self.current_steps = 0
        self.held_object_id = None
        self.held_object_idx = None
        self.goal_config = {}  # Reset goal config

        # --- 1. Cleanup from previous episode ---
        if self.grasp_constraint_id is not None:
            try:
                p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
            except Exception:
                pass  # Ignore if removal fails
            self.grasp_constraint_id = None

        # Remove old blocks and targets
        for body_id_list in [self.block_ids, self.target_ids]:
            for body_id in body_id_list:
                try:
                    p.removeBody(body_id, physicsClientId=self.client)
                except Exception:
                    pass
        self.block_ids = []
        self.target_ids = []
        # ------------------------------------

        # --- 2. Reset Robot Pose to Home ---
        logging.debug("Resetting robot to home pose...")
        if hasattr(self, 'arm_joint_indices') and self.arm_joint_indices:  # Check if initialized
            for i, idx in enumerate(self.arm_joint_indices):
                p.resetJointState(self.robot_id, idx, self.home_pose_joints[i], 0.0, self.client)
                # Re-apply motor control to hold pose
                p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL,
                                        targetPosition=self.home_pose_joints[i],
                                        force=self.arm_max_forces[i],
                                        positionGain=self.arm_kp[i], velocityGain=self.arm_kd[i],
                                        physicsClientId=self.client)
            # Ensure gripper is open
            self._set_gripper_state("open", wait=False)
            wait_steps(50, self.client, timestep=self.timestep, use_gui=self.use_gui)  # Short wait for stability
        else:
            logging.warning("Warning: Robot arm/gripper indices not available during reset. Skipping pose reset.")
        # --------------------------------

        # --- 3. Delegate Task-Specific Setup ---
        if self.task is None:
            raise RuntimeError("Task object (self.task) not initialized before reset. Check __init__.")

        logging.debug("Calling task reset_task_scenario...")
        task_info = None  # Initialize task_info
        try:
            # The task will define targets, goals, and trigger spawning
            task_info = self.task.reset_task_scenario()  # Store the result
            logging.debug("Task scenario reset complete.")
        except Exception as e:
            logging.critical(f"FATAL: Error during task reset_task_scenario: {e}")
            import traceback
            traceback.print_exc()
            self.close()
            raise RuntimeError(f"Failed to reset task scenario: {e}")
        # -----------------------------------------

        # --- 4. Settle Simulation ---
        logging.debug("Waiting for objects to settle after task reset...")
        wait_steps(150, self.client, timestep=self.timestep, use_gui=self.use_gui)
        # -------------------------

        # --- 5. Get Initial Observation and Info ---
        observation = self._get_obs()
        # Get base info, then merge task-specific info if provided
        info = self._get_info()
        if isinstance(task_info, dict):
            info.update(task_info)  # Add info returned by the task

        logging.debug("Environment reset finished.")
        # --- ADD THIS LINE ---
        return observation, info
            # ---------------------

    # ==================================================================
    # --- Core Step and Primitive Execution ---
    # ==================================================================

    def step(self, action_):

        success_ = self._execute_primitive(action_) # Execute skill

        observation_ = self._get_obs()
        terminated_ = self.task.check_goal()
        reward_ = self.goal_reward if terminated_ else self.step_penalty
        self.current_steps += 1
        truncated_ = self.current_steps >= self.max_steps

        # Add penalties based on primitive failure
        if not success_:
            reward_ += self.move_fail_penalty

        info_ = self._get_info()
        info_['primitive_success'] = success_

        return observation_, reward_, terminated_, truncated_, info_

    def _execute_primitive(self, action_index):
        """ Executes the high-level skill based on action_index. Returns True if skill sequence succeeds. """
        logging.debug(f"\n--- Executing Action Index: {action_index} ---")
        ori_down = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])

        try:
            # --- Action: Pick_Block(block_idx) ---
            if 0 <= action_index < self.num_blocks:
                block_idx_to_pick = action_index
                logging.debug(f"Attempting Pick_Block({block_idx_to_pick})")
                # Check preconditions: not holding, valid index
                if self.held_object_id is not None:
                    logging.debug("  Failure: Already holding.")
                    return False
                if block_idx_to_pick >= len(self.block_ids):
                    logging.debug(f"  Failure: Invalid block index {block_idx_to_pick}")
                    return False
                block_id = self.block_ids[block_idx_to_pick]

                try:  # Get block pose AND ORIENTATION
                    block_pos, block_orn_quat = p.getBasePositionAndOrientation(block_id, physicsClientId=self.client)
                    block_euler = p.getEulerFromQuaternion(block_orn_quat)
                    logging.debug(
                        f"  Block {block_idx_to_pick} Pose: Pos={np.round(block_pos, 3)}, Euler={np.round(block_euler, 2)}")
                except Exception as e:
                    logging.warning( f"  Failure: Cannot get pose for block {block_id}. {e}")
                    return False

                # --- Calculate Target Orientation based on Block Yaw ---
                block_yaw = block_euler[2]
                target_ori = p.getQuaternionFromEuler([np.pi, 0.0, block_yaw])
                logging.debug(f"Target Grasp Ori (Euler): {np.round(p.getEulerFromQuaternion(target_ori), 2)}")

                # Calculate poses relative to current block pos and TOP surface
                object_top_z = block_pos[2] + self.block_half_extents[2]
                pre_grasp_pos_z = object_top_z + self.z_hover_offset
                grasp_pos_z = object_top_z + self.grasp_clearance_above_top  # Using clearance from top
                lift_pos_z = object_top_z + self.z_hover_offset + 0.05  # Lift slightly higher
                pre_grasp_pos = [block_pos[0], block_pos[1], pre_grasp_pos_z]
                grasp_pos = [block_pos[0], block_pos[1], grasp_pos_z]
                lift_pos = [block_pos[0], block_pos[1], lift_pos_z]

                # --- Pick Sequence ---
                logging.debug("  1. Opening gripper (just in case)...")
                self._set_gripper_state("open", wait=True)

                logging.debug("  2. Moving above block (adjusted ori)...")
                # Move to pre-grasp using the block-aligned orientation
                if not self._move_ee_to_pose(pre_grasp_pos, target_ori):
                    logging.warning("  Failure: Could not reach pre-grasp pose.")
                    return False  # Give up if pre-grasp fails

                logging.debug("  3. Moving down to grasp (adjusted ori)...")
                # Move down using the block-aligned orientation
                if not self._move_ee_to_pose(grasp_pos, target_ori):
                    logging.warning("  Move down failed. Aborting pick.")
                    self._set_gripper_state("open", wait=False)
                    current_pos, current_ori = self._get_ee_pose()
                    if current_pos:  # Try to recover upwards
                        recover_pos = [current_pos[0], current_pos[1], current_pos[2] + 0.05]
                        # Use last target orientation or current if available for recovery
                        self._move_ee_to_pose(recover_pos, current_ori if current_ori else target_ori)
                    return False

                # *** Move down succeeded, now close the gripper ***
                logging.debug("  4. Closing gripper...")
                self._set_gripper_state("close", wait=True)

                # Optional: Check if gripper closed sufficiently
                if self.finger_indices:
                    states = p.getJointStates(self.robot_id, self.finger_indices, self.client)
                    # Check if fingers are near the *intended* closed value
                    closed_check_val = self.gripper_closed_value + 0.01  # Allow tolerance
                    finger1_val = states[0][0]
                    finger2_val = states[1][0]
                    logging.debug(
                        f"  Gripper state after close command: {finger1_val:.4f}, {finger2_val:.4f} (Target: {self.gripper_closed_value:.4f})")
                    # Check if BOTH fingers are sufficiently closed
                    if finger1_val > closed_check_val or finger2_val > closed_check_val:
                        logging.warning(f"  WARNING: Gripper did not close sufficiently. Aborting grasp.")
                        self._set_gripper_state("open", wait=False)
                        self._move_ee_to_pose(pre_grasp_pos, target_ori)  # Move back up
                        return False

                logging.debug("  5. Attaching object...")
                try:
                    # Calculate relative orientation at grasp moment
                    ee_state = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.client)
                    hand_pos_w, hand_ori_w = ee_state[4], ee_state[5]  # Use world frame pos/ori
                    obj_pos_w, obj_ori_w = p.getBasePositionAndOrientation(block_id, physicsClientId=self.client)
                    inv_hand_pos_w, inv_hand_ori_w = p.invertTransform(hand_pos_w, hand_ori_w)

                    _, obj_ori_in_hand = p.multiplyTransforms(inv_hand_pos_w, inv_hand_ori_w, obj_pos_w, obj_ori_w)

                    self.grasp_constraint_id = p.createConstraint(
                        parentBodyUniqueId=self.robot_id,
                        parentLinkIndex=self.ee_link_index,
                        childBodyUniqueId=block_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=self.grasp_offset_in_hand_frame,
                        childFramePosition=[0, 0, 0],
                        parentFrameOrientation=obj_ori_in_hand,
                        childFrameOrientation=[0, 0, 0, 1],
                        physicsClientId=self.client
                    )

                    if self.grasp_constraint_id < 0: raise Exception("createConstraint failed")
                    logging.debug(f"  Constraint created: {self.grasp_constraint_id}")
                except Exception as e:
                    logging.error(f"  Failure: Error creating constraint: {e}")
                    self._set_gripper_state("open", wait=False)
                    return False
                wait_steps(60, self.client, timestep=self.timestep, use_gui=self.use_gui)
                self.held_object_id = block_id
                self.held_object_idx = block_idx_to_pick

                logging.debug("  6. Lifting block...")
                if not self._move_ee_to_pose(lift_pos, target_ori):
                    # Lift failure cleanup
                    logging.warning("  Lift failed, releasing constraint and object.")

                    # 1. Attempt to remove constraint
                    if self.grasp_constraint_id is not None:
                        try:
                            p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
                        except:
                            pass

                    # 2. Update internal state
                    self.grasp_constraint_id = None
                    self.held_object_id = None
                    self.held_object_idx = None

                    # 3. Open Gripper
                    self._set_gripper_state('open', wait=False)

                    # 4. Return Failure
                    return False

                logging.debug("  Pick sequence successful.")
                return True


            # --- Action: Place_Target / Place_Dump ---
            elif self.num_blocks <= action_index <= self.num_blocks + self.num_locations:

                is_dump = (action_index == self.num_blocks + self.num_locations)
                target_loc_idx = action_index - self.num_blocks if not is_dump else -1
                loc_name = "Dump" if is_dump else f"Target({target_loc_idx})"
                logging.debug(f"Attempting Place_{loc_name}")

                if self.held_object_id is None:
                    logging.warning("  Failure: Not holding object.")
                    return False

                target_pos_table = self.dump_location_pos if is_dump else self.target_locations_pos[target_loc_idx]
                target_base_z = self.table_height + self.block_half_extents[2]

                pre_place_pos_z = target_base_z + self.block_half_extents[2] + self.z_hover_offset
                pre_place_pos = [target_pos_table[0], target_pos_table[1], pre_place_pos_z]

                place_pos_z = target_base_z + self.block_half_extents[2] + self.place_clearance_above_top
                place_pos = [target_pos_table[0], target_pos_table[1], place_pos_z]

                post_place_pos_z = pre_place_pos_z
                post_place_pos = [target_pos_table[0], target_pos_table[1], post_place_pos_z]

                logging.debug(f"  1. Moving above {loc_name}...")
                if not self._move_ee_to_pose(pre_place_pos, ori_down):
                    return False

                logging.debug(f"  2. Moving down to {loc_name}...")
                place_move_success = self._move_ee_to_pose(place_pos, ori_down)  # Uses new place_pos
                if not place_move_success:
                    logging.warning(f"  Warning: Did not fully reach {loc_name} pose.")

                logging.debug("  3. Releasing object...")
                if self.grasp_constraint_id is not None:
                    try:
                        p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
                    except Exception as e:
                        logging.debug(f"  Warning: Failed removing constraint {self.grasp_constraint_id}: {e}")
                    self.grasp_constraint_id = None


                self._set_gripper_state("open", wait=True)

                self.held_object_id = None
                self.held_object_idx = None

                wait_steps(50, self.client, timestep=self.timestep, use_gui=self.use_gui)

                logging.debug("  4. Moving arm up...")
                self._move_ee_to_pose(post_place_pos, ori_down)
                logging.debug(f"  {loc_name} sequence finished.")
                return True

            else:  # Unknown action
                logging.warning(f"Warning: Unknown action index {action_index}")
                return False

        except Exception as e:  # Catch any unexpected errors in primitive execution
            logging.error(f"!! Error during primitive execution for action {action_index}: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            if self.grasp_constraint_id is not None:
                try:
                    p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
                except Exception:
                    pass
            self.grasp_constraint_id = None
            self.held_object_id = None
            self.held_object_idx = None

            self._set_gripper_state("open", wait=False)

            return False  # Primitive failed

    # ==================================================================
    # --- Robot Control / Helper Methods ---
    # ==================================================================

    def _set_gripper_state(self, state: str, wait: bool = True):
        """
        Sets gripper state to 'open' or 'close' using position control.

        Args:
            state (str): Desired state, either 'open' or 'close'.
            wait (bool): Whether to wait for the action to complete using simulation steps.

        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        logging.debug(f"Commanding gripper {state.upper()}")
        if state == 'open':
            target_value = self.gripper_open_value
        elif state == 'close':
            target_value = self.gripper_closed_value
        else:
            logging.error(f"Error: Invalid gripper state '{state}'. Use 'open' or 'close'.")
            return False

        if not self.finger_indices:
            logging.error("Error: Gripper finger indices not set.")
            return False
        if len(self.finger_indices) != 2:
            logging.error(f"Error: Expected 2 finger indices, found {len(self.finger_indices)}.")
            return False

        try:
            # Directly issue the command to control both fingers
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.finger_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[target_value] * 2,  # Command both fingers to the same value
                forces=[self.gripper_max_force] * 2,
                positionGains=[self.gripper_kp] * 2,
                velocityGains=[self.gripper_kd] * 2,
                physicsClientId=self.client
            )
            # Wait for action completion if requested
            if wait:
                wait_steps(self.gripper_wait_steps, self.client, timestep=self.timestep, use_gui=self.use_gui)
            return True  # Command sent successfully
        except Exception as e:
            logging.error(f"Error during setJointMotorControlArray for gripper: {e}")
            return False


    def _move_ee_to_pose(self, target_pos, target_ori, max_steps_override=None):
        """ Calculates IK and commands arm to target pose. Returns True if successful. """
        self._last_ik_failure = False # Reset IK failure flag for this attempt

        if self.arm_limits is None:
            logging.error("Error: Arm limits not set.")
            return False

        ll, ul, jr = self.arm_limits

        start_ik_time = time.time()
        joint_poses = p.calculateInverseKinematics(
            self.robot_id, self.ee_link_index, target_pos, target_ori,
            lowerLimits=ll, upperLimits=ul, jointRanges=jr,
            restPoses=self.rest_poses_for_ik, solver=self.ik_solver,
            maxNumIterations=200, residualThreshold=1e-4, physicsClientId=self.client
        )
        ik_time = time.time() - start_ik_time

        valid_solution = joint_poses is not None and len(joint_poses) >= len(self.arm_joint_indices)

        if valid_solution:
            arm_joint_poses = joint_poses[:len(self.arm_joint_indices)]
            logging.debug(f"  ----> IK Succeeded! (Took {ik_time:.4f} s)")
            logging.debug("        Commanding Arm Motion...")

            p.setJointMotorControlArray(self.robot_id, self.arm_joint_indices, p.POSITION_CONTROL,
                                        targetPositions=arm_joint_poses,
                                        forces=self.arm_max_forces, positionGains=self.arm_kp,
                                        velocityGains=self.arm_kd, physicsClientId=self.client)

            max_steps = max_steps_override if max_steps_override is not None else self.primitive_max_steps
            wait_steps(max_steps, self.client, timestep=self.timestep, use_gui=self.use_gui)

            # Check final pose
            try:
                final_ee_state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True, physicsClientId=self.client)
                final_ee_pos, final_ee_ori = final_ee_state[4:6]
                final_dist = np.linalg.norm(np.array(final_ee_pos) - np.array(target_pos))

                ori_diff = p.getDifferenceQuaternion(target_ori, final_ee_ori)
                _, ori_angle = p.getAxisAngleFromQuaternion(ori_diff)

                pos_ok = final_dist < self.pose_reached_threshold
                ori_ok = abs(ori_angle) < self.orientation_reached_threshold

                if pos_ok and ori_ok:
                    logging.debug(f"        SUCCESS: Reached Target Pose! Dist: {final_dist:.4f}, Ori angle err: {abs(ori_angle):.4f}")
                else:
                    logging.debug(f"        FAILURE: Did not reach Target Pose. Dist: {final_dist:.4f} (OK={pos_ok}), Ori angle err: {abs(ori_angle):.4f} (OK={ori_ok})")
                return pos_ok and ori_ok

            except Exception as e:
                logging.debug(f"        Error checking final pose: {e}") # Added indent
                return False
        else:
            logging.warning(f"  ----> IK Failed! (Took {ik_time:.4f} s)")
            self._last_ik_failure = True # Set flag if IK itself failed
            return False

    def _get_ee_pose(self):
        """ Gets the current world pose of the end-effector link. """
        try:
            link_state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True,
                                        physicsClientId=self.client)
            return link_state[4], link_state[5]  # world pos, world orn
        except Exception as e:
            logging.error(f"Error getting EE pose: {e}")
            return None, None

    # ==================================================================
    # --- Environment Specific Methods (Observation, Goal Check etc.) ---
    # ==================================================================#

    def _place_target_visuals(self):
        """Places visual markers (plates) at the target locations."""
        self.target_ids = []  # Clear previous visuals
        if not hasattr(self, 'target_locations_pos') or not self.target_locations_pos:
            logging.warning("Warning: Target locations not defined, cannot place visuals.")
            return
        if not hasattr(self, 'goal_config') or not self.goal_config:
            logging.warning("Warning: Goal config not defined, cannot color visuals correctly.")
            # Decide on fallback behavior? e.g., use sequential colors?
            # return # Or proceed with default colors

        logging.debug(f"Placing {len(self.target_locations_pos)} target visuals...")

        # We need to map target_location_index back to the original block_index
        # to get the correct color based on the goal config.
        # Create inverse mapping: required_block_idx -> target_loc_idx
        block_to_target_map = {v: k for k, v in self.goal_config.items()}

        plate_half_extents = [0.04, 0.04, 0.0005]  # Thin plate
        plate_center_z = self.table_height + plate_half_extents[2] + 0.0001

        for i in range(len(self.target_locations_pos)):  # Iterate through defined location positions
            target_pos = self.target_locations_pos[i]  # Get the [x,y,z] for this location index 'i'

            # Find which original block index is assigned to this location index 'i'
            assigned_block_idx = -1
            for block_idx, loc_idx in self.goal_config.items():
                if loc_idx == i:
                    assigned_block_idx = block_idx
                    break

            if assigned_block_idx != -1:
                target_color_rgba = self.target_colors_rgba[assigned_block_idx % len(self.target_colors_rgba)]
                try:
                    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=plate_half_extents,
                                                 rgbaColor=target_color_rgba, physicsClientId=self.client)
                    plate_id = p.createMultiBody(
                        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_id,
                        basePosition=[target_pos[0], target_pos[1], plate_center_z],
                        baseOrientation=[0, 0, 0, 1], physicsClientId=self.client
                    )
                    if plate_id >= 0:
                        self.target_ids.append(plate_id)
                    else:
                        logging.warning(f"Warning: Failed to create target plate visual for loc idx {i}")
                except Exception as e:
                    logging.error(f"Error creating target plate visual for loc idx {i}: {e}")
            else:
                logging.warning(f"Warning: No block assigned to target location index {i} in goal_config. Skipping visual.")

    def _spawn_blocks(self, num_blocks_to_spawn):
        """Spawns the required number of blocks in valid random positions."""
        self.block_ids = []  # Clear previous blocks
        logging.debug(f"Spawning {num_blocks_to_spawn} blocks...")
        if not hasattr(self, 'spawn_area_bounds'):
            self.spawn_area_bounds = self.task.define_spawn_area()

        available_spawn_locations = self._get_valid_spawn_positions(num_blocks_to_spawn)
        if len(available_spawn_locations) < num_blocks_to_spawn:
            raise RuntimeError(
                f"Not enough valid spawn locations ({len(available_spawn_locations)}) for {num_blocks_to_spawn} blocks.")

        for i in range(num_blocks_to_spawn):
            spawn_pos_xy = available_spawn_locations[i]
            spawn_z = self.table_height + self.block_half_extents[2] + 0.001
            block_start_pos = [spawn_pos_xy[0], spawn_pos_xy[1], spawn_z]
            block_start_orientation = p.getQuaternionFromEuler([0, 0, self.np_random.uniform(0, 2 * np.pi)])

            try:
                block_id = p.loadURDF("cube.urdf", block_start_pos, block_start_orientation,
                                      globalScaling=self.block_scale, physicsClientId=self.client)
                if block_id < 0: raise Exception("p.loadURDF failed for block")

                block_color_rgba = self.block_colors_rgba[i % len(self.block_colors_rgba)]
                p.changeVisualShape(block_id, -1, rgbaColor=block_color_rgba, physicsClientId=self.client)
                p.changeDynamics(block_id, -1, mass=0.1, lateralFriction=0.6, physicsClientId=self.client)
                self.block_ids.append(block_id)
            except Exception as e:
                logging.error(f"Error loading block {i}: {e}")
                raise e

    # Inside PhysicsBlockRearrangementEnv class

    def _get_valid_spawn_positions(self, num_required):
        """Generates valid spawn positions avoiding targets and dump locations."""
        valid_positions = []
        min_dist_sq = (self.block_scale * 1.2) ** 2  # Min dist between spawned blocks

        target_dump_poses_xy = [[loc[0], loc[1]] for loc in self.target_locations_pos]
        # Only add dump location if it exists and is not empty
        if hasattr(self, 'dump_location_pos') and self.dump_location_pos:
            # Check if dump_location_pos is a list of lists (multiple dumps) or just one list
            if isinstance(self.dump_location_pos[0], list):  # Multiple dump locs
                dump_poses_xy = [[loc[0], loc[1]] for loc in self.dump_location_pos]
                target_dump_poses_xy.extend(dump_poses_xy)
            elif len(self.dump_location_pos) >= 2:  # Single dump loc [x, y, z]
                target_dump_poses_xy.append([self.dump_location_pos[0], self.dump_location_pos[1]])
            else:
                logging.warning("Warning: dump_location_pos format unexpected, skipping dump location check.")

        attempts = 0
        max_attempts = num_required * 100

        while len(valid_positions) < num_required and attempts < max_attempts:
            attempts += 1
            x = self.np_random.uniform(self.spawn_area_bounds[0], self.spawn_area_bounds[1])
            y = self.np_random.uniform(self.spawn_area_bounds[2], self.spawn_area_bounds[3])
            candidate_pos = [x, y]

            # Check distance to other blocks spawned in this reset
            too_close_to_spawn = any(((pos[0] - x) ** 2 + (pos[1] - y) ** 2 < min_dist_sq) for pos in valid_positions)
            if too_close_to_spawn: continue

            # Check distance to targets and dump locations
            too_close_to_target_or_dump = any(
                ((target_pos[0] - x) ** 2 + (target_pos[1] - y) ** 2 < (self.block_scale * 1.5) ** 2) for target_pos in
                target_dump_poses_xy)
            if too_close_to_target_or_dump: continue

            # If all checks pass, add the position
            valid_positions.append(candidate_pos)

        if len(valid_positions) < num_required:
            logging.warning(f"Warning: Only found {len(valid_positions)}/{num_required} valid spawn locations.")
        return valid_positions

    def _get_obs(self):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(self.camera_target_pos, self.camera_distance, self.camera_yaw, self.camera_pitch, 0, 2, self.client)
        proj_matrix = p.computeProjectionMatrixFOV(60, float(self.image_size)/self.image_size, 0.1, 2.0, self.client)
        try:
            (_, _, px, _, _) = p.getCameraImage(self.image_size, self.image_size, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.client)
            rgb_array = np.array(px, dtype=np.uint8)[:, :, :3]
            return rgb_array
        except Exception as e:
             logging.error(f"Error getting camera image: {e}. Returning blank image.")
             return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def _get_info(self):
        info = {'held_object_idx': self.held_object_idx,'current_steps': self.current_steps}
        return info

    def _check_goal(self):
        if self.held_object_id is not None: return False
        on_target_count = 0
        goal_dist_threshold = 0.04
        for target_loc_idx, required_block_idx in self.goal_config.items():
            if required_block_idx < len(self.block_ids):
                block_id = self.block_ids[required_block_idx]
                target_pos = self.target_locations_pos[target_loc_idx]
                try:
                    current_pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self.client)
                    dist_xy = np.linalg.norm(np.array(current_pos[:2]) - np.array(target_pos[:2]))
                    on_surface = abs(current_pos[2] - (self.table_height + self.block_half_extents[2])) < 0.02
                    if dist_xy < goal_dist_threshold and on_surface:
                        on_target_count += 1
                except Exception: return False
        return on_target_count == len(self.goal_config)

    def render(self, mode='human'):
        if mode == 'rgb_array': return self._get_obs()
        elif mode == 'human':
            if self.use_gui:
                time.sleep(0.01)
                return None
        else: return super(PhysicsBlockRearrangementEnv, self).render()

    def close(self):
        if hasattr(self, 'client') and self.client >= 0:
            try:
                if p.isConnected(physicsClientId=self.client):
                     p.disconnect(physicsClientId=self.client)
            except Exception as e: logging.error(f"Error disconnecting PyBullet: {e}")
            self.client = -1


if __name__ == '__main__':
    env = None
    try:
        # Increase clearance for testing
        env = PhysicsBlockRearrangementEnv(use_gui=False, render_mode='human')
        # You might need to adjust self.grasp_clearance_above_top inside __init__ if needed
        # env.grasp_clearance_above_top = 0.04 # Example override if needed

        for episode in range(3):
            print(f"\n--- Episode {episode+1} ---")
            obs, info = env.reset()
            terminated, truncated = False, False
            step = 0
            action_sequence = [0, 4, 1, 3] # Pick 0, Dump, Pick 1, Place Target 1
            action_idx = 0
            while not terminated and not truncated:
                if action_idx < len(action_sequence):
                     action = action_sequence[action_idx]
                     action_idx += 1
                else:
                     print("Sequence finished. Ending episode.")
                     break
                print(f"\nStep: {step}, Executing Action: {action}")
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"  -> Reward: {reward:.3f}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
                env.render()
                step += 1
                if terminated: print("--- Goal Reached! ---")
                if truncated: print("--- Max steps reached. ---")
                time.sleep(0.2)
            print(f"Episode {episode+1} finished after {step} steps.")
            time.sleep(1)
    except Exception as main_e:
         print(f"An error occurred in the main execution: {main_e}")
         import traceback
         traceback.print_exc()
    finally:
        if env: env.close()
        print("Environment closed.")