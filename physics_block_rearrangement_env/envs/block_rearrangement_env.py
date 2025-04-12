import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import os
from pathlib import Path
import yaml
import math

from physics_block_rearrangement_env.utils.robot_utils import *


class PhysicsBlockRearrangementEnv(gym.Env):
    """
    Gymnasium environment for block rearrangement using PyBullet.
    Integrates robust Panda IK/Gripper control.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, use_gui=False,
                 task_config_file="../configs/task_config.yaml", # Default task name
                 base_config_file="../configs/base_config.yaml"):
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

        # --- Read Task Name From Loaded Config ---
        # Store it after loading for reference or logging
        self.task_name = self.config.get('task', {}).get('name', task_config_file)
        print(f"Initializing environment for task: '{self.task_name}' (from {task_config_file})")

        # --- Setup Physics, Scene, Robot (using base config) ---
        self.client = -1
        self._setup_pybullet_connection() # Connect and set basic physics params
        self._load_scene_assets() # Uses self.config for paths
        # self._load_robot_parameters() # Load params needed for loading/init
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
        self.spawn_area_bounds = self._define_spawn_area()  # Define spawn area

        print(f"Environment Initialized.")

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
                    print(f"Error loading base config {base_config_path}: {e}")
        else:
            print(f"Warning: Base config file not found: {base_config_path}")

        # Load Task Config using the provided filename
        task_config = {}
        if task_config_path.exists():
            with open(task_config_path, 'r') as f:
                try:
                    task_config = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    print(f"Error loading task config {task_config_path}: {e}")
        else:
            # Try looking in the main config dir as a fallback? Optional.
            task_config_path_alt = config_dir / task_config_file
            if task_config_path_alt.exists():
                print(f"Note: Task config '{task_config_file}' not found in tasks/, loading from {config_dir}")
                task_config_path = task_config_path_alt
                with open(task_config_path, 'r') as f:
                    try:
                        task_config = yaml.safe_load(f) or {}
                    except yaml.YAMLError as e:
                        print(f"Error loading task config {task_config_path}: {e}")
            else:
                print(f"Warning: Task config file not found: {task_config_path} or {task_config_path_alt}")

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
        print(f"Configuration loaded and merged using task file: '{task_config_file}'.")

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
        print("PyBullet connection established.")

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
        print(f"Scene assets loaded. Table height: {self.table_height:.3f}")

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
            print(f"Loaded robot with ID: {self.robot_id} from {robot_urdf_path}")
            self.arm_joint_names = _robot_cfg.get('panda', {}).get('arm_joint_names',
                                                                   ["panda_joint1", "panda_joint2", "panda_joint3",
                                                                    "panda_joint4", "panda_joint5", "panda_joint6",
                                                                    "panda_joint7"])

            self.finger_joint_names = _robot_cfg.get('panda', {}).get('finger_joint_names',["panda_finger_joint1",
                                                                                            "panda_finger_joint2"])

            self.preferred_ee_link_name =_robot_cfg.get('panda', {}).get('preferred_ee_link_name', "panda_hand")
            self.fallback_ee_link_name_1 = _robot_cfg.get('panda', {}).get('fallback_ee_link_name_1', "panda_lin7")
            self.fallback_ee_link_name_2 = _robot_cfg.get('panda', {}).get('fallback_ee_link_name_2', "panda_lin8")

        except Exception as e:
            print(f"LOAD FAILED: {e}"); self.close(); raise e

    def _load_robot_parameters(self):
        """Loads robot parameters from config and stores essential ones."""
        # Already loaded most into self attributes in __init__ for clarity
        # This method could potentially load less frequently accessed ones if needed
        pass  # Most are already loaded into self attributes

    def _initialize_robot_info_and_state(self):
        """ Finds indices, limits, resets robot to home pose, enables motors. """

        print("Initializing robot info and state...")
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
        print(f"  Using EE Link '{self.ee_link_name}' at index: {self.ee_link_index}")

        _panda_cfg = self.config.get('robot', {}).get('panda', {})  # Get panda params again for gains

        # 2. Get Arm Limits and Gripper Limits using imported function
        self.arm_limits = get_arm_kinematic_limits_and_ranges(self.robot_id, self.arm_joint_indices, self.client)
        self.gripper_open_value = _panda_cfg.get('gripper_open_value')
        self.gripper_closed_value = _panda_cfg.get('gripper_closed_value')
        print(f"  Arm Limits Extracted.")

        # 3. Reset Arm Joints State & Enable Motors (using self.home_pose_joints etc.)
        print("  Initializing arm joints to home pose...")
        self.arm_kp = _panda_cfg.get('arm_kp', [0.05] * 7)
        self.arm_kd = _panda_cfg.get('arm_kd', [1.0] * 7)
        self.arm_max_forces = _panda_cfg.get('arm_max_forces', [100.0] * 7)
        self.home_pose_joints = _panda_cfg.get('home_pose_joints', [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
        self.rest_poses_for_ik = self.home_pose_joints

        for i, idx in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, idx, self.home_pose_joints[i], 0.0, self.client)
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=self.home_pose_joints[i],
                                    force=self.arm_max_forces[i], positionGain=self.arm_kp[i],
                                    velocityGain=self.arm_kd[i], physicsClientId=self.client)

        if _panda_cfg.get('ik_solver', 'IK_DLS') == 'IK_DLS':
            self.ik_solver = p.IK_DLS
        else:
            raise NotImplementedError("IK SOLVER: Only IDK_DLS is supported yet!")

        # 4. Reset Gripper State & Enable Motors (using self attributes)
        print("  Initializing gripper to OPEN state...")
        self.gripper_kp = _panda_cfg.get('gripper_kp', 0.2)
        self.gripper_kd = _panda_cfg.get('gripper_kd', 1.0)
        self.gripper_max_force = _panda_cfg.get('gripper_max_force', 50.0)

        if self.finger_indices:
            p.resetJointState(self.robot_id, self.finger_indices[0], self.gripper_open_value, 0.0, self.client)
            p.resetJointState(self.robot_id, self.finger_indices[1], self.gripper_open_value, 0.0, self.client)
            self._set_gripper_state('open', wait=False)  # Use internal method

        # 5. Settle Physics
        print("  Stabilizing robot in home pose...")
        wait_steps(100,client=self.client)
        print("Robot Initialized.")

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



        print("Task parameters loaded/updated.")

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
            print(f"Warning: num_blocks > defined colors. Colors will repeat.")
            self.block_colors_rgba = (all_block_colors * (self.num_blocks // len(all_block_colors) + 1))[
                                     :self.num_blocks]
            self.target_colors_rgba = (all_target_colors * (self.num_blocks // len(all_target_colors) + 1))[
                                      :self.num_blocks]
        print("Colors loaded and sliced.")

    def _setup_rl_interface(self):
        """Sets up action space and observation space based on loaded task config."""
        print("Setting up RL interface...")
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
        print(
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
        print(f"  Observation Space: Box(shape=({self.image_size}, {self.image_size}, 3))")

        print("RL interface setup complete.")

    # ==================================================================
    # --- Reset ---
    # ==================================================================
    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new initial state.

        - Removes old objects.
        - Resets robot to home pose with gripper open.
        - Randomly selects a target layout pattern.
        - Defines target locations based on the pattern.
        - Places target visuals (plates).
        - Randomly places blocks in the spawn area, avoiding targets/dump.
        - Lets the simulation settle.
        - Returns the initial observation and info dict.
        """
        super().reset(seed=seed)  # Handles seeding via Gymnasium wrapper
        self.current_steps = 0
        self.held_object_id = None
        self.held_object_idx = None

        # --- Cleanup from previous episode ---
        # Remove constraint first if it exists
        if self.grasp_constraint_id is not None:
            try:
                p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
                # print(f"Reset: Removed constraint {self.grasp_constraint_id}") # Optional debug
            except Exception as e:
                # print(f"Warning: Tried to remove constraint {self.grasp_constraint_id} but failed: {e}")
                pass  # Ignore if removal fails (might already be gone)
            self.grasp_constraint_id = None  # Ensure it's None after attempt

        # Remove old blocks
        for block_id in self.block_ids:
            try:
                p.removeBody(block_id, physicsClientId=self.client)
            except Exception:
                pass
        # Remove old target visuals
        for target_id in self.target_ids:
            try:
                p.removeBody(target_id, physicsClientId=self.client)
            except Exception:
                pass
        self.block_ids = []
        self.target_ids = []
        # ------------------------------------

        # --- Reset Robot Pose to Home ---
        print("Resetting robot to home pose...")
        for i, idx in enumerate(self.arm_joint_indices):
            # Set motor target to home pose (should already be there from init, but ensures consistency)
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL,
                                    targetPosition=self.home_pose_joints[i],
                                    force=self.arm_max_forces[i],
                                    positionGain=self.arm_kp[i], velocityGain=self.arm_kd[i],
                                    physicsClientId=self.client)
        # Ensure gripper is open using the helper function
        self._set_gripper_state("open", wait=False)
        wait_steps(50, self.client, timestep=self.timestep, use_gui=self.use_gui)
        # --------------------------------

        # --- Define Target and Dump Locations for this Episode ---
        # Select a random pattern for the targets
        selected_pattern = self.np_random.choice(self.target_pattern_options)
        # Define target locations based on pattern
        self.target_locations_pos = self._define_locations(
            num_locs=self.num_locations,
            is_target=True,
            pattern_type=selected_pattern
            # Optional: Pass target_area_bounds if needed for 'random_scatter'
            # target_area_bounds=[...]
        )
        # Define the fixed dump location(s)
        # Assuming self._dump_location_base_pos was set in __init__
        self.dump_location_pos = self._define_locations(
            num_locs=self.num_dump_locations,
            is_target=False
        )[0]  # Get the first (likely only) dump location position

        print(f"  Generated target pattern: {selected_pattern}")
        print(f"  Target Locations (base Z): {np.round(self.target_locations_pos, 2)}")
        print(f"  Dump Location (base Z): {np.round(self.dump_location_pos, 2)}")
        # ---------------------------------------------------------

        # --- Place Target Visuals (Plates) ---
        self.goal_config = {}  # Stores { target_loc_idx: required_block_idx }
        # Generate target indices and shuffle them to randomize which location gets which color/block index
        target_loc_indices = list(range(self.num_locations))
        self.np_random.shuffle(target_loc_indices)  # Use seeded random generator

        for i in range(self.num_locations):
            # Check if enough locations were generated (e.g., random scatter might fail)
            if i >= len(self.target_locations_pos):
                print(
                    f"Warning: Not enough target locations generated ({len(self.target_locations_pos)}) for goal assignment {i}.")
                break

            loc_idx = target_loc_indices[i]  # Get the shuffled location index
            target_pos = self.target_locations_pos[loc_idx]  # Get the actual [x,y,z] coords for this location index
            target_color_idx = i % len(self.target_colors_rgba)  # Color index cycles through available colors
            target_color_rgba = self.target_colors_rgba[target_color_idx]

            # Assign Goal: The block with index `i` needs to go to the location at index `loc_idx`
            self.goal_config[loc_idx] = i

            # Create visual plate for the target
            plate_half_extents = [0.04, 0.04,
                                  0.0005]  # Make plate slightly smaller than block base? Plate thickness 1mm.
            # Calculate center Z so plate bottom is slightly above table surface
            plate_center_z = self.table_height + plate_half_extents[2] + 0.0001  # Tiny clearance above table

            try:
                vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=plate_half_extents, rgbaColor=target_color_rgba,
                                             physicsClientId=self.client)
                coll_id = -1  # No collision for target plate visuals
                plate_id = p.createMultiBody(
                    baseMass=0,  # Static object
                    baseCollisionShapeIndex=coll_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[target_pos[0], target_pos[1], plate_center_z],  # Use target X/Y, calculated Z
                    baseOrientation=[0, 0, 0, 1],  # Flat orientation
                    physicsClientId=self.client
                )
                if plate_id < 0:  # Check if body creation failed
                    print(f"Warning: Failed to create target plate visual {i} at location index {loc_idx}")
                else:
                    self.target_ids.append(plate_id)  # Store ID only if successful
            except Exception as e:
                print(f"Error creating target plate visual {i}: {e}")
        # -----------------------------------

        # --- Place Dynamic Blocks Randomly ---
        # Needs self.target_locations_pos and self.dump_location_pos for collision avoidance
        available_spawn_locations = self._get_valid_spawn_positions(self.num_blocks)
        if len(available_spawn_locations) < self.num_blocks:
            # This could happen if spawn area is too small or cluttered
            raise RuntimeError(
                f"Not enough valid spawn locations ({len(available_spawn_locations)}) for {self.num_blocks} blocks.")

        for i in range(self.num_blocks):
            spawn_pos_xy = available_spawn_locations[i]
            # Place block base slightly above table surface
            spawn_z = self.table_height + self.block_half_extents[2] + 0.001
            block_start_pos = [spawn_pos_xy[0], spawn_pos_xy[1], spawn_z]
            # Random initial Z rotation
            block_start_orientation = p.getQuaternionFromEuler([0, 0, self.np_random.uniform(0, 2 * np.pi)])

            try:
                # Use scaling with cube.urdf (ensure cube.urdf is unit size)
                block_id = p.loadURDF("cube.urdf", block_start_pos, block_start_orientation,
                                      globalScaling=self.block_scale,  # Apply scaling here
                                      physicsClientId=self.client)

                if block_id < 0: raise Exception("p.loadURDF failed for block")

                block_color_rgba = self.block_colors_rgba[i]
                p.changeVisualShape(block_id, -1, rgbaColor=block_color_rgba, physicsClientId=self.client)
                # Set dynamics: mass, friction
                p.changeDynamics(block_id, -1, mass=0.1, lateralFriction=0.6, physicsClientId=self.client)
                self.block_ids.append(block_id)
            except Exception as e:
                print(f"Error loading block {i}: {e}")
                # Decide how to handle: raise error, skip block?
                raise e  # Raise error for now
        # ------------------------------------

        # --- Settle Simulation ---
        print("Waiting for objects to settle...")
        wait_steps(150, self.client, timestep=self.timestep, use_gui=self.use_gui)
        # -------------------------

        # --- Get Initial Observation and Info ---
        observation = self._get_obs()
        info = self._get_info()
        print("Reset finished.")
        return observation, info

    # ==================================================================
    # --- Core Step and Primitive Execution ---
    # ==================================================================

    def step(self, action):
        # Store state before action for potential reward shaping or info
        # info_before = self._get_info()
        # current_goal_status = self._check_goal() # Check before action

        success = self._execute_primitive(action) # Execute skill

        observation = self._get_obs()
        terminated = self._check_goal()
        reward = self.goal_reward if terminated else self.step_penalty
        self.current_steps += 1
        truncated = self.current_steps >= self.max_steps

        # Add penalties based on primitive failure
        if not success:
            reward += self.move_fail_penalty # Generic failure penalty (covers IK, reachability, grasp check etc.)

        info = self._get_info()
        info['primitive_success'] = success

        # More advanced reward shaping could go here
        # e.g., reward for picking up correct block, reward for placing block on target

        return observation, reward, terminated, truncated, info

    def _execute_primitive(self, action_index):
        """ Executes the high-level skill based on action_index. Returns True if skill sequence succeeds. """
        print(f"\n--- Executing Action Index: {action_index} ---")
        ori_down = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])  # Used for placing

        try:
            # --- Action: Pick_Block(block_idx) ---
            if 0 <= action_index < self.num_blocks:
                block_idx_to_pick = action_index
                print(f"Attempting Pick_Block({block_idx_to_pick})")
                # Check preconditions: not holding, valid index
                if self.held_object_id is not None:
                    print("  Failure: Already holding.")
                    return False
                if block_idx_to_pick >= len(self.block_ids):
                    print(f"  Failure: Invalid block index {block_idx_to_pick}")
                    return False
                block_id = self.block_ids[block_idx_to_pick]

                try:  # Get block pose AND ORIENTATION
                    block_pos, block_orn_quat = p.getBasePositionAndOrientation(block_id, physicsClientId=self.client)
                    block_euler = p.getEulerFromQuaternion(block_orn_quat)
                    print(
                        f"  Block {block_idx_to_pick} Pose: Pos={np.round(block_pos, 3)}, Euler={np.round(block_euler, 2)}")
                except Exception as e:
                    print(f"  Failure: Cannot get pose for block {block_id}. {e}")
                    return False

                # --- Calculate Target Orientation based on Block Yaw ---
                block_yaw = block_euler[2]
                target_ori = p.getQuaternionFromEuler([np.pi, 0.0, block_yaw])
                print(f"Target Grasp Ori (Euler): {np.round(p.getEulerFromQuaternion(target_ori), 2)}")

                # Calculate poses relative to current block pos and TOP surface
                object_top_z = block_pos[2] + self.block_half_extents[2]
                pre_grasp_pos_z = object_top_z + self.z_hover_offset
                grasp_pos_z = object_top_z + self.grasp_clearance_above_top  # Using clearance from top
                lift_pos_z = object_top_z + self.z_hover_offset + 0.05  # Lift slightly higher
                pre_grasp_pos = [block_pos[0], block_pos[1], pre_grasp_pos_z]
                grasp_pos = [block_pos[0], block_pos[1], grasp_pos_z]
                lift_pos = [block_pos[0], block_pos[1], lift_pos_z]

                # --- Pick Sequence ---
                print("  1. Opening gripper (just in case)...")
                self._set_gripper_state("open", wait=True)

                print("  2. Moving above block (adjusted ori)...")
                # Move to pre-grasp using the block-aligned orientation
                if not self._move_ee_to_pose(pre_grasp_pos, target_ori):
                    print("  Failure: Could not reach pre-grasp pose.")
                    return False  # Give up if pre-grasp fails

                print("  3. Moving down to grasp (adjusted ori)...")
                # Move down using the block-aligned orientation
                if not self._move_ee_to_pose(grasp_pos, target_ori):
                    print("  Move down failed. Aborting pick.")
                    self._set_gripper_state("open", wait=False)
                    current_pos, current_ori = self._get_ee_pose()
                    if current_pos:  # Try to recover upwards
                        recover_pos = [current_pos[0], current_pos[1], current_pos[2] + 0.05]
                        # Use last target orientation or current if available for recovery
                        self._move_ee_to_pose(recover_pos, current_ori if current_ori else target_ori)
                    return False

                # *** Move down succeeded, now close the gripper ***
                print("  4. Closing gripper...")
                self._set_gripper_state("close", wait=True)

                # Optional: Check if gripper closed sufficiently
                if self.finger_indices:
                    states = p.getJointStates(self.robot_id, self.finger_indices, self.client)
                    # Check if fingers are near the *intended* closed value
                    closed_check_val = self.gripper_closed_value + 0.01  # Allow tolerance
                    finger1_val = states[0][0]
                    finger2_val = states[1][0]
                    print(
                        f"  Gripper state after close command: {finger1_val:.4f}, {finger2_val:.4f} (Target: {self.gripper_closed_value:.4f})")
                    # Check if BOTH fingers are sufficiently closed
                    if finger1_val > closed_check_val or finger2_val > closed_check_val:
                        print(f"  WARNING: Gripper did not close sufficiently. Aborting grasp.")
                        self._set_gripper_state("open", wait=False)
                        self._move_ee_to_pose(pre_grasp_pos, target_ori)  # Move back up
                        return False

                print("  5. Attaching object...")
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
                    print(f"  Constraint created: {self.grasp_constraint_id}")
                except Exception as e:
                    print(f"  Failure: Error creating constraint: {e}")
                    self._set_gripper_state("open", wait=False)
                    return False
                wait_steps(60, self.client, timestep=self.timestep, use_gui=self.use_gui)
                self.held_object_id = block_id
                self.held_object_idx = block_idx_to_pick

                print("  6. Lifting block...")
                if not self._move_ee_to_pose(lift_pos, target_ori):
                    # Lift failure cleanup
                    print("  Lift failed, releasing constraint and object.")

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

                print("  Pick sequence successful.")
                return True


            # --- Action: Place_Target / Place_Dump ---
            elif self.num_blocks <= action_index <= self.num_blocks + self.num_locations:

                is_dump = (action_index == self.num_blocks + self.num_locations)
                target_loc_idx = action_index - self.num_blocks if not is_dump else -1
                loc_name = "Dump" if is_dump else f"Target({target_loc_idx})"
                print(f"Attempting Place_{loc_name}")

                if self.held_object_id is None:
                    print("  Failure: Not holding object.")
                    return False

                target_pos_table = self.dump_location_pos if is_dump else self.target_locations_pos[target_loc_idx]
                target_base_z = self.table_height + self.block_half_extents[2]

                pre_place_pos_z = target_base_z + self.block_half_extents[2] + self.z_hover_offset
                pre_place_pos = [target_pos_table[0], target_pos_table[1], pre_place_pos_z]

                place_pos_z = target_base_z + self.block_half_extents[2] + self.place_clearance_above_top
                place_pos = [target_pos_table[0], target_pos_table[1], place_pos_z]

                post_place_pos_z = pre_place_pos_z
                post_place_pos = [target_pos_table[0], target_pos_table[1], post_place_pos_z]

                print(f"  1. Moving above {loc_name}...")
                if not self._move_ee_to_pose(pre_place_pos, ori_down):
                    return False

                print(f"  2. Moving down to {loc_name}...")
                place_move_success = self._move_ee_to_pose(place_pos, ori_down)  # Uses new place_pos
                if not place_move_success:
                    print(f"  Warning: Did not fully reach {loc_name} pose.")

                print("  3. Releasing object...")
                if self.grasp_constraint_id is not None:
                    try:
                        p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
                    except Exception as e:
                        print(f"  Warning: Failed removing constraint {self.grasp_constraint_id}: {e}")
                    self.grasp_constraint_id = None


                self._set_gripper_state("open", wait=True)

                self.held_object_id = None
                self.held_object_idx = None

                wait_steps(50, self.client, timestep=self.timestep, use_gui=self.use_gui)

                print("  4. Moving arm up...")
                self._move_ee_to_pose(post_place_pos, ori_down)
                print(f"  {loc_name} sequence finished.")
                return True

            else:  # Unknown action
                print(f"Warning: Unknown action index {action_index}")
                return False

        except Exception as e:  # Catch any unexpected errors in primitive execution
            print(f"!! Error during primitive execution for action {action_index}: {type(e).__name__} - {e}")
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
        print(f"Commanding gripper {state.upper()}")
        if state == 'open':
            target_value = self.gripper_open_value
        elif state == 'close':
            target_value = self.gripper_closed_value
        else:
            print(f"Error: Invalid gripper state '{state}'. Use 'open' or 'close'.")
            return False

        if not self.finger_indices:
            print("Error: Gripper finger indices not set.")
            return False
        if len(self.finger_indices) != 2:
            print(f"Error: Expected 2 finger indices, found {len(self.finger_indices)}.")
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
            print(f"Error during setJointMotorControlArray for gripper: {e}")
            return False


    def _move_ee_to_pose(self, target_pos, target_ori, max_steps_override=None):
        """ Calculates IK and commands arm to target pose. Returns True if successful. """
        self._last_ik_failure = False # Reset IK failure flag for this attempt

        if self.arm_limits is None:
            print("Error: Arm limits not set.")
            return False

        ll, ul, jr = self.arm_limits

        # print(f"  Attempting IK for Pose: Pos={np.round(target_pos, 3)}, Ori={np.round(p.getEulerFromQuaternion(target_ori), 2)}") # Moved print outside
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
            print(f"  ----> IK Succeeded! (Took {ik_time:.4f} s)")
            print("        Commanding Arm Motion...")

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
                    print(f"        SUCCESS: Reached Target Pose! Dist: {final_dist:.4f}, Ori angle err: {abs(ori_angle):.4f}")
                else:
                    print(f"        FAILURE: Did not reach Target Pose. Dist: {final_dist:.4f} (OK={pos_ok}), Ori angle err: {abs(ori_angle):.4f} (OK={ori_ok})")
                return pos_ok and ori_ok

            except Exception as e:
                print(f"        Error checking final pose: {e}") # Added indent
                return False
        else:
            print(f"  ----> IK Failed! (Took {ik_time:.4f} s)")
            self._last_ik_failure = True # Set flag if IK itself failed
            return False

    def _get_ee_pose(self):
        """ Gets the current world pose of the end-effector link. """
        try:
            link_state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True,
                                        physicsClientId=self.client)
            return link_state[4], link_state[5]  # world pos, world orn
        except Exception as e:
            print(f"Error getting EE pose: {e}")
            return None, None

    # ==================================================================
    # --- Environment Specific Methods (Observation, Goal Check etc.) ---
    # ==================================================================

    def _define_locations(self, num_locs, is_target, pattern_type='line_y', target_area_bounds=None):
        """
        Defines locations on the table.
        If is_target=True, generates positions based on pattern_type.
        If is_target=False, generates dump location(s).
        """
        locations = []
        center_x = self.table_start_pos[0] + 0.0 # Center relative to table X
        center_y = 0.0
        z_pos = self.table_height # Base Z for targets/dump is table height

        if is_target:
            print(f"  Defining {num_locs} target locations with pattern: {pattern_type}")
            min_target_dist_sq = (self.block_scale * 2.0)**2 # Min distance between target centers

            if pattern_type == 'line_y':
                target_spacing = 0.15
                line_x = center_x + 0.20 # Place line forward from center
                y_start = center_y - target_spacing * (num_locs - 1) / 2.0 # Center the line around Y=0
                for i in range(num_locs):
                    locations.append([line_x, y_start + i * target_spacing, z_pos])

            elif pattern_type == 'circle':
                radius = 0.18 # Radius of the circle
                # Slightly offset circle center forward to avoid robot base
                circle_center_x = center_x + 0.15
                circle_center_y = center_y
                # Start angle slightly offset so positions aren't exactly on axes if possible
                angle_offset = self.np_random.uniform(0, math.pi / num_locs) if num_locs > 0 else 0
                for i in range(num_locs):
                    angle = angle_offset + 2 * math.pi * i / num_locs
                    x = circle_center_x + radius * math.cos(angle)
                    y = circle_center_y + radius * math.sin(angle)
                    # Basic check to keep within reasonable table bounds (optional)
                    if abs(x - self.table_start_pos[0]) > 0.4 or abs(y) > 0.4:
                         print(f"Warning: Circle target {i} potentially off table, adjusting.")
                         # Simple fallback: place near center (or implement retry)
                         x = circle_center_x + 0.1 * math.cos(angle)
                         y = circle_center_y + 0.1 * math.sin(angle)
                    locations.append([x, y, z_pos])

            elif pattern_type == 'random_scatter':
                 if target_area_bounds is None:
                      # Define default target area bounds: right side of the table
                      target_area_bounds = [center_x + 0.05, center_x + 0.25, -0.2, 0.2] # [minX, maxX, minY, maxY]
                 print(f"    Scattering targets within: {np.round(target_area_bounds, 2)}")
                 attempts = 0
                 max_attempts = num_locs * 50 # Limit attempts
                 while len(locations) < num_locs and attempts < max_attempts:
                    attempts += 1
                    x = self.np_random.uniform(target_area_bounds[0], target_area_bounds[1])
                    y = self.np_random.uniform(target_area_bounds[2], target_area_bounds[3])
                    # Check distance to other targets chosen *this episode*
                    too_close = False
                    for loc in locations:
                         dist_sq = (loc[0]-x)**2 + (loc[1]-y)**2
                         if dist_sq < min_target_dist_sq:
                              too_close = True
                              break
                    if not too_close:
                         locations.append([x, y, z_pos]) # Store base Z only
                 if len(locations) < num_locs: print(f"Warning: Only placed {len(locations)}/{num_locs} random targets.")

            else:
                print(f"Warning: Unknown target pattern type '{pattern_type}'. Using default line_y.")
                # Fallback to line_y (copy logic from above)
                target_spacing = 0.15
                line_x = center_x + 0.20
                y_start = center_y - target_spacing * (num_locs - 1) / 2.0
                for i in range(num_locs):
                    locations.append([line_x, y_start + i * target_spacing, z_pos])

        else: # Define Dump location(s) - Keep fixed relative to table center for simplicity
             dump_x = self._dump_location_base_pos[0]
             dump_y = self._dump_location_base_pos[1]
             for i in range(num_locs): # Allows multiple dump locs if needed later
                 # Simple fixed position for the first dump location, stagger others
                 locations.append([dump_x, dump_y + i*0.1, z_pos])

        return locations

    def _define_spawn_area(self):
        # Rectangle in front-left relative to robot base
        min_x = self.table_start_pos[0] + 0.0 # Align with table center X
        max_x = self.table_start_pos[0] + 0.25
        min_y = -0.20
        max_y = -0.05 # Spawn on left side
        return [min_x, max_x, min_y, max_y]

    def _get_valid_spawn_positions(self, num_required):
        valid_positions = []
        min_dist_sq = (self.block_scale * 1.2)**2 # Check distance
        target_dump_poses_xy = [[loc[0], loc[1]] for loc in self.target_locations_pos] + [[self.dump_location_pos[0], self.dump_location_pos[1]]]
        attempts = 0
        max_attempts = num_required * 100

        while len(valid_positions) < num_required and attempts < max_attempts:
            attempts += 1
            x = self.np_random.uniform(self.spawn_area_bounds[0], self.spawn_area_bounds[1])
            y = self.np_random.uniform(self.spawn_area_bounds[2], self.spawn_area_bounds[3])
            candidate_pos = [x, y]
            too_close_to_spawn = any(((pos[0]-x)**2 + (pos[1]-y)**2 < min_dist_sq) for pos in valid_positions)
            if too_close_to_spawn: continue
            too_close_to_target = any(((target_pos[0]-x)**2 + (target_pos[1]-y)**2 < (self.block_scale*1.5)**2) for target_pos in target_dump_poses_xy)
            if too_close_to_target: continue
            valid_positions.append(candidate_pos)
        if len(valid_positions) < num_required: print(f"Warning: Only found {len(valid_positions)}/{num_required} valid spawn locations.")
        return valid_positions

    def _get_obs(self):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(self.camera_target_pos, self.camera_distance, self.camera_yaw, self.camera_pitch, 0, 2, self.client)
        proj_matrix = p.computeProjectionMatrixFOV(60, float(self.image_size)/self.image_size, 0.1, 2.0, self.client)
        try:
            (_, _, px, _, _) = p.getCameraImage(self.image_size, self.image_size, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.client)
            rgb_array = np.array(px, dtype=np.uint8)[:, :, :3]
            return rgb_array
        except Exception as e:
             print(f"Error getting camera image: {e}. Returning blank image.")
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
        else: return super(PhysicsBlockRearrangementEnv, self).render(mode=mode)

    def close(self):
        if hasattr(self, 'client') and self.client >= 0:
            try:
                if p.isConnected(physicsClientId=self.client):
                     p.disconnect(physicsClientId=self.client)
            except Exception as e: print(f"Error disconnecting PyBullet: {e}")
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