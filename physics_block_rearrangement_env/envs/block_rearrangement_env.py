# physics_block_rearrangement_env/envs/block_rearrangement_env.py

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import random
import os
import time # Only needed for GUI sleeps if used directly

class PhysicsBlockRearrangementEnv(gym.Env):
    """
    Gymnasium environment for block rearrangement using PyBullet.

    - **Goal:** Move N colored blocks to N target locations of matching color.
    - **Obstructions:** Blocks start in random valid positions and may obstruct
      each other or target locations. A dump location is available.
    - **Observation:** RGB Image.
    - **Actions (High-Level Skills):**
        - Pick_Block(block_index) : 0 to N-1
        - Place_Target(target_index): N to N+M-1 (M=num_targets=num_blocks)
        - Place_Dump() : N+M
    - **Reward:** Sparse goal reward + step penalty.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, use_gui=False, num_blocks=3, num_dump_locations=1, robot_type='ur3e'):
        super().__init__()
        self.num_blocks = num_blocks
        # Assuming one target location per block + 1 dump location
        self.num_locations = self.num_blocks # Number of goal targets
        self.num_dump_locations = num_dump_locations
        self.num_total_placements = self.num_locations + self.num_dump_locations

        self.render_mode = render_mode
        self.use_gui = use_gui or (render_mode == 'human')
        self.robot_type = robot_type

        # --- Physics and Sim Parameters ---
        self.gravity = [0, 0, -9.81]
        self.timestep = 1./240. # Default PyBullet timestep
        self.primitive_max_steps = 150 # Simulation steps per primitive execution attempt
        self.max_steps = 50 * self.num_blocks # Heuristic max steps per episode
        self.step_penalty = -0.01
        self.goal_reward = 1.0

        # --- Robot Parameters ---
        # TCP offset from tool0/flange (NEEDS VERIFICATION FOR YOUR GRIPPER MODEL)
        self.tcp_offset_pos = [0.0, 0.0, 0.200]  # Example offset (e.g., 20cm) - TUNE THIS
        self.tcp_offset_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.inv_tcp_offset_pos, self.inv_tcp_offset_orn = p.invertTransform(
            self.tcp_offset_pos, self.tcp_offset_orn)

        # Gripper joint values (NEEDS VERIFICATION based on URDF limits and testing)
        self.gripper_open_value = 0.2  # Assuming 0 is open based on previous discussion
        self.gripper_closed_value = 0.695  # Assuming 0.7 is closed (use slightly less, e.g., 0.68?)
        self.gripper_force = 5.0  # Increased force - TUNE THIS

        # --- PyBullet Connection ---
        if self.use_gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # <-- Turn OFF normal visual rendering
            # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)  # <-- Turn ON wireframe (helps see overlaps)
            # Set initial camera view (TUNE THIS)
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=-40, cameraTargetPosition=[0.5, 0.0, 0.65])
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
        p.setPhysicsEngineParameter(fixedTimeStep=self.timestep, numSolverIterations=1000)
        p.setPhysicsEngineParameter(physicsClientId=self.client)
        p.setRealTimeSimulation(0)

        # --- Load Scene ---
        # Asset Paths
        self.assets_path = os.path.join(os.path.dirname(__file__), '..', 'assets')
        self.plane_urdf_path = "plane.urdf" # Use pybullet_data path
        self.table_urdf_path = "table/table.urdf" # Use pybullet_data path

        if self.robot_type == 'ur3e':
            self.robot_urdf_path = os.path.join(self.assets_path, "urdf/robots/ur3e_robotiq/ur3e_robotiq_140.urdf")
        elif self.robot_type == 'panda':
             self.robot_urdf_path = os.path.join(self.assets_path, "urdf/robots/panda/panda.urdf") # TODO: Get Panda URDF
        else:
            raise ValueError(f"Unsupported robot_type: {self.robot_type}")

        # --- Load Static Assets ---
        self.plane_id = p.loadURDF(self.plane_urdf_path, physicsClientId=self.client)
        self.table_start_pos = [0.5, 0, 0]
        self.table_id = p.loadURDF(self.table_urdf_path, basePosition=self.table_start_pos, useFixedBase=True, physicsClientId=self.client)
        aabb = p.getAABB(self.table_id, -1, physicsClientId=self.client)
        self.table_height = aabb[1][2] # Max Z

        # --- Load Robot ---
        self.robot_start_pos = [0, 0, self.table_height] # Base at table height
        self.robot_start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(self.robot_urdf_path, self.robot_start_pos, self.robot_start_ori, useFixedBase=True, physicsClientId=self.client)

        # --- RL Parameters ---
        # Action Space Definition
        # N pick actions, M place-target actions, 1 place-dump action
        self.num_actions = self.num_blocks + self.num_locations + self.num_dump_locations
        self.action_space = spaces.Discrete(self.num_actions)
        print(f"Initialized Env: {self.num_blocks} blocks, {self.num_locations} targets, {self.num_dump_locations} dump -> {self.num_actions} actions")

        # Observation Space
        self.image_size = 56
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.image_size, self.image_size, 3),
                                            dtype=np.uint8)

        # --- Task Parameters ---
        self.block_scale = 0.07
        self.block_half_height = self.block_scale / 2.0
        self.grasp_approach_dist = 0.03 # Height above table for grasp attempt
        self.release_dist = 0.03 # Height above table/target for release
        self.safe_raise_height_abs = 0.1 # Absolute Z height relative to table for "Raise" primitive

        # --- Colors ---
        # Example: Red, Green, Blue, Yellow, Magenta
        self.block_colors_rgba = [
            [0.8, 0.1, 0.1, 1.0],
            [0.1, 0.8, 0.1, 1.0],
            [0.1, 0.1, 0.8, 1.0],
            [0.8, 0.8, 0.1, 1.0],
            [0.8, 0.1, 0.8, 1.0],
        ][:self.num_blocks]
        # Target colors should match block colors
        self.target_colors_rgba = self.block_colors_rgba

        # --- Define Placement Locations ---
        self.target_locations_pos = self._define_locations(self.num_locations, is_target=True)
        self.dump_location_pos = self._define_locations(self.num_dump_locations, is_target=False)[0] # Assuming one dump location
        self.spawn_area_bounds = self._define_spawn_area() # Define bounds [min_x, max_x, min_y, max_y]

        # --- Get Robot Info ---
        self.ee_link_index = self._find_link_index("tool0") # Verify name!
        self.gripper_joint_indices = self._find_joint_indices(["finger_joint"]) # Verify name!
        self._extract_joint_limits_and_set_rest_pose() # Populate limits/ranges/rest

        # --- Internal State Variables ---
        self.block_ids = []
        self.target_ids = []
        self.goal_config = {} # Maps target_loc_idx -> required_block_color_idx
        self.current_steps = 0
        self.held_object_id = None # ID of the block currently held
        self.held_object_idx = None # Index (0 to N-1) of the block currently held
        self.grasp_constraint_id = None

        # --- Camera Setup ---
        # TODO: Tune these values using visualize_env.py
        self.camera_target_pos = [self.table_start_pos[0], self.table_start_pos[1], self.table_height + 0.1]
        self.camera_distance = 1.0
        self.camera_yaw = 90
        self.camera_pitch = -45
        print("Environment Initialized.")

    def _define_locations(self, num_locs, is_target):
        """ Defines fixed locations on the table for targets or dump area. """
        locations = []
        # TODO: Define robust layout logic (e.g., grid, circle)
        # Example: Place targets on right (+y), dump on left (-y)
        center_x = self.table_start_pos[0] + 0.1 # Offset from table center slightly
        center_y = 0.0
        spacing = 0.12
        z_pos = self.table_height + 0.001 # Slightly above table

        if is_target:
            y_start = spacing * (num_locs -1) / 2.0
            for i in range(num_locs):
                locations.append([center_x - 0.4, y_start - i * spacing, z_pos])
        else: # Dump location
             locations.append([center_x - 0.15, 0.0, z_pos])

        # TODO: Add visualization for locations (e.g., debug lines or transparent markers)
        # if self.use_gui:
        #     for loc in locations:
        #         p.addUserDebugText(f"{'T' if is_target else 'D'}{len(locations)-1}", [loc[0], loc[1], loc[2]+0.05])

        return locations

    def _define_spawn_area(self):
        """ Defines the XY bounds where blocks can initially spawn. """
        # TODO: Define bounds to avoid targets/dump and robot base
        # Example: A rectangle in front of the robot, away from target/dump sides
        min_x = self.table_start_pos[0] - 0.3
        max_x = self.table_start_pos[0] - 0.1
        min_y = -0.2
        max_y = 0.2
        return [min_x, max_x, min_y, max_y]

    def _find_link_index(self, link_name):
        """ Utility to find link index by name. """
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            if info[12].decode('UTF-8') == link_name:
                return i # Joint index usually corresponds to the link it connects to for kinematic chain
        print(f"Warning: Link '{link_name}' not found directly by joint connection.")
        # Fallback check if it's the base link (-1) - less common for EE
        try: # Check requires pybullet 3.1.7+
             base_info = p.getBodyInfo(self.robot_id, physicsClientId=self.client)
             if base_info[0].decode('UTF-8') == link_name: return -1
        except: pass
        # Fallback: Check common end-effector names
        possible_ee_names = ["tool0", "flange", "panda_hand", "panda_hand_tcp", "ee_link", "wrist_3_link"]
        for name in possible_ee_names:
             for i in range(num_joints):
                 info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
                 link_name_decoded = info[12].decode('UTF-8')
                 if link_name_decoded == name:
                      print(f"Warning: Using fallback end effector link '{name}' index {i}")
                      return i
        raise ValueError(f"End effector link '{link_name}' (or common fallbacks) not found.")


    def _find_joint_indices(self, joint_names):
        """ Utility to find multiple joint indices by name. """
        indices = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for name_to_find in joint_names:
            found = False
            for i in range(num_joints):
                info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
                if info[1].decode('UTF-8') == name_to_find:
                    indices.append(i)
                    found = True
                    break
            if not found:
                 raise ValueError(f"Gripper joint '{name_to_find}' not found.")
        return indices

    def _extract_joint_limits_and_set_rest_pose(self):
        """ Extracts arm joint limits and sets the rest pose attribute. """
        self.arm_joint_indices = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        self.joint_ranges = []
        self.joint_rest_poses = []

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)

        joint_index_counter = 0
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            joint_type = info[2]
            lower_limit = info[8]
            upper_limit = info[9]
            joint_name = info[1].decode('UTF-8')

            is_limited_revolute = (joint_type == p.JOINT_REVOLUTE and lower_limit < upper_limit)

            # Assumption for UR/Panda: First 6 limited revolute joints are the arm
            if is_limited_revolute and len(self.arm_joint_indices) < 6:
                self.arm_joint_indices.append(i)
                self.joint_lower_limits.append(lower_limit)
                self.joint_upper_limits.append(upper_limit)
                self.joint_ranges.append(upper_limit - lower_limit)
                self.joint_rest_poses.append((lower_limit + upper_limit) / 2.0) # Default rest is midpoint
                print(f"  -> Added Arm Joint: Index={i}, Name='{joint_name}', Limits=[{lower_limit:.2f}, {upper_limit:.2f}]")

        # Override rest poses with a specific neutral configuration
        neutral_pose_angles = self._get_neutral_joint_angles() # Get from helper method
        if len(neutral_pose_angles) == len(self.arm_joint_indices):
            self.joint_rest_poses = neutral_pose_angles
            print("DEBUG: Overrode rest poses with specific neutral configuration.")
        else:
             print(f"DEBUG: Using calculated midpoints as rest poses (found {len(self.arm_joint_indices)} arm joints, needed {len(neutral_pose_angles)} for override).")

        print(f"DEBUG: Finished extracting limits for {len(self.arm_joint_indices)} arm joints.")
        if len(self.arm_joint_indices) != 6:
             print("WARNING: Did not find exactly 6 movable arm joints with limits!")


    def _get_neutral_joint_angles(self):
        """ Returns a list of neutral joint angles for the robot arm. """
        # TODO: TUNE THESE ANGLES FOR A GOOD STARTING POSE
        # Example for UR type robots (elbow up, pointing somewhat down)
        if 'ur' in self.robot_type:
            return [0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0]
        elif 'panda' in self.robot_type:
            # Example for Panda (needs verification)
            return [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785] # 7 DoF
        else:
            return [(l+u)/2.0 for l, u in zip(self.joint_lower_limits, self.joint_upper_limits)] # Default to midpoint


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles seeding for random processes
        self.current_steps = 0
        self.held_object_id = None
        self.held_object_idx = None
        if self.grasp_constraint_id is not None:
            try:
                p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
            except Exception as e: pass
            self.grasp_constraint_id = None

        # --- Remove old objects ---
        for block_id in self.block_ids:
            try: p.removeBody(block_id, physicsClientId=self.client)
            except Exception as e: pass
        for target_id in self.target_ids:
            try: p.removeBody(target_id, physicsClientId=self.client)
            except Exception as e: pass
        self.block_ids = []
        self.target_ids = []

        # --- Reset Robot Pose to Neutral ---
        neutral_angles = self._get_neutral_joint_angles()
        if len(neutral_angles) == len(self.arm_joint_indices):
            for i, joint_idx in enumerate(self.arm_joint_indices):
                # Use force=0 to avoid applying torque during reset if using POSITION_CONTROL later
                p.resetJointState(self.robot_id, joint_idx,
                                  targetValue=neutral_angles[i],
                                  targetVelocity=0.0,
                                  physicsClientId=self.client)
        else:
            print("Warning: Mismatch between neutral angles and arm joints found during reset.")


        self._set_gripper(open_gripper=True) # Reset gripper to open
        self._wait_steps(50) # Short wait after reset

        # --- Place Static Target Locations Visually ---
        self.goal_config = {} # Stores { target_loc_idx: target_block_color_idx }
        target_z = self.table_height + 0.001 # Slightly above table
        # Example: Target 0 needs Block 0 color, Target 1 needs Block 1 color, etc.
        target_loc_indices = list(range(self.num_locations))
        # TODO: Define how target colors map to locations if not 1-to-1 with block index
        for i in range(self.num_locations):
             loc_idx = target_loc_indices[i]
             target_pos = self.target_locations_pos[loc_idx]
             target_color_rgba = self.target_colors_rgba[i % len(self.target_colors_rgba)]
             self.goal_config[loc_idx] = i

             # *** Replace cube loading with plate creation: ***
             plate_half_extents = [0.04, 0.04, 0.0005]  # 8cm x 8cm x 1mm plate
             plate_z_center = self.table_height + plate_half_extents[2]  # Center Z slightly above table

             visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                   halfExtents=plate_half_extents,
                                                   rgbaColor=target_color_rgba,
                                                   physicsClientId=self.client)
             # Use a simple collision shape (or GEOM_PLANE for no collision)
             collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                         halfExtents=plate_half_extents,
                                                         physicsClientId=self.client)
             plate_id = p.createMultiBody(baseMass=0,  # Static object
                                          baseCollisionShapeIndex=collision_shape_id,
                                          baseVisualShapeIndex=visual_shape_id,
                                          basePosition=[target_pos[0], target_pos[1], plate_z_center],
                                          # Use calculated center Z
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                          physicsClientId=self.client)
             self.target_ids.append(plate_id)

        # --- Place Dynamic Blocks Randomly ---
        available_spawn_locations = self._get_valid_spawn_positions(self.num_blocks)
        if len(available_spawn_locations) < self.num_blocks:
             raise RuntimeError(f"Not enough valid spawn locations ({len(available_spawn_locations)}) for {self.num_blocks} blocks.")

        for i in range(self.num_blocks):
            spawn_pos_xy = available_spawn_locations[i]
            half_cube_height = self.block_scale / 2.0
            spawn_z = self.table_height + half_cube_height + 0.001 # On table surface + buffer
            block_start_pos = [spawn_pos_xy[0], spawn_pos_xy[1], spawn_z]
            block_start_orientation = p.getQuaternionFromEuler([0, 0, self.np_random.uniform(0, 2*np.pi)])

            block_id = p.loadURDF("cube.urdf",
                                  block_start_pos,
                                  block_start_orientation,
                                  globalScaling=self.block_scale,
                                  physicsClientId=self.client)
            block_color_rgba = self.block_colors_rgba[i]
            p.changeVisualShape(block_id, -1, rgbaColor=block_color_rgba, physicsClientId=self.client)
            # Set physics properties
            p.changeDynamics(block_id, -1, mass=0.1, lateralFriction=0.6, physicsClientId=self.client)
            self.block_ids.append(block_id)

        # --- Settle ---
        self._wait_steps(100) # Let objects settle

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_valid_spawn_positions(self, num_required):
        """ Finds random valid XY spawn positions within bounds, avoiding targets/dump. """
        valid_positions = []
        min_dist_sq = (self.block_scale * 1.5)**2 # Min distance between block centers
        target_dump_poses_xy = [[loc[0], loc[1]] for loc in self.target_locations_pos] + [[self.dump_location_pos[0], self.dump_location_pos[1]]]

        attempts = 0
        max_attempts = num_required * 50 # Limit attempts

        while len(valid_positions) < num_required and attempts < max_attempts:
            attempts += 1
            # Sample random position within spawn bounds
            x = self.np_random.uniform(self.spawn_area_bounds[0], self.spawn_area_bounds[1])
            y = self.np_random.uniform(self.spawn_area_bounds[2], self.spawn_area_bounds[3])
            candidate_pos = [x, y]

            # Check collision with existing spawns
            too_close_to_spawn = False
            for pos in valid_positions:
                dist_sq = (pos[0] - x)**2 + (pos[1] - y)**2
                if dist_sq < min_dist_sq:
                    too_close_to_spawn = True
                    break
            if too_close_to_spawn:
                continue

            # Check collision with target/dump locations
            too_close_to_target = False
            for target_pos in target_dump_poses_xy:
                 dist_sq = (target_pos[0] - x)**2 + (target_pos[1] - y)**2
                 # Use a slightly larger radius for targets/dump
                 if dist_sq < (self.block_scale * 2.0)**2:
                     too_close_to_target = True
                     break
            if too_close_to_target:
                 continue

            # If checks pass, add position
            valid_positions.append(candidate_pos)

        if len(valid_positions) < num_required:
             print(f"Warning: Could only find {len(valid_positions)} valid spawn locations out of {num_required} required.")
             # Handle error? For now, just use what we found

        return valid_positions


    def step(self, action):
        # Execute the high-level skill corresponding to the action
        success = self._execute_primitive(action) # Primitive execution attempts the skill

        observation = self._get_obs()
        terminated = self._check_goal()
        reward = self.goal_reward if terminated else self.step_penalty
        self.current_steps += 1
        truncated = self.current_steps >= self.max_steps

        # Consider adding penalty if primitive failed?
        if not success:
            reward -= 0.05 # Small penalty for failed primitive attempt

        info = self._get_info()
        info['primitive_success'] = success # Info about primitive success

        return observation, reward, terminated, truncated, info

    def _execute_primitive(self, action_index):
        """ Executes the high-level skill based on action_index. Returns True if skill sequence succeeds. """
        print(f"\n--- Executing Action Index: {action_index} ---")
        target_ori_tcp = p.getQuaternionFromEuler([np.pi, 0, 0]) # Point down

        try:
            # --- Action: Pick_Block(block_idx) ---
            if 0 <= action_index < self.num_blocks:
                block_idx_to_pick = action_index
                print(f"Attempting Pick_Block({block_idx_to_pick})")

                if self.held_object_id is not None:
                    print("  Failure: Already holding an object.")
                    return False # Precondition fail

                if block_idx_to_pick >= len(self.block_ids):
                    print(f"  Failure: Invalid block index {block_idx_to_pick}")
                    return False

                block_id = self.block_ids[block_idx_to_pick]
                try:
                    block_pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self.client)
                except Exception as e:
                    print(f"  Failure: Cannot get pose for block {block_id}. {e}")
                    return False

                # 1. Move Above Block
                print("  1. Moving above block...")
                target_tcp_pos_above = [block_pos[0], block_pos[1], self.table_height + self.safe_raise_height_abs]
                tool0_target_pos_above, tool0_target_orn_above = self._get_tool0_pose_from_tcp_pose(target_tcp_pos_above, target_ori_tcp)
                if not self._move_ee_to_pose(tool0_target_pos_above, tool0_target_orn_above):
                    print("  Failure: Move above block failed (IK or timeout).")
                    return False

                # 2. Lower to Grasp Height
                print("  2. Lowering to grasp height...")
                target_tcp_pos_grasp = [block_pos[0], block_pos[1], self.table_height + self.grasp_approach_dist]
                tool0_target_pos_grasp, tool0_target_orn_grasp = self._get_tool0_pose_from_tcp_pose(target_tcp_pos_grasp, target_ori_tcp)
                if not self._move_ee_to_pose(tool0_target_pos_grasp, tool0_target_orn_grasp, max_steps=50): # Faster move down
                    print("  Failure: Lowering failed (IK or timeout).")
                    return False
                self._wait_steps(20)

                # 3. Close Gripper
                print("  3. Closing gripper...")
                if not self._set_gripper(open_gripper=False):
                    print("  Failure: Closing gripper failed.")
                    return False
                self._wait_steps(50)

                # 4. Check Grasp & Create Constraint
                print("  4. Checking grasp...")
                self.held_object_id = self._get_object_in_gripper(check_dist=0.04)
                if self.held_object_id is None or self.held_object_id != block_id :
                     print(f"  Failure: Grasped wrong object or no object (Targeted: {block_id}, Got: {self.held_object_id}). Reopening.")
                     self.held_object_id = None
                     self._set_gripper(open_gripper=True)
                     return False
                # Successfully detected target block
                self.held_object_idx = block_idx_to_pick
                print(f"  Successfully detected block {self.held_object_idx} (ID: {self.held_object_id})")
                # Create constraint
                ee_state = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.client)
                obj_pos, obj_ori = p.getBasePositionAndOrientation(self.held_object_id, physicsClientId=self.client)
                inv_ee_pos, inv_ee_ori = p.invertTransform(ee_state[4], ee_state[5])
                rel_pos, rel_ori = p.multiplyTransforms(inv_ee_pos, inv_ee_ori, obj_pos, obj_ori)
                try:
                    self.grasp_constraint_id = p.createConstraint(
                        parentBodyUniqueId=self.robot_id, parentLinkIndex=self.ee_link_index,
                        childBodyUniqueId=self.held_object_id, childLinkIndex=-1,
                        jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                        parentFramePosition=rel_pos, childFramePosition=[0, 0, 0],
                        parentFrameOrientation=rel_ori, childFrameOrientation=[0,0,0,1],
                        physicsClientId=self.client )
                    print(f"  Constraint created: {self.grasp_constraint_id}")
                except Exception as e:
                    print(f"  Failure: Error creating constraint: {e}")
                    self.held_object_id = None
                    self.held_object_idx = None
                    self._set_gripper(open_gripper=True)
                    return False

                # 5. Raise
                print("  5. Raising block...")
                # Keep X,Y same as block, raise Z to safe height
                current_tcp_pos_grasp = [block_pos[0], block_pos[1], self.table_height + self.grasp_approach_dist]
                target_tcp_pos_raise = [current_tcp_pos_grasp[0], current_tcp_pos_grasp[1], self.table_height + self.safe_raise_height_abs]
                tool0_target_pos_raise, tool0_target_orn_raise = self._get_tool0_pose_from_tcp_pose(target_tcp_pos_raise, target_ori_tcp)
                if not self._move_ee_to_pose(tool0_target_pos_raise, tool0_target_orn_raise):
                    print("  Warning: Raise after grasp failed (IK or timeout). Continuing, but pose might be wrong.")
                    # Return True anyway, as grasp succeeded? Or False? Let's return False for strictness.
                    # Cleanup potentially broken state:
                    if self.grasp_constraint_id is not None:
                         try: p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
                         except Exception as e: pass
                         self.grasp_constraint_id = None
                    self.held_object_id = None
                    self.held_object_idx = None
                    self._set_gripper(open_gripper=True)
                    return False
                print("  Pick sequence successful.")
                return True


            # --- Action: Place_Target(target_loc_idx) ---
            elif self.num_blocks <= action_index < self.num_blocks + self.num_locations:
                target_loc_idx = action_index - self.num_blocks
                print(f"Attempting Place_Target({target_loc_idx})")

                if self.held_object_id is None:
                    print("  Failure: Not holding an object.")
                    return False # Precondition fail

                target_pos_table = self.target_locations_pos[target_loc_idx]

                # 1. Move Above Target Location
                print("  1. Moving above target location...")
                target_tcp_pos_above = [target_pos_table[0], target_pos_table[1], self.table_height + self.safe_raise_height_abs] # Use safe Z
                tool0_target_pos_above, tool0_target_orn_above = self._get_tool0_pose_from_tcp_pose(target_tcp_pos_above, target_ori_tcp)
                if not self._move_ee_to_pose(tool0_target_pos_above, tool0_target_orn_above):
                    print("  Failure: Move above target failed.")
                    return False

                # 2. Lower to Release Height
                print("  2. Lowering to release height...")
                target_tcp_pos_release = [target_pos_table[0], target_pos_table[1], self.table_height + self.release_dist]
                tool0_target_pos_release, tool0_target_orn_release = self._get_tool0_pose_from_tcp_pose(target_tcp_pos_release, target_ori_tcp)
                if not self._move_ee_to_pose(tool0_target_pos_release, tool0_target_orn_release, max_steps=50):
                    print("  Failure: Lowering to release failed.")
                    # Should we still try to release? Maybe.
                    pass # Continue to release attempt

                # 3. Release Gripper (Remove constraint first!)
                print("  3. Releasing object...")
                if self.grasp_constraint_id is not None:
                     try: p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
                     except Exception as e: print(f"  Warning: Failed to remove constraint {self.grasp_constraint_id}: {e}")
                     self.grasp_constraint_id = None
                release_success = self._set_gripper(open_gripper=True)
                self.held_object_id = None
                self.held_object_idx = None
                self._wait_steps(50) # Allow time for release and object to settle

                # 4. Raise to Safe Height
                print("  4. Raising gripper after release...")
                current_tool0_pos, current_tool0_orn = self._get_tool0_pose()
                if current_tool0_pos is None: current_tool0_pos = tool0_target_pos_release # Use last target if current unavailable
                target_tcp_pos_raise = [current_tool0_pos[0], current_tool0_pos[1], self.table_height + self.safe_raise_height_abs]
                tool0_target_pos_raise, tool0_target_orn_raise = self._get_tool0_pose_from_tcp_pose(target_tcp_pos_raise, target_ori_tcp) # Use default down orientation
                raise_success = self._move_ee_to_pose(tool0_target_pos_raise, tool0_target_orn_raise)

                print(f"  Place sequence finished (Release success: {release_success}, Raise success: {raise_success}).")
                return release_success # Success depends mainly on releasing

            # --- Action: Place_Dump() ---
            elif action_index == self.num_blocks + self.num_locations:
                print(f"Attempting Place_Dump()")
                if self.held_object_id is None:
                     print("  Failure: Not holding an object.")
                     return False # Precondition fail

                target_pos_table = self.dump_location_pos

                # Sequence similar to Place_Target
                # 1. Move Above Dump Location
                print("  1. Moving above dump location...")
                target_tcp_pos_above = [target_pos_table[0], target_pos_table[1], self.table_height + self.safe_raise_height_abs]
                tool0_target_pos_above, tool0_target_orn_above = self._get_tool0_pose_from_tcp_pose(target_tcp_pos_above, target_ori_tcp)
                if not self._move_ee_to_pose(tool0_target_pos_above, tool0_target_orn_above):
                    print("  Failure: Move above dump failed.")
                    return False

                # 2. Lower to Release Height
                print("  2. Lowering to release height...")
                target_tcp_pos_release = [target_pos_table[0], target_pos_table[1], self.table_height + self.release_dist]
                tool0_target_pos_release, tool0_target_orn_release = self._get_tool0_pose_from_tcp_pose(target_tcp_pos_release, target_ori_tcp)
                if not self._move_ee_to_pose(tool0_target_pos_release, tool0_target_orn_release, max_steps=50):
                     print("  Failure: Lowering to release failed.")
                     pass # Continue to release attempt

                # 3. Release Gripper
                print("  3. Releasing object...")
                if self.grasp_constraint_id is not None:
                    try: p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
                    except Exception as e: print(f"  Warning: Failed to remove constraint {self.grasp_constraint_id}: {e}")
                    self.grasp_constraint_id = None
                release_success = self._set_gripper(open_gripper=True)
                self.held_object_id = None
                self.held_object_idx = None
                self._wait_steps(50)

                # 4. Raise to Safe Height
                print("  4. Raising gripper after release...")
                current_tool0_pos, current_tool0_orn = self._get_tool0_pose()
                if current_tool0_pos is None: current_tool0_pos = tool0_target_pos_release
                target_tcp_pos_raise = [current_tool0_pos[0], current_tool0_pos[1], self.table_height + self.safe_raise_height_abs]
                tool0_target_pos_raise, tool0_target_orn_raise = self._get_tool0_pose_from_tcp_pose(target_tcp_pos_raise, target_ori_tcp)
                raise_success = self._move_ee_to_pose(tool0_target_pos_raise, tool0_target_orn_raise)

                print(f"  Place sequence finished (Release success: {release_success}, Raise success: {raise_success}).")
                return release_success

            else:
                print(f"Warning: Unknown action index {action_index}")
                return False

        except Exception as e:
            print(f"Error during primitive execution for action {action_index}: {e}")
            import traceback
            traceback.print_exc()
            # Cleanup potentially broken state
            if self.grasp_constraint_id is not None:
                try: p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
                except Exception as e_c: pass
                self.grasp_constraint_id = None
            self.held_object_id = None
            self.held_object_idx = None
            return False

    # ===========================================================================
    # --- Helper methods for movement, grasping, observation, goal check ---
    # ===========================================================================

    def _get_tool0_pose_from_tcp_pose(self, target_tcp_pos, target_tcp_orn):
        """
        Calculates the required world pose for the tool0 link that would
        place the actual TCP at the desired target_tcp_pos/orn.
        This is the target pose to feed into p.calculateInverseKinematics.

        Args:
            target_tcp_pos ([float, float, float]): Desired TCP position [x,y,z].
            target_tcp_orn ([float, float, float, float]): Desired TCP orientation [x,y,z,w].

        Returns:
            (tool0_target_pos, tool0_target_orn)
        """
        # We want target_tool0 * tcp_offset = target_tcp
        # So, target_tool0 = target_tcp * inv(tcp_offset)
        tool0_target_pos, tool0_target_orn = p.multiplyTransforms(
            target_tcp_pos, target_tcp_orn,
            self.inv_tcp_offset_pos, self.inv_tcp_offset_orn,
            physicsClientId=self.client
        )
        return tool0_target_pos, tool0_target_orn

    def _get_tool0_pose(self):
        """ Gets the current world pose of the tool0 link (robot flange). """
        try:
            link_state = p.getLinkState(self.robot_id,
                                        self.ee_link_index, # Index for tool0/flange
                                        computeForwardKinematics=True, # Ensure FK is computed
                                        physicsClientId=self.client)
            # getLinkState returns: world link pos, world link orn, local inertial pos, local inertial orn,
            # world frame pos, world frame orn ... (indices 4 and 5)
            tool0_pos = link_state[4]
            tool0_orn = link_state[5]
            return tool0_pos, tool0_orn
        except Exception as e:
            print(f"Error getting tool0 pose: {e}")
            # Handle error, maybe return None or a default pose
            return None, None

    def _get_tcp_pose(self):
        """
        Calculates the current world pose of the Tool Center Point (TCP)
        by applying the offset to the tool0 link pose.
        Returns: (tcp_position, tcp_orientation) or (None, None) if error.
        """
        tool0_pos, tool0_orn = self._get_tool0_pose()
        if tool0_pos is None:
            return None, None

        actual_tcp_pos, actual_tcp_orn = p.multiplyTransforms(
            tool0_pos, tool0_orn,
            self.tcp_offset_pos, self.tcp_offset_orn,
            physicsClientId=self.client
        )
        return actual_tcp_pos, actual_tcp_orn

    def _move_ee_to_pose(self, target_pos, target_ori, max_steps=None):
        """ Moves the end effector to a target pose using IK. Returns True if successful."""
        if max_steps is None: max_steps = self.primitive_max_steps
        # TODO: Implement robust IK solving and joint position control.
        # This is a placeholder implementation. Needs refinement.
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_pos,
            targetOrientation=target_ori,
            lowerLimits=self.joint_lower_limits,
            upperLimits=self.joint_upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.joint_rest_poses,
            # solver=p.IK_SDLS,
            maxNumIterations=100,
            residualThreshold=1e-4,
            physicsClientId=self.client
        )

        if joint_poses is None:
            print("Warning: Inverse Kinematics failed.")
            return False

        # Get controllable joint indices (non-fixed)
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        movable_joint_indices = [i for i in range(num_joints) if
                                 p.getJointInfo(self.robot_id, i, physicsClientId=self.client)[2] != p.JOINT_FIXED]

        if len(joint_poses) < len(movable_joint_indices):
            print(
                f"Warning: IK solution size {len(joint_poses)} doesn't match movable joints {len(movable_joint_indices)}")
            return False  # Or handle partial solution if appropriate

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=movable_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_poses[:len(movable_joint_indices)],
            # TODO: Tune forces/gains for smoother/faster movement
            forces=[100.0] * len(movable_joint_indices),
            positionGains=[0.03] * len(movable_joint_indices),  # Example gains
            velocityGains=[1.0] * len(movable_joint_indices),  # Example gains
            physicsClientId=self.client
        )

        # Step simulation to allow movement
        for _ in range(max_steps):
            p.stepSimulation(self.client)
            if self.use_gui: time.sleep(1. / 240.)
            # TODO: Add check if target pose is reached within tolerance?
            # current_ee_pos = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.client)[4]
            # if np.linalg.norm(np.array(current_ee_pos) - np.array(target_pos)) < 0.01:
            #     return True # Reached target

        # Check final pose (optional, depends if exact pose needed)
        final_ee_pos = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.client)[4]
        if np.linalg.norm(np.array(final_ee_pos) - np.array(target_pos)) < 0.02:  # Looser tolerance after timeout
            return True
        else:
            print(
                f"Warning: Move EE failed to reach target. Final dist: {np.linalg.norm(np.array(final_ee_pos) - np.array(target_pos)):.3f}")
            return False

    def _set_gripper(self, open_gripper):
        """ Opens or closes the gripper by setting target position for ALL related joints. """
        target_val_main = self.gripper_open_value if open_gripper else self.gripper_closed_value
        main_joint_idx = self._find_joint_indices(["finger_joint"])[0]
        mimic_names = [
            "left_inner_knuckle_joint", "right_outer_knuckle_joint",
            "right_inner_knuckle_joint", "left_inner_finger_joint",
            "right_inner_finger_joint"
        ]
        mimic_indices = self._find_joint_indices(mimic_names)
        mimic_multipliers = [-1, -1, -1, 1, 1]

        all_joint_indices = [main_joint_idx] + mimic_indices
        target_positions = [target_val_main] + [m * target_val_main for m in mimic_multipliers]

        try:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=all_joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_positions,
                forces=[200.0] * len(all_joint_indices)  # Adjust force as needed
                , physicsClientId=self.client
            )
            self._wait_steps(50)
            return True
        except Exception as e:
            print(f"Error setting gripper joints: {e}")
            return False


    def _get_object_in_gripper(self, check_dist=0.05):
        """ Checks if any manipulable block is close to the TCP. """
        tcp_pos, _ = self._get_tcp_pose()
        if tcp_pos is None: return None

        for i, block_id in enumerate(self.block_ids):
            if block_id == self.held_object_id: continue # Don't detect object already held
            try:
                block_pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self.client)
                # Check distance primarily in XY plane, allow larger Z diff
                dist_xy = np.linalg.norm(np.array(tcp_pos[:2]) - np.array(block_pos[:2]))
                dist_z = abs(tcp_pos[2] - (block_pos[2])) # Z dist to block center
                # print(f"DEBUG: Check grasp: Dist to block {i} (ID:{block_id}): XY={dist_xy:.3f}, Z={dist_z:.3f}")
                # Adjust thresholds: close in XY, TCP slightly above block center
                if dist_xy < (self.block_scale * 0.6) and dist_z < (self.block_half_height * 1.5) :
                    print(f"DEBUG: Object {i} (ID:{block_id}) is potentially in gripper.")
                    return block_id
            except Exception as e:
                # Block might have been removed or invalid
                continue
        return None

    def _wait_steps(self, steps):
        """ Steps simulation for a number of steps. """
        for _ in range(steps):
            p.stepSimulation(self.client)
            if self.use_gui: time.sleep(self.timestep) # Adjust sleep for desired sim speed in GUI

    def _get_obs(self):
        """ Renders the environment image from a fixed viewpoint. """
        # Use parameters found via interactive testing or defaults
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target_pos,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.client
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.image_size) / self.image_size,
            nearVal=0.1,
            farVal=2.0, # Reduced far plane slightly
            physicsClientId=self.client
        )
        try:
            # Request RGB image only
            (_, _, px, _, _) = p.getCameraImage(
                width=self.image_size, height=self.image_size,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL, # Use faster renderer if available, else ER_TINY_RENDERER
                physicsClientId=self.client
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = rgb_array[:, :, :3] # Remove alpha channel if present
            return rgb_array
        except Exception as e:
             print(f"Error getting camera image: {e}. Returning blank image.")
             # Return a blank image or handle error appropriately
             return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)


    def _get_info(self):
        """ Returns auxiliary environment info. """
        info = {
            'held_object_idx': self.held_object_idx,
            'current_steps': self.current_steps,
            # Add positions if needed for debugging / analysis outside RL state
            # 'block_positions': [p.getBasePositionAndOrientation(bid)[0] for bid in self.block_ids],
            # 'goal_config': self.goal_config,
        }
        return info

    def _check_goal(self):
        """ Checks if the current block configuration matches the goal. """
        if self.held_object_id is not None: return False # Cannot be goal state while holding

        on_target_count = 0
        goal_dist_threshold = 0.04 # Tolerance for block being on target

        for target_loc_idx, required_block_idx in self.goal_config.items():
            if required_block_idx < len(self.block_ids):
                block_id = self.block_ids[required_block_idx]
                target_pos = self.target_locations_pos[target_loc_idx]
                try:
                    current_pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self.client)
                    # Check primarily XY distance, ensure Z is close to table surface
                    dist_xy = np.linalg.norm(np.array(current_pos[:2]) - np.array(target_pos[:2]))
                    on_surface = abs(current_pos[2] - (self.table_height + self.block_half_height)) < 0.02

                    if dist_xy < goal_dist_threshold and on_surface:
                        on_target_count += 1
                except Exception as e:
                    # Block might not exist (shouldn't happen in normal operation)
                    print(f"Error checking goal for block {block_id}: {e}")
                    return False # Cannot reach goal if expected block is missing
        # Goal met if all required blocks are on their respective targets
        return on_target_count == len(self.goal_config)


    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self._get_obs()
        elif mode == 'human':
            # GUI mode handles rendering, maybe add slight delay
            if self.use_gui: time.sleep(0.01)
            return None
        else:
            # Raise error for unsupported modes
            return super(PhysicsBlockRearrangementEnv, self).render(mode=mode)

    def close(self):
        if hasattr(self, 'client') and self.client >= 0:
            try:
                if p.isConnected(physicsClientId=self.client):
                     p.disconnect(physicsClientId=self.client)
            except Exception as e:
                print(f"Error disconnecting PyBullet: {e}")
            self.client = -1

# Example usage (if run directly)
if __name__ == '__main__':
    # Example of how to use the environment
    # Use smaller number of blocks/locations for easier testing initially
    env = PhysicsBlockRearrangementEnv(use_gui=True, render_mode='human', num_blocks=2, num_dump_locations=1)

    for episode in range(5):
        print(f"\n--- Episode {episode+1} ---")
        obs, info = env.reset()
        print("Reset done. Initial Info:", info)
        # print("Goal Config:", env.goal_config) # Debug: See the goal
        # print("Block IDs:", env.block_ids)     # Debug: See block IDs

        terminated = False
        truncated = False
        step = 0
        # Try a manual sequence: Pick block 0, place at target 0
        # Action indices: Pick0=0, Pick1=1, PlaceTarg0=2, PlaceTarg1=3, PlaceDump=4
        # Manual sequence assumes num_blocks=2, num_locations=2
        # action_sequence = [0, 2] # Pick Block 0, Place Target 0
        # action_sequence = [1, 3] # Pick Block 1, Place Target 1
        action_sequence = [0, 4, 1, 3] # Pick 0, Dump, Pick 1, Place Target 1
        action_idx = 0

        while not terminated and not truncated:
            # --- Replace random action with test sequence or manual input ---
            # action = env.action_space.sample() # Random actions
            # action = int(input("Enter action index: ")) # Manual input
            if action_idx < len(action_sequence):
                 action = action_sequence[action_idx]
                 action_idx += 1
            else:
                 action = env.action_space.sample() # Fallback to random if sequence ends
                 print("Sequence finished, taking random action.")
            # --- ----------------------------------------------------- ---

            print(f"Step: {step}, Executing Action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  -> Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
            env.render()
            step += 1
            if terminated: print("--- Goal Reached! ---")
            if truncated: print("--- Max steps reached. ---")
            # Add a small sleep for better visualization
            # time.sleep(0.1)

        print(f"Episode {episode+1} finished after {step} steps.")
        time.sleep(1) # Pause between episodes

    env.close()
    print("Environment closed.")