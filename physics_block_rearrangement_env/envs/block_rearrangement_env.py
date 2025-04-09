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
    High-level discrete actions map to pre-defined motion primitives
    for a UR3e/Panda arm with a Robotiq/Panda gripper.
    Goal: Arrange blocks in specific fixed locations, handling obstructions.
    Observation: RGB Image.
    Action: Discrete indices mapping to primitives.
    Reward: Sparse goal reward + step penalty.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, use_gui=False, num_blocks=3, num_locations=5, robot_type='ur3e'):
        super().__init__()
        assert num_locations >= num_blocks, "Must have at least as many locations as blocks"
        self.render_mode = render_mode
        self.use_gui = use_gui or (render_mode == 'human') # Force GUI if human rendering
        self.num_blocks = num_blocks
        self.num_locations = num_locations
        self.robot_type = robot_type # 'ur3e' or 'panda'

        # --- PyBullet Connection ---
        if self.use_gui:
            self.client = p.connect(p.GUI)
            # Improve GUI visuals
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1) # Enable shadows
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0) # We will use stepSimulation

        # --- Environment Parameters ---
        self.table_height = 0.0 # Z-coordinate of table surface
        self.fixed_locations = self._define_locations() # List of [x, y, z] poses on table surface
        # Gripper parameters (adjust based on chosen gripper URDF)
        self.gripper_open_value = 0.08 # Example: Max joint value for open
        self.gripper_closed_value = 0.0  # Example: Min joint value for closed
        self.primitive_max_steps = 100 # Max simulation steps per primitive action
        self.step_penalty = -0.01
        self.goal_reward = 1.0
        self.image_size = 84

        # --- Spaces ---
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.image_size, self.image_size, 3),
                                            dtype=np.uint8)
        # Actions: Move_Above(Loc0..M-1), Lower_Grasp, Raise, Release
        self.num_base_actions = self.num_locations + 3
        self.action_space = spaces.Discrete(self.num_base_actions)

        # --- Asset Paths ---
        self.assets_path = os.path.join(os.path.dirname(__file__), '..', 'assets')
        self.table_urdf_path = os.path.join(self.assets_path, "urdf/objects/table.urdf") # TODO: Get table URDF
        self.block_urdf_path = os.path.join(self.assets_path, "urdf/objects/cube.urdf")  # TODO: Get block URDF
        if self.robot_type == 'ur3e':
            self.robot_urdf_path = os.path.join(self.assets_path, "urdf/robots/ur3e_robotiq/ur3e_robotiq_140.urdf") # TODO: Verify filename
        elif self.robot_type == 'panda':
            self.robot_urdf_path = os.path.join(self.assets_path, "urdf/robots/panda/panda.urdf") # TODO: Get Panda URDF
        else:
            raise ValueError(f"Unsupported robot_type: {self.robot_type}")

        # --- Load Static Assets ---
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.table_id = p.loadURDF(self.table_urdf_path, basePosition=[0.5, 0, 0], useFixedBase=True, physicsClientId=self.client) # Example position

        # --- Robot Setup ---
        self.robot_start_pos = [0, 0, 0.63] # Adjust Z based on table height + robot base
        self.robot_start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(self.robot_urdf_path, self.robot_start_pos, self.robot_start_ori, useFixedBase=True, physicsClientId=self.client)
        # TODO: Find end effector link index and gripper joint indices *by name*
        self.ee_link_index = self._find_link_index("tool0") # Example name, verify for your URDF
        self.gripper_joint_indices = self._find_joint_indices(["finger_joint"]) # Example, verify for your URDF

        # --- Internal State ---
        self.block_ids = []
        self.goal_config = {} # e.g., {block_index_0: location_index_3}
        self.current_steps = 0
        self.max_steps = 500 # Default max steps per episode
        self.held_object_id = None
        self.grasp_constraint_id = None

        # --- Camera Setup ---
        self.camera_target_pos = [0.5, 0.0, self.table_height + 0.1] # Look slightly above table center
        self.camera_distance = 1.0
        self.camera_yaw = 90
        self.camera_pitch = -45
        # Update these with values found from interactive testing!


    def _define_locations(self):
        """ Defines the fixed locations on the table. """
        # TODO: Return a list of self.num_locations target poses [[x, y, z], ...] on the table
        locations = []
        table_z = self.table_height # Assume table URDF origin is at table height
        spacing = 0.15
        center_x = 0.5 # Match table position
        center_y = 0.0
        rows = int(np.ceil(np.sqrt(self.num_locations)))
        cols = int(np.ceil(self.num_locations / rows))

        for i in range(self.num_locations):
            row = i // cols
            col = i % cols
            x = center_x - (cols - 1) * spacing / 2.0 + col * spacing
            y = center_y - (rows - 1) * spacing / 2.0 + row * spacing
            locations.append([x, y, table_z + 0.01]) # Z slightly above table
        return locations

    def _find_link_index(self, link_name):
        """ Utility to find link index by name. """
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            if info[12].decode('UTF-8') == link_name:
                return i
        print(f"Warning: Link '{link_name}' not found.")
        # Fallback: Check common end-effector names
        possible_ee_names = ["tool0", "panda_hand_tcp", "ee_link"] # Add more if needed
        for name in possible_ee_names:
             for i in range(num_joints):
                 info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
                 if info[12].decode('UTF-8') == name:
                      print(f"Warning: Using fallback end effector link '{name}'")
                      return i
        raise ValueError(f"End effector link '{link_name}' (or fallbacks) not found.")


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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_steps = 0
        self.held_object_id = None
        if self.grasp_constraint_id is not None:
             try: # Constraint might already be removed
                 p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
             except Exception as e: pass
             self.grasp_constraint_id = None

        # --- Reset Robot Pose ---
        # TODO: Implement moving arm to a safe neutral pose (using IK or predefined joint angles)
        # Example: Set joints directly to a known neutral configuration
        # neutral_joint_angles = [...]
        # for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
        #     # Check only movable joints (revolute/prismatic)
        #     joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
        #     if joint_info[2] != p.JOINT_FIXED:
        #          p.resetJointState(self.robot_id, i, targetValue=neutral_joint_angles[joint_info[0]], physicsClientId=self.client) # Assuming index matches
        # Or call a primitive: self._move_ee_to_pose(some_neutral_xyz, some_neutral_ori)
        self._set_gripper(open_gripper=True) # Ensure gripper is open

        # --- Reset Blocks ---
        for block_id in self.block_ids:
            try: # Body might already be removed
                 p.removeBody(block_id, physicsClientId=self.client)
            except Exception as e: pass
        self.block_ids = []
        initial_loc_indices = random.sample(range(self.num_locations), self.num_blocks)
        for i in range(self.num_blocks):
            loc_idx = initial_loc_indices[i]
            pos = self.fixed_locations[loc_idx]
            block_start_ori = p.getQuaternionFromEuler([0, 0, random.uniform(0, 2*np.pi)]) # Random Z rot
            # Add Z offset = half block height + tiny buffer
            block_id = p.loadURDF(self.block_urdf_path, [pos[0], pos[1], pos[2] + 0.025], block_start_ori, physicsClientId=self.client)
            # TODO: Set block color/visuals if needed for distinction
            # p.changeVisualShape(block_id, -1, rgbaColor=[...])
            self.block_ids.append(block_id)

        # --- Set Goal ---
        # TODO: Define the target configuration dynamically or statically
        # Example: Move block 0 (first loaded) to location 0
        self.goal_config = {0: 0}

        # --- Settle ---
        self._wait_steps(100) # Let objects settle

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Execute the primitive corresponding to the action
        success = self._execute_primitive(action) # Primitive might return success/fail

        observation = self._get_obs()
        terminated = self._check_goal()
        reward = self.goal_reward if terminated else self.step_penalty
        self.current_steps += 1
        truncated = self.current_steps >= self.max_steps

        # Optional: Add penalty if primitive failed?
        # if not success: reward -= 0.1

        info = self._get_info()
        info['primitive_success'] = success # Add auxiliary info

        return observation, reward, terminated, truncated, info

    def _execute_primitive(self, action_index):
        """ Executes the motion primitive corresponding to the action index. Returns True if successful."""
        # Action mapping: 0..M-1: Move_Above(Loc), M: Lower_Grasp, M+1: Raise, M+2: Release
        target_ori_ee = p.getQuaternionFromEuler([np.pi, 0, 0]) # Default: Point down

        try:
            if action_index < self.num_locations: # Move Above Location
                loc_idx = action_index
                target_pos_table = self.fixed_locations[loc_idx]
                target_pos_ee = [target_pos_table[0], target_pos_table[1], target_pos_table[2] + 0.15] # Z offset above location
                return self._move_ee_to_pose(target_pos_ee, target_ori_ee)

            elif action_index == self.num_locations: # Lower and Grasp
                if self.held_object_id is not None: return False # Cannot grasp if already holding

                current_pose_pos, current_pose_ori = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.client)[4:6] # World link frame pose
                target_pos_ee_down = [current_pose_pos[0], current_pose_pos[1], self.table_height + 0.03] # Lower Z slightly above table

                # Move down first
                if not self._move_ee_to_pose(target_pos_ee_down, current_pose_ori): return False # Use current orientation
                self._wait_steps(20)

                # Close gripper
                if not self._set_gripper(open_gripper=False): return False
                self._wait_steps(50)

                # Check for grasped object and optionally create constraint
                grasped_obj_id = self._get_object_in_gripper(check_dist=0.04) # Check slightly larger radius
                if grasped_obj_id is not None:
                    self.held_object_id = grasped_obj_id
                    # Create constraint (optional but recommended for stability)
                    ee_state = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.client)
                    obj_pos, obj_ori = p.getBasePositionAndOrientation(self.held_object_id, physicsClientId=self.client)
                    # Calculate relative pose from EE to object
                    inv_ee_pos, inv_ee_ori = p.invertTransform(ee_state[4], ee_state[5])
                    rel_pos, rel_ori = p.multiplyTransforms(inv_ee_pos, inv_ee_ori, obj_pos, obj_ori)
                    # Create fixed constraint
                    self.grasp_constraint_id = p.createConstraint(
                        parentBodyUniqueId=self.robot_id,
                        parentLinkIndex=self.ee_link_index,
                        childBodyUniqueId=self.held_object_id,
                        childLinkIndex=-1, # Base link of object
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=rel_pos,
                        childFramePosition=[0, 0, 0],
                        parentFrameOrientation=rel_ori,
                        childFrameOrientation=[0,0,0,1],
                        physicsClientId=self.client
                    )
                    return True
                else: # Failed grasp, reopen
                    self._set_gripper(open_gripper=True)
                    return False # Indicate grasp failure

            elif action_index == self.num_locations + 1: # Raise Gripper
                current_pose_pos, current_pose_ori = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.client)[4:6]
                target_pos_ee = [current_pose_pos[0], current_pose_pos[1], self.table_height + 0.15] # Raise Z
                return self._move_ee_to_pose(target_pos_ee, current_pose_ori) # Keep current orientation

            elif action_index == self.num_locations + 2: # Release
                if self.held_object_id is None: return False # Cannot release if not holding

                # Remove constraint FIRST
                if self.grasp_constraint_id is not None:
                     try: p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
                     except Exception as e: pass # Ignore error if already removed
                     self.grasp_constraint_id = None

                # Open gripper
                success = self._set_gripper(open_gripper=True)
                self.held_object_id = None
                self._wait_steps(50) # Allow time for release
                return success

            else:
                print(f"Warning: Unknown action index {action_index}")
                return False

        except Exception as e:
            print(f"Error during primitive execution for action {action_index}: {e}")
            return False # Indicate failure

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
            # TODO: Add null space control, joint limits etc. for robustness
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
        movable_joint_indices = [i for i in range(num_joints) if p.getJointInfo(self.robot_id, i, physicsClientId=self.client)[2] != p.JOINT_FIXED]

        if len(joint_poses) < len(movable_joint_indices):
             print(f"Warning: IK solution size {len(joint_poses)} doesn't match movable joints {len(movable_joint_indices)}")
             return False # Or handle partial solution if appropriate

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=movable_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_poses[:len(movable_joint_indices)],
            # TODO: Tune forces/gains for smoother/faster movement
            forces=[100.0] * len(movable_joint_indices),
            positionGains=[0.03] * len(movable_joint_indices), # Example gains
            velocityGains=[1.0] * len(movable_joint_indices), # Example gains
            physicsClientId=self.client
        )

        # Step simulation to allow movement
        for _ in range(max_steps):
            p.stepSimulation(self.client)
            if self.use_gui: time.sleep(1./240.)
            # TODO: Add check if target pose is reached within tolerance?
            # current_ee_pos = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.client)[4]
            # if np.linalg.norm(np.array(current_ee_pos) - np.array(target_pos)) < 0.01:
            #     return True # Reached target

        # Check final pose (optional, depends if exact pose needed)
        final_ee_pos = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.client)[4]
        if np.linalg.norm(np.array(final_ee_pos) - np.array(target_pos)) < 0.02: # Looser tolerance after timeout
             return True
        else:
             print(f"Warning: Move EE failed to reach target. Final dist: {np.linalg.norm(np.array(final_ee_pos) - np.array(target_pos)):.3f}")
             return False


    def _set_gripper(self, open_gripper):
        """ Opens or closes the gripper by setting target position for gripper joints. """
        target_val = self.gripper_open_value if open_gripper else self.gripper_closed_value
        # TODO: Check if your gripper needs position, velocity, or torque control
        # TODO: Check if it's one joint or multiple (e.g., mimic joints might be handled by simulator)
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.gripper_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[target_val] * len(self.gripper_joint_indices),
            forces=[50.0] * len(self.gripper_joint_indices) # Adjust force as needed
            # positionGains=[0.03] * len(self.gripper_joint_indices),
            # velocityGains=[1.0] * len(self.gripper_joint_indices),
             , physicsClientId=self.client
        )
        # Allow time for gripper to move
        self._wait_steps(30)
        # Optional: Check if gripper reached target state?
        # current_gripper_state = p.getJointState(self.robot_id, self.gripper_joint_indices[0], physicsClientId=self.client)[0]
        # if abs(current_gripper_state - target_val) < 0.01: return True else: return False
        return True # Assume success for now


    def _get_object_in_gripper(self, check_dist=0.05):
        """ Simple check if any manipulable block is close to the gripper center. """
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.client)
        ee_pos = ee_state[4] # World position of EE link

        for block_id in self.block_ids:
             try:
                 block_pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self.client)
                 dist = np.linalg.norm(np.array(ee_pos) - np.array(block_pos))
                 if dist < check_dist:
                     # Optional: Check if block is roughly 'below' EE?
                     if block_pos[2] < ee_pos[2]:
                         return block_id
             except Exception as e:
                 # Block might have been removed or invalid
                 continue
        return None

    def _wait_steps(self, steps):
        """ Steps simulation for a number of steps. """
        for _ in range(steps):
            p.stepSimulation(self.client)
            if self.use_gui: time.sleep(1./240.) # Adjust sleep for desired sim speed in GUI

    def _get_obs(self):
        """ Renders the environment image. """
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
            farVal=10.0,
            physicsClientId=self.client
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=self.image_size,
            height=self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL, # Use faster renderer
            physicsClientId=self.client
        )
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3] # Remove alpha channel
        return rgb_array

    def _get_info(self):
        """ Returns auxiliary environment info. """
        # Example: Return current block poses relative to target locations
        info = {}
        # TODO: Add relevant info if needed by agent or for logging
        # e.g., info['held_object'] = self.held_object_id
        return info

    def _check_goal(self):
        """ Checks if the current block configuration matches the goal. """
        # TODO: Implement robust check based on self.goal_config
        on_target_count = 0
        for block_idx, target_loc_idx in self.goal_config.items():
            if block_idx < len(self.block_ids):
                block_id = self.block_ids[block_idx]
                target_pos = self.fixed_locations[target_loc_idx]
                try:
                    current_pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self.client)
                    dist = np.linalg.norm(np.array(current_pos[:2]) - np.array(target_pos[:2])) # Check XY distance
                    z_dist = abs(current_pos[2] - target_pos[2])
                    if dist < 0.03 and z_dist < 0.03: # Distance thresholds for goal check
                        on_target_count += 1
                except Exception as e:
                    continue # Block may not exist
        return on_target_count == len(self.goal_config)


    def render(self, mode='human'):
        # PyBullet handles 'human' mode rendering if connected with p.GUI
        # For 'rgb_array', return the observation
        if mode == 'rgb_array':
            return self._get_obs()
        elif mode == 'human':
            # GUI connection handles rendering, maybe add a small delay?
            if self.use_gui: time.sleep(0.01)
            return None
        else:
            super(PhysicsBlockRearrangementEnv, self).render(mode=mode) # Raise error for unsupported modes

    def close(self):
        if self.client >= 0:
            try:
                p.disconnect(physicsClientId=self.client)
            except Exception as e:
                pass
            self.client = -1

# Example usage (if run directly)
if __name__ == '__main__':
    # Example of how to use the environment
    env = PhysicsBlockRearrangementEnv(use_gui=True, render_mode='human')
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Action space size:", env.action_space.n)

    for episode in range(3):
        obs, info = env.reset()
        terminated = False
        truncated = False
        step = 0
        while not terminated and not truncated:
            action = env.action_space.sample() # Take random actions
            print(f"Step: {step}, Action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            env.render() # No-op in GUI mode usually, but good practice
            step += 1
            if terminated: print("Goal Reached!")
            if truncated: print("Max steps reached.")
        print(f"Episode {episode+1} finished after {step} steps.")

    env.close()