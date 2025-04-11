# scripts/test_panda_ik_control_franka_inspired_grasp.py

import pybullet as p
import pybullet_data
import time
import os
import numpy as np

# --- Configuration Flags ---
USE_DLS_SOLVER = True
PREFERRED_EE_LINK_NAME = "panda_hand"
FALLBACK_EE_LINK_NAME = "panda_link7"
FINAL_FALLBACK_EE_LINK_NAME = "panda_link8"

# --- Parameters ---
PANDA_URDF_PATH = "franka_panda/panda.urdf"
TABLE_URDF_PATH = "table/table.urdf"
# Use a standard cube URDF from pybullet_data
OBJECT_URDF_PATH = "cube_small.urdf" # Or "cube.urdf" or "block.urdf"

ARM_JOINT_NAMES = [f"panda_joint{i+1}" for i in range(7)]

# --- Control Parameters ---
MAX_FORCES = [100.0] * 7
POSITION_GAINS = [0.05] * 7
VELOCITY_GAINS = [1.0] * 7

# --- Gripper Constants ---
finger_indices = []
GRIPPER_OPEN_VALUE = 0.04
GRIPPER_CLOSED_VALUE = 0.005 # Adjust based on object size if not using constraints
GRIPPER_MAX_FORCE = 40 # Slightly increased force for grasping maybe
GRIPPER_KP = 0.2
GRIPPER_KD = 1.0
GRIPPER_WAIT_STEPS = 120

# --- Simulation & Task Parameters ---
PRIMITIVE_MAX_STEPS = 400
POSE_REACHED_THRESHOLD = 0.02
ORIENTATION_REACHED_THRESHOLD = 0.1
# Define locations
OBJECT_START_POS = [0.5, 0.0, 0.0] # Base position on table, Z will be adjusted
PLACE_POS = [0.5, 0.3, 0.0]      # Where to place the object
# Define offsets for grasping
Z_HOVER_OFFSET = 0.15  # How far above the object to hover
Z_GRASP_OFFSET = 0.01 # How far above the object's center to grasp (tune this)
GRASP_CLEARANCE_ABOVE_TOP = 0.075
# Constraint ID for attaching object
grasp_constraint_id = None


# --- Helper Functions ---
# find_link_index_safely, find_joint_indices, get_arm_kinematic_limits_and_ranges, wait_steps
# (Keep these functions as they were in the last working version)
def find_link_index_safely(robot_id, link_name, client_id):
    num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i, physicsClientId=client_id)
        try:
            link_name_decoded = info[12].decode('UTF-8')
            if link_name_decoded == link_name: return i
        except UnicodeDecodeError: continue
    try:
         base_info = p.getBodyInfo(robot_id, physicsClientId=client_id)
         if base_info[0].decode('UTF-8') == link_name: return -1
    except Exception: pass
    print(f"Warning: Link '{link_name}' not found via getJointInfo.")
    return None

def find_joint_indices(robot_id, joint_names, client_id):
    num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
    name_to_index_map = {}
    for i in range(num_joints):
        try:
             joint_name = p.getJointInfo(robot_id, i, physicsClientId=client_id)[1].decode('UTF-8')
             name_to_index_map[joint_name] = i
        except UnicodeDecodeError: continue
    indices = []; missing = []
    for name in joint_names:
        if name in name_to_index_map: indices.append(name_to_index_map[name])
        else: missing.append(name)
    if missing: raise ValueError(f"Joint(s) not found: {', '.join(missing)}")
    return indices

def get_arm_kinematic_limits_and_ranges(robot_id, arm_joint_indices, client_id):
    lower_limits = []; upper_limits = []; joint_ranges = []
    for i in arm_joint_indices:
        info = p.getJointInfo(robot_id, i, physicsClientId=client_id)
        ll, ul = info[8], info[9]
        if ll > ul: ll, ul = -2*np.pi, 2*np.pi
        lower_limits.append(ll); upper_limits.append(ul); joint_ranges.append(ul - ll)
    return lower_limits, upper_limits, joint_ranges

def wait_steps(client_id, steps, use_gui):
    for _ in range(steps):
        p.stepSimulation(physicsClientId=client_id)
        if use_gui: time.sleep(1./240.)


# --- Gripper Control Functions ---
# control_gripper, open_gripper, close_gripper
# (Keep these functions as they were)
def control_gripper(robot_id, target_val, force, kp, kd, client_id):
    global finger_indices
    if not finger_indices: print("Error: Gripper indices not set."); return False
    if len(finger_indices) != 2: print("Error: Expected 2 finger indices."); return False
    p.setJointMotorControlArray(robot_id,finger_indices,p.POSITION_CONTROL,
        targetPositions=[target_val]*2, forces=[force]*2, positionGains=[kp]*2, velocityGains=[kd]*2,
        physicsClientId=client_id)
    return True

def open_gripper(robot_id, client_id, wait=True, use_gui=True):
    print("Commanding gripper OPEN")
    if control_gripper(robot_id, GRIPPER_OPEN_VALUE, GRIPPER_MAX_FORCE, GRIPPER_KP, GRIPPER_KD, client_id):
        if wait: wait_steps(client_id, GRIPPER_WAIT_STEPS, use_gui)

def close_gripper(robot_id, client_id, wait=True, use_gui=True):
    print("Commanding gripper CLOSE")
    if control_gripper(robot_id, GRIPPER_CLOSED_VALUE, GRIPPER_MAX_FORCE, GRIPPER_KP, GRIPPER_KD, client_id):
        if wait: wait_steps(client_id, GRIPPER_WAIT_STEPS, use_gui)


# --- Motion Planning and Execution ---
def move_to_pose(robot_id, target_pos, target_ori, ee_link_index, arm_joint_indices, limits, client_id, use_gui=True):
    """ Calculates IK and commands arm to target pose. Returns True if successful. """
    ll, ul, jr = limits # Unpack limits
    # Use the neutral pose as rest pose bias
    rest_poses = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
    solver = p.IK_DLS if USE_DLS_SOLVER else 0

    print(f"  Attempting IK for Pose: Pos={np.round(target_pos, 3)}, Ori={np.round(p.getEulerFromQuaternion(target_ori), 2)}")
    start_ik_time = time.time()
    joint_poses = p.calculateInverseKinematics(
        robot_id, ee_link_index, target_pos, target_ori,
        lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rest_poses,
        solver=solver, maxNumIterations=200, residualThreshold=1e-4, physicsClientId=client_id
    )
    ik_time = time.time() - start_ik_time

    valid_solution = joint_poses is not None and len(joint_poses) >= len(arm_joint_indices)

    if valid_solution:
        arm_joint_poses = joint_poses[:len(arm_joint_indices)]
        print(f"  ----> IK Succeeded! (Took {ik_time:.4f} s)")
        print(f"        Solution: {np.round(arm_joint_poses, 3)}")
        print("        Commanding Arm Motion...")
        p.setJointMotorControlArray(robot_id, arm_joint_indices, p.POSITION_CONTROL, arm_joint_poses, forces=MAX_FORCES, positionGains=POSITION_GAINS, velocityGains=VELOCITY_GAINS, physicsClientId=client_id)
        wait_steps(client_id, PRIMITIVE_MAX_STEPS, use_gui)

        # Check final pose
        final_ee_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True, physicsClientId=client_id)
        final_ee_pos, final_ee_ori = final_ee_state[4:6]
        final_dist = np.linalg.norm(np.array(final_ee_pos) - np.array(target_pos))
        ori_diff = p.getDifferenceQuaternion(target_ori, final_ee_ori); _, ori_angle = p.getAxisAngleFromQuaternion(ori_diff)
        pos_ok = final_dist < POSE_REACHED_THRESHOLD
        ori_ok = abs(ori_angle) < ORIENTATION_REACHED_THRESHOLD
        if pos_ok and ori_ok: print(f"        SUCCESS: Reached Target Pose! Dist: {final_dist:.4f}, Ori angle err: {abs(ori_angle):.4f}")
        else: print(f"        FAILURE: Did not reach Target Pose. Dist: {final_dist:.4f} (OK={pos_ok}), Ori angle err: {abs(ori_angle):.4f} (OK={ori_ok})")
        return pos_ok and ori_ok # Return True only if pose reached accurately
    else:
        print(f"  ----> IK Failed! (Took {ik_time:.4f} s)")
        return False

# --- Main Execution ---
# --- Main Execution ---
if __name__ == "__main__":
    use_gui = True
    client = p.connect(p.GUI if use_gui else p.DIRECT)
    # --- Basic Setup ---
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
    p.resetDebugVisualizerCamera(1.5, 50, -35, [0.5, 0, 0.5], physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setRealTimeSimulation(0, physicsClientId=client) # Keep simulation stepped

    print(f"--- Testing Panda Pick and Place ---")

    # --- Load Scene (Table) ---
    p.loadURDF("plane.urdf", physicsClientId=client)
    table_pos = [0.5, 0, 0]
    table_id = p.loadURDF(TABLE_URDF_PATH, basePosition=table_pos, useFixedBase=True, physicsClientId=client)
    table_aabb = p.getAABB(table_id, -1, physicsClientId=client)
    table_height = table_aabb[1][2]

    # --- Load Robot ---
    robot_pos = [0, 0, table_height]
    robot_ori = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = -1 # Initialize
    try:
        robot_id = p.loadURDF(PANDA_URDF_PATH, robot_pos, robot_ori, useFixedBase=True,
                              flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
                              physicsClientId=client)
        print(f"Loaded robot with ID: {robot_id}")
    except Exception as e: print(f"LOAD FAILED: {e}"); p.disconnect(client); exit()

    # --- IMMEDIATELY INITIALIZE ROBOT STATE ---
    ee_link_index = None
    ee_link_name = ""
    arm_limits = None
    arm_joint_indices = []
    # Define the desired home pose here
    home_pose_joints = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
    rest_poses_default = home_pose_joints # Use this for IK bias later

    try:
        # 1. Find necessary indices and info FIRST
        arm_joint_indices = find_joint_indices(robot_id, ARM_JOINT_NAMES, client)
        finger_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        finger_indices = find_joint_indices(robot_id, finger_joint_names, client)
        print(f"Found Arm Joints: {arm_joint_indices}")
        print(f"Found Gripper Fingers: {finger_indices}")

        # Find EE Link (needed for move_to_pose function later)
        preferred_link_idx = find_link_index_safely(robot_id, PREFERRED_EE_LINK_NAME, client)
        if preferred_link_idx is not None: ee_link_index, ee_link_name = preferred_link_idx, PREFERRED_EE_LINK_NAME
        else: # Fallback logic...
            fallback_link_idx = find_link_index_safely(robot_id, FALLBACK_EE_LINK_NAME, client)
            if fallback_link_idx is not None: ee_link_index, ee_link_name = fallback_link_idx, FALLBACK_EE_LINK_NAME
            else:
                 final_fallback_link_idx = find_link_index_safely(robot_id, FINAL_FALLBACK_EE_LINK_NAME, client)
                 if final_fallback_link_idx is not None: ee_link_index, ee_link_name = final_fallback_link_idx, FINAL_FALLBACK_EE_LINK_NAME
                 else: raise ValueError("Could not find suitable EE Link.")
        print(f"Using EE Link '{ee_link_name}' at index: {ee_link_index}")

        # Get Arm Limits (needed for move_to_pose function later)
        arm_limits = get_arm_kinematic_limits_and_ranges(robot_id, arm_joint_indices, client)

        # 2. Reset Arm Joints State & Enable Motors
        print("Initializing arm joints to home pose...")
        for i, idx in enumerate(arm_joint_indices):
            # Set state directly
            p.resetJointState(robot_id, idx, home_pose_joints[i], targetVelocity=0.0, physicsClientId=client)
            # Enable position control motor to hold the pose
            p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL,
                                    targetPosition=home_pose_joints[i],
                                    force=MAX_FORCES[i],
                                    positionGain=POSITION_GAINS[i], # Use arm gains
                                    velocityGain=VELOCITY_GAINS[i], # Use arm gains
                                    physicsClientId=client)

        # 3. Reset Gripper State & Enable Motors
        print("Initializing gripper to OPEN state...")
        if finger_indices:
            # Set state directly
            p.resetJointState(robot_id, finger_indices[0], GRIPPER_OPEN_VALUE, 0.0, client)
            p.resetJointState(robot_id, finger_indices[1], GRIPPER_OPEN_VALUE, 0.0, client)
            # Enable position control motor using the gripper control function
            control_gripper(robot_id, GRIPPER_OPEN_VALUE, GRIPPER_MAX_FORCE, GRIPPER_KP, GRIPPER_KD, client)

        # 4. Let simulation run briefly to stabilize the initialized state
        print("Stabilizing robot in home pose...")
        wait_steps(client, 100, use_gui) # Short wait
        print("Robot Initialized.")
        time.sleep(0.1) # Very short pause

    except Exception as e:
        print(f"ERROR during robot initialization: {e}")
        p.disconnect(client)
        exit()
    # --- ROBOT IS NOW INITIALIZED AND STABLE ---

    # --- Load Object ---
    object_id = -1
    obj_start_pos_actual = [0, 0, 0]
    obj_start_ori = [0, 0, 0, 1]
    # IMPORTANT: Make sure these approximate dimensions match your object URDF
    object_half_extents = [0.025, 0.025, 0.025]  # Approx for cube_small.urdf (0.05m side)
    try:
        object_start_z = table_height + object_half_extents[2]
        obj_start_pos_initial = [OBJECT_START_POS[0], OBJECT_START_POS[1], object_start_z]
        obj_start_ori_q = p.getQuaternionFromEuler([0, 0, 0])  # Use consistent naming
        object_id = p.loadURDF(OBJECT_URDF_PATH, obj_start_pos_initial, obj_start_ori_q, physicsClientId=client)
        print(f"Loaded object '{OBJECT_URDF_PATH}' with ID: {object_id} at {np.round(obj_start_pos_initial, 3)}")
        wait_steps(client, 100, use_gui)
        obj_start_pos_actual, obj_start_ori_q = p.getBasePositionAndOrientation(object_id, physicsClientId=client)
        print(f"Object settled at: {np.round(obj_start_pos_actual, 3)}")
        if obj_start_pos_actual[2] < table_height * 0.9:  # Check if significantly below table
            raise Exception(
                f"Object fell through table! Settled Z: {obj_start_pos_actual[2]:.3f}, Table Z: {table_height:.3f}")
    except Exception as e:
        print(f"Failed to load object: {e}"); p.disconnect(client); exit()

    # --- Define Task Poses & Orientation ---
    ori_down = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])

    # Use object's settled position
    obj_pos = np.array(obj_start_pos_actual)
    object_top_z = obj_pos[2] + object_half_extents[2]  # Calculate top surface Z

    # Poses calculated relative to object TOP surface Z
    pre_grasp_pos_z = object_top_z + Z_HOVER_OFFSET
    pre_grasp_pos = [obj_pos[0], obj_pos[1], pre_grasp_pos_z]

    grasp_pos_z = object_top_z + GRASP_CLEARANCE_ABOVE_TOP  # Use new clearance parameter
    grasp_pos = [obj_pos[0], obj_pos[1], grasp_pos_z]

    lift_pos_z = object_top_z + Z_HOVER_OFFSET + 0.05  # Lift slightly higher than hover
    lift_pos = [obj_pos[0], obj_pos[1], lift_pos_z]

    # Poses relative to place location (adjust Z based on object height)
    place_target_z = table_height + object_half_extents[2]  # Target Z for the object base when placed
    pre_place_pos_z = place_target_z + object_half_extents[2] + Z_HOVER_OFFSET  # Hover above place spot top
    pre_place_pos = [PLACE_POS[0], PLACE_POS[1], pre_place_pos_z]

    place_pos_z = place_target_z + object_half_extents[
        2] + GRASP_CLEARANCE_ABOVE_TOP  # Place EEF at same clearance above surface
    place_pos = [PLACE_POS[0], PLACE_POS[1], place_pos_z]

    # Add printouts to verify calculations
    print("-" * 20)
    print(f"Table Height Z: {table_height:.3f}")
    print(f"Object Settled Base Z: {obj_pos[2]:.3f}")
    print(f"Object Est. Top Z: {object_top_z:.3f}")
    print(f"Calculated Pre-Grasp Z: {pre_grasp_pos[2]:.3f}")
    print(f"Calculated Grasp Z: {grasp_pos[2]:.3f}")
    print(f"Calculated Lift Z: {lift_pos[2]:.3f}")
    print(f"Calculated Pre-Place Z: {pre_place_pos[2]:.3f}")
    print(f"Calculated Place Z: {place_pos[2]:.3f}")
    print("-" * 20)


    # --- Execute Pick and Place ---
    # (The pick and place try/except/finally block - same as before)
    # ... (Rest of the pick/place code) ...
    grasp_success = False
    grasp_constraint_id = None # Ensure it's defined before try block
    try:
        print("\n--- Starting Pick Sequence ---")
        # 1. Open gripper (it should already be open from init)
        # open_gripper(robot_id, client, use_gui=use_gui) # Optional re-open

        # 2. Move to hover position above object
        print("\n1. Moving to Pre-Grasp Hover")
        if not move_to_pose(robot_id, pre_grasp_pos, ori_down, ee_link_index, arm_joint_indices, arm_limits, client, use_gui): raise Exception("Failed to reach pre-grasp pose")
        # ... (Rest of pick sequence: move down, close, constrain, lift) ...
        print("\n2. Moving Down to Grasp")
        if not move_to_pose(robot_id, grasp_pos, ori_down, ee_link_index, arm_joint_indices, arm_limits, client, use_gui): raise Exception("Failed to reach grasp pose")
        print("\n3. Closing Gripper")
        close_gripper(robot_id, client, use_gui=use_gui)
        print("\n4. Attaching Object via Constraint")
        hand_link_state = p.getLinkState(robot_id, ee_link_index, physicsClientId=client)
        hand_pos, hand_ori = hand_link_state[0], hand_link_state[1]
        obj_pos_world, obj_ori_world = p.getBasePositionAndOrientation(object_id, physicsClientId=client)
        inv_hand_pos, inv_hand_ori = p.invertTransform(hand_pos, hand_ori)
        obj_pos_in_hand, obj_ori_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_ori, obj_pos_world, obj_ori_world)
        grasp_constraint_id = p.createConstraint(robot_id, ee_link_index, object_id, -1, p.JOINT_FIXED,[0,0,0], obj_pos_in_hand, [0,0,0], parentFrameOrientation=obj_ori_in_hand, childFrameOrientation=[0,0,0,1], physicsClientId=client)
        if grasp_constraint_id < 0: raise Exception("Failed to create grasp constraint")
        print(f"  Constraint ID: {grasp_constraint_id}")
        wait_steps(client, 50, use_gui)
        print("\n5. Lifting Object")
        if not move_to_pose(robot_id, lift_pos, ori_down, ee_link_index, arm_joint_indices, arm_limits, client, use_gui): raise Exception("Failed to lift object")
        grasp_success = True

        # --- Place Sequence ---
        print("\n--- Starting Place Sequence ---")
        # ... (Rest of place sequence: move pre-place, move place, open, remove constraint, move up) ...
        print("\n6. Moving to Pre-Place Hover")
        if not move_to_pose(robot_id, pre_place_pos, ori_down, ee_link_index, arm_joint_indices, arm_limits, client, use_gui): raise Exception("Failed to reach pre-place pose")
        print("\n7. Moving Down to Place")
        if not move_to_pose(robot_id, place_pos, ori_down, ee_link_index, arm_joint_indices, arm_limits, client, use_gui): print("Warning: Place position might not be perfectly reached.")
        print("\n8. Opening Gripper to Release")
        open_gripper(robot_id, client, use_gui=use_gui)
        print("\n9. Detaching Object (Removing Constraint)")
        if grasp_constraint_id is not None:
            p.removeConstraint(grasp_constraint_id, physicsClientId=client)
            grasp_constraint_id = None
        wait_steps(client, 50, use_gui)
        print("\n10. Moving Arm Up After Place")
        post_place_pos = [PLACE_POS[0], PLACE_POS[1], table_height + Z_HOVER_OFFSET]
        move_to_pose(robot_id, post_place_pos, ori_down, ee_link_index, arm_joint_indices, arm_limits, client, use_gui)

    except Exception as e:
        print(f"!!!! Task Failed: {e} !!!!")
        if grasp_constraint_id is not None:
            try: p.removeConstraint(grasp_constraint_id, physicsClientId=client)
            except: pass

    finally:
        print("\n--- Pick and Place Sequence Finished ---")
        print("\nTest finished. Press Enter to exit.")
        input()
        p.disconnect(client)