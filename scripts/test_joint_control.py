# scripts/test_gripper_standalone.py

import pybullet as p
import pybullet_data
import time
import os
import numpy as np

# --- Parameters to Verify/Adjust ---
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Make sure this path is correct!
ROBOT_URDF_PATH = os.path.join(
    PROJECT_ROOT,
    "physics_block_rearrangement_env",
    "assets",
    "urdf",
    "robots",
    "ur3e_robotiq",
    "ur3e_robotiq_140.urdf" # Check your final URDF filename
)
ROBOT_START_POS = [0, 0, 0.63] # Place it slightly above ground for clarity
ROBOT_START_ORI = p.getQuaternionFromEuler([0, 0, 0])

# Verify these joint names from your URDF
MAIN_GRIPPER_JOINT_NAME = "finger_joint"
MIMIC_JOINT_NAMES = [
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint"
]
# Verify these multipliers from the <mimic> tags in your URDF!
MIMIC_MULTIPLIERS = [-1, -1, -1, 1, 1] # Should be this based on URDF snippet

# Verify these values based on URDF limits and visual testing
# Assuming limits [0, 0.7] and 0=Open, 0.7=Closed
GRIPPER_OPEN_VAL = 0.0
GRIPPER_CLOSED_VAL = 0.68 # Slightly less than limit 0.7

# Test different forces
GRIPPER_FORCE = 300.0
# --- End Parameters ---

def find_joint_indices(robot_id, joint_names, client_id):
    """ Utility to find multiple joint indices by name. """
    indices = []
    num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
    joint_info_dict = {p.getJointInfo(robot_id, i, physicsClientId=client_id)[1].decode('UTF-8'): i for i in range(num_joints)}

    for name_to_find in joint_names:
        if name_to_find in joint_info_dict:
            indices.append(joint_info_dict[name_to_find])
        else:
            raise ValueError(f"Gripper joint '{name_to_find}' not found in URDF.")
    return indices

def set_gripper_explicit(robot_id, open_gripper, client_id, main_idx, mimic_indices, mimic_multipliers, open_val, closed_val, force):
    """ Opens or closes the gripper by setting target position for ALL related joints explicitly. """
    target_val_main = open_val if open_gripper else closed_val
    target_mimic_vals = [m * target_val_main for m in mimic_multipliers]
    all_indices = [main_idx] + mimic_indices
    all_target_positions = [target_val_main] + target_mimic_vals

    # print(f"DEBUG: Controlling gripper joints {all_indices} to targets {np.round(all_target_positions, 3)}")
    try:
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=all_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=all_target_positions,
            forces=[force] * len(all_indices),
            physicsClientId=client_id
        )
        return True
    except Exception as e:
        print(f"Error setting gripper joints explicitly: {e}")
        return False

def wait_steps(client_id, steps, use_gui):
    """ Steps simulation for a number of steps. """
    for _ in range(steps):
        p.stepSimulation(physicsClientId=client_id)
        if use_gui: time.sleep(1./240.)

# --- Main Test Script ---
if __name__ == "__main__":
    use_gui = True
    client = p.connect(p.GUI if use_gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("Failed to connect to PyBullet")

    print(f"Using GUI: {use_gui}")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[ROBOT_START_POS[0], ROBOT_START_POS[1], ROBOT_START_POS[2]+0.3])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    # Load plane and robot
    p.loadURDF("plane.urdf", physicsClientId=client)
    try:
        robot_id = p.loadURDF(ROBOT_URDF_PATH, ROBOT_START_POS, ROBOT_START_ORI, useFixedBase=True, physicsClientId=client)
        print(f"Loaded robot from {ROBOT_URDF_PATH}")
    except Exception as e:
        print(f"***** Failed to load robot URDF: {e} *****")
        print("Please check ROBOT_URDF_PATH and mesh paths inside the URDF.")
        p.disconnect(client)
        exit()

    # --- Find Gripper Joints ---
    try:
        main_gripper_idx = find_joint_indices(robot_id, [MAIN_GRIPPER_JOINT_NAME], client)[0]
        mimic_gripper_indices = find_joint_indices(robot_id, MIMIC_JOINT_NAMES, client)
        print(f"Found main gripper joint '{MAIN_GRIPPER_JOINT_NAME}' at index: {main_gripper_idx}")
        print(f"Found mimic gripper joints '{MIMIC_JOINT_NAMES}' at indices: {mimic_gripper_indices}")
    except ValueError as e:
        print(f"ERROR finding joint indices: {e}")
        p.disconnect(client)
        exit()

    # --- Reset Arm Pose (Optional but good) ---
    arm_joint_indices = []
    num_joints = p.getNumJoints(robot_id, physicsClientId=client)
    for i in range(num_joints):
         info = p.getJointInfo(robot_id, i, physicsClientId=client)
         if info[2] == p.JOINT_REVOLUTE and info[8] < info[9] and len(arm_joint_indices) < 6:
             arm_joint_indices.append(i)
    neutral_pose_angles = [0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0] # Example UR3e pose
    if len(arm_joint_indices) == 6:
        for i, idx in enumerate(arm_joint_indices):
            p.resetJointState(robot_id, idx, neutral_pose_angles[i], physicsClientId=client)
        print("Reset arm pose to neutral.")
        wait_steps(client, 50, use_gui) # Settle

    # --- Test Loop ---
    print("\nStarting gripper open/close test loop...")
    while p.isConnected(client):
        # OPEN
        print("Commanding OPEN...")
        set_gripper_explicit(robot_id, True, client, main_gripper_idx, mimic_gripper_indices, MIMIC_MULTIPLIERS, GRIPPER_OPEN_VAL, GRIPPER_CLOSED_VAL, GRIPPER_FORCE)
        wait_steps(client, 100, use_gui) # Wait longer to see state
        time.sleep(1.0 if use_gui else 0) # Pause for visual inspection

        # CLOSE
        print("Commanding CLOSE...")
        set_gripper_explicit(robot_id, False, client, main_gripper_idx, mimic_gripper_indices, MIMIC_MULTIPLIERS, GRIPPER_OPEN_VAL, GRIPPER_CLOSED_VAL, GRIPPER_FORCE)
        wait_steps(client, 100, use_gui)
        time.sleep(1.0 if use_gui else 0)