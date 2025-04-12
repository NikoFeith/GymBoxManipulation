import pybullet as p
import numpy as np
import time

def find_link_index_safely(robot_id, link_name, client):
    """ Safely finds link index by name. """
    num_joints = p.getNumJoints(robot_id, physicsClientId=client)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i, physicsClientId=client)
        try:
            if info[12].decode('UTF-8') == link_name: return i
        except UnicodeDecodeError:
            continue
    try:
        base_info = p.getBodyInfo(robot_id, physicsClientId=client)
        if base_info[0].decode('UTF-8') == link_name: return -1
    except Exception:
        pass
    print(f"Warning: Link '{link_name}' not found.")
    return None


def find_joint_indices(robot_id, joint_names, client):
    """ Finds multiple joint indices by name, handles errors. """
    num_joints = p.getNumJoints(robot_id, physicsClientId=client)
    name_to_index_map = {}
    for i in range(num_joints):
        try:
            joint_name = p.getJointInfo(robot_id, i, physicsClientId=client)[1].decode('UTF-8')
            name_to_index_map[joint_name] = i
        except UnicodeDecodeError:
            continue
    indices = []
    missing = []
    for name in joint_names:
        if name in name_to_index_map:
            indices.append(name_to_index_map[name])
        else:
            missing.append(name)
    if missing: raise ValueError(f"Joint(s) not found: {', '.join(missing)}")
    return indices


def get_arm_kinematic_limits_and_ranges(robot_id,arm_joint_indices, client):
    """ Gets limits and ranges tuple (ll, ul, jr) for arm joints. """
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    for i in arm_joint_indices:
        info = p.getJointInfo(robot_id, i, physicsClientId=client)
        ll, ul = info[8], info[9]
        if ll > ul: ll, ul = -2 * np.pi, 2 * np.pi
        lower_limits.append(ll)
        upper_limits.append(ul)
        joint_ranges.append(ul - ll)
    return lower_limits, upper_limits, joint_ranges


def wait_steps(steps, client, timestep=None ,use_gui=False):
    """ Steps simulation for a number of steps. """
    # Re-use implementation from test script
    for _ in range(steps):
        p.stepSimulation(client)
        if use_gui and timestep: time.sleep(timestep) # Adjust sleep