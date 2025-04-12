import pybullet as p
import pybullet_data
import time
import os # To construct path correctly

# Get the path to the URDF relative to this script
# Adjust this path based on your exact file structure
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir) # Assumes scripts/ is one level down from root
urdf_path = os.path.join(project_root, "physics_block_rearrangement_env", "assets", "urdf", "robots", "ur3e_robotiq", "ur3e_robotiq_140.urdf") # Check filename

# Connect to PyBullet GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # For plane.urdf
# In visualize_env.py
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,    # How far the camera is
    cameraYaw=50,        # Rotation around Z axis (degrees)
    cameraPitch=-35,     # Angle up/down (degrees)
    cameraTargetPosition=[0, 0.3, 0.5] # Point camera looks at [x,y,z] - adjust for robot/table center
)

# Basic setup
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")
tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0.5, 0]) # Example table

# --- LOAD YOUR ROBOT ---
robot_start_pos = [0, 0, 0.63] # Adjust Z based on table height
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
print(f"Loading URDF from: {urdf_path}")
try:
    robot_id = p.loadURDF(urdf_path, robot_start_pos, robot_start_orientation, useFixedBase=True)
    print(f"Robot loaded successfully with ID: {robot_id}")

except Exception as e:
    print(f"***** ERROR loading URDF: {e} *****")
    print("Check the URDF file path and mesh paths inside the URDF!")

# Keep the simulation running
p.setRealTimeSimulation(0)
while p.isConnected():
    # p.stepSimulation() # Not strictly needed if nothing is moving
    time.sleep(0.1)