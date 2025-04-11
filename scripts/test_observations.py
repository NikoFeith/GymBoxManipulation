# scripts/test_observations.py

import gymnasium as gym
import numpy as np
import pybullet as p # Need pybullet if Env class imports it at top level
import time
import traceback

# --- Step 1: Ensure registration happens ---
print("--- Importing top-level package to trigger registration ---")
try:
    import physics_block_rearrangement_env
    print("Import successful.")
except Exception as e:
    print(f"ERROR importing top-level package: {type(e).__name__} - {e}")
    print("Cannot proceed without successful import.")
    exit()

# --- Step 2: Check Gymnasium Registry ---
print("\n--- Checking Gymnasium Registry ---")
env_id = 'PhysicsBlockRearrangement-v0' # Use the ID you registered
if env_id in gym.envs.registry:
    print(f"SUCCESS: Found '{env_id}' in gym.envs.registry.")
    try:
        spec = gym.envs.registry[env_id]
        print(f"  Spec found: {spec}")
        print(f"  Spec entry point: '{spec.entry_point}'")

        # --- Step 3: Try loading creator via Gymnasium's function ---
        print("  Attempting gym.envs.registration.load_env_creator...")
        env_creator = gym.envs.registration.load_env_creator(spec.entry_point)
        print(f"  SUCCESS: Loaded creator function: {env_creator}")

        # --- Step 4: Optional - Try direct instantiation via creator ---
        # print("  Attempting instantiation via creator...")
        # env_instance = env_creator(use_gui=False) # Minimal args
        # print("  Direct instantiation via creator successful.")
        # env_instance.close()

    except Exception as e:
        print(f"  ERROR during Gymnasium spec processing or creator loading: {type(e).__name__} - {e}")
        traceback.print_exc()
else:
    print(f"ERROR: Environment ID '{env_id}' NOT FOUND in gym.envs.registry!")
    print("Available IDs:", list(gym.envs.registry.keys()))
    print("Check if registration in physics_block_rearrangement_env/__init__.py executed.")

# --- Step 5: Compare with direct import again ---
print("\n--- Attempting direct import for comparison ---")
try:
    from physics_block_rearrangement_env.envs import block_rearrangement_env
    print("  Direct import successful.")
    print(f"  Module file location: {block_rearrangement_env.__file__}")
    class_found = hasattr(block_rearrangement_env, 'PhysicsBlockRearrangementEnv')
    print(f"  Class 'PhysicsBlockRearrangementEnv' found via hasattr: {class_found}")
    if not class_found:
        print("  Attributes found in module:", dir(block_rearrangement_env))
except ImportError as e:
    print(f"  ERROR: Failed to import module 'physics_block_rearrangement_env.envs': {e}")
except Exception as e:
    print(f"  ERROR during direct import/check: {type(e).__name__} - {e}")

# --- Step 6: Final attempt with gym.make ---
print("\n--- Finally attempting gym.make again ---")
try:
    env = gym.make(env_id, use_gui=False) # Minimal args for testing make
    print("SUCCESS: gym.make worked!")
    env.close()
except Exception as e:
    print(f"ERROR: gym.make failed: {type(e).__name__} - {e}")
    if isinstance(e, AttributeError):
         traceback.print_exc() # Show full traceback for AttributeError

print("\nTest script finished.")