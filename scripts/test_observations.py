# scripts/test_observations.py

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time
import traceback

# Import your environment package to register it
import physics_block_rearrangement_env

# --- Test Configuration ---
ENV_ID = "PhysicsBlockRearrangement-v0"
# Specify a task config file if you don't want the env's default ('place_3_line.yaml')
# TASK_CONFIG_FILE = "some_other_task.yaml"
TASK_CONFIG_FILE = None # Set to None to use the default task from __init__

if __name__ == "__main__":
    print("Creating environment...")
    env = None
    env_kwargs = {'use_gui': False, 'render_mode': 'rgb_array'}
    if TASK_CONFIG_FILE:
        env_kwargs['task_config_file'] = TASK_CONFIG_FILE
        print(f"Using Task Config: {TASK_CONFIG_FILE}")
    else:
        print("Using default task config from environment __init__.")

    try:
        # --- Corrected gym.make call ---
        # Remove num_blocks, num_dump_locations, robot_type arguments
        # Pass only arguments accepted by __init__: render_mode, use_gui,
        # task_config_file (optional), base_config_file (optional)
        env = gym.make(ENV_ID, **env_kwargs)
        # ---------------------------------

        print("Environment created successfully.")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")


        # --- Get Initial Observation ---
        print("Resetting environment...")
        obs, info = env.reset()
        print(f"Reset complete. Observation shape: {obs.shape}, dtype: {obs.dtype}, Info: {info}")

        # --- Plot Initial Observation ---
        plt.figure(1)
        plt.imshow(obs)
        plt.title("Initial Observation (After Reset)")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.show(block=False) # Show plot without blocking execution
        plt.pause(0.1) # Allow plot window to appear

        print("\nTaking 5 random steps...")
        last_obs = obs # Store the last valid observation
        for i in range(5):
            action = env.action_space.sample()
            print(f"Step {i+1}, Action: {action}")
            step_result = env.step(action)
            obs, reward, terminated, truncated, info = step_result
            last_obs = obs # Update last observation
            print(f"  -> Reward: {reward:.3f}, Term: {terminated}, Trunc: {truncated}, Info: {info}")
            if terminated or truncated:
                print("Episode ended early.")
                break
            time.sleep(0.1) # Small pause

        # --- Plot Observation After Steps ---
        print("\nPlotting observation after random steps...")
        plt.figure(2)
        plt.imshow(last_obs) # Use the observation from the last successful step
        plt.title(f"Observation After Random Steps")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.show() # Show plot and block until closed

    except Exception as e:
        print(f"\n!!!!!! An error occurred during testing !!!!!!")
        print(traceback.format_exc())
    finally:
        if env is not None:
            print("\nClosing environment.")
            env.close()
            print("Environment closed.")