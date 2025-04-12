# scripts/run_and_visualize_obs.py

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time
import traceback

# Import your environment package to register it
import physics_block_rearrangement_env

# --- Test Configuration ---
ENV_ID = "PhysicsBlockRearrangement-v0"
# Specify the task config file you want to run
# TASK_CONFIG_FILE = "some_other_task.yaml"
TASK_CONFIG_FILE = None # Use default "place_3_line.yaml"

# Define a sequence of actions to test (indices based on your env's action space)
# Example for place_3_line (3 blocks, 3 targets, 0 dump):
# Actions: 0=Pick0, 1=Pick1, 2=Pick2, 3=PlaceTgt0, 4=PlaceTgt1, 5=PlaceTgt2
ACTION_SEQUENCE = [0, 3, 1, 4, 2, 5] # Pick 0, Place 0, Pick 1, Place 1, etc.
# Or set ACTION_SEQUENCE = None to use random actions

NUM_EPISODES = 2
MAX_STEPS_PER_EPISODE = len(ACTION_SEQUENCE) if ACTION_SEQUENCE else 20 # Limit steps

RENDER_DELAY_SEC = 0.5 # Pause between steps to see GUI and plot

# --- Main Test Function ---
def run_visual_test():
    print(f"--- Starting Visual Test with Observations for '{ENV_ID}' ---")
    if TASK_CONFIG_FILE:
        print(f"Using Task Config: {TASK_CONFIG_FILE}")
    if ACTION_SEQUENCE:
        print(f"Using Action Sequence: {ACTION_SEQUENCE}")
    else:
        print(f"Using Random Actions.")

    env = None
    plt.ion()  # Turn on interactive mode for matplotlib
    fig, ax = plt.subplots()
    img_display = None # Placeholder for the imshow object

    try:
        # --- Create Environment with GUI Enabled ---
        env_kwargs = {'use_gui': True} # <<< Key change: Enable GUI
                         # 'render_mode' is less critical now, can be 'human' or 'rgb_array'
        if TASK_CONFIG_FILE:
            env_kwargs['task_config_file'] = TASK_CONFIG_FILE

        print("Creating environment (with GUI)...")
        env = gym.make(ENV_ID, **env_kwargs)
        print("Environment created.")

        for episode in range(NUM_EPISODES):
            print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")
            obs, info = env.reset()

            # --- Initialize Observation Plot ---
            print("Displaying initial observation...")
            if img_display is None:
                img_display = ax.imshow(obs)
                plt.title(f"Observation (Episode {episode + 1}, Step 0)")
            else:
                img_display.set_data(obs)
                plt.title(f"Observation (Episode {episode + 1}, Step 0)")
            fig.canvas.draw()
            plt.pause(0.5) # Pause to see initial state

            terminated = False
            truncated = False
            action_idx = 0

            for step_count in range(MAX_STEPS_PER_EPISODE):
                if terminated or truncated: break

                # --- Choose Action ---
                if ACTION_SEQUENCE:
                    if action_idx >= len(ACTION_SEQUENCE):
                        print("Action sequence finished.")
                        break
                    action = ACTION_SEQUENCE[action_idx]
                    action_idx += 1
                else:
                    action = env.action_space.sample() # Random action

                print(f"\nStep {step_count + 1}: Executing Action: {action}")

                # --- Step Environment ---
                obs, reward, terminated, truncated, info = env.step(action)

                print(f"  -> Reward: {reward:.3f}, Term: {terminated}, Trunc: {truncated}, Info: {info}")

                print(f"  --- Obs Check (Step {step_count + 1}) ---")
                print(f"      Obs Type: {type(obs)}")
                print(f"      Is NumPy Array: {isinstance(obs, np.ndarray)}")
                if isinstance(obs, np.ndarray):
                    print(f"      Obs Shape: {obs.shape}")
                    print(f"      Obs Dtype: {obs.dtype}")
                    if obs.size > 0:
                        print(f"      Obs Stats: min={np.min(obs)}, max={np.max(obs)}, mean={np.mean(obs):.2f}")
                    else:
                        print(f"      Obs data is empty (size 0).")
                else:
                    print(f"      Obs data is not a NumPy array.")
                print(f"  -------------------------")

                # --- Update Observation Plot ---
                # The 'obs' here should be the robot-free image from _get_obs
                img_display.set_data(obs)
                plt.title(f"Observation (Episode {episode + 1}, Step {step_count + 1})")
                fig.canvas.draw_idle() # Efficiently update plot
                plt.pause(0.01) # Allow plot to refresh

                # --- Pause to see GUI ---
                # Add delay *after* step and plot update so user can see the result
                print(f"  Pausing for {RENDER_DELAY_SEC}s to observe GUI...")
                time.sleep(RENDER_DELAY_SEC)


            print(f"Episode {episode + 1} finished.")
            if terminated: print("  Reason: Terminated (Goal Reached?)")
            if truncated: print("  Reason: Truncated")
            time.sleep(1.0) # Pause between episodes


    except Exception as e:
        print("\n!!!!!! ERROR during Visual Test !!!!!!")
        print(traceback.format_exc())

    finally:
        plt.ioff() # Turn off interactive mode
        plt.close(fig) # Close the plot window
        if env is not None:
            print("\nClosing environment.")
            env.close()
            print("Environment closed.")

# --- Run the Test ---
if __name__ == "__main__":
    run_visual_test()