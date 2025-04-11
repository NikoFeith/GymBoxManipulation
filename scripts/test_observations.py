# scripts/test_observation.py

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time

# IMPORTANT: Make sure your custom environment package is installed or accessible
# (e.g., by running 'pip install -e .' in your project root directory)
import physics_block_rearrangement_env # Import to register the env

if __name__ == "__main__":
    print("Creating environment...")
    # Create env in headless mode (faster, no GUI needed just for obs)
    # Use the parameters matching your setup
    env = gym.make("PhysicsBlockRearrangement-v0",
                   use_gui=False,
                   render_mode='rgb_array', # Use rgb_array if you might call render()
                   num_blocks=4,           # Example: match your test setup
                   num_dump_locations=1,
                   robot_type='panda'      # Or 'ur3e' if you switched back
                  )
    print("Environment created.")

    try:
        # --- Get Initial Observation ---
        print("Resetting environment...")
        obs, info = env.reset()
        print(f"Reset complete. Observation shape: {obs.shape}, dtype: {obs.dtype}")

        # --- Plot Initial Observation ---
        plt.figure(1)
        plt.imshow(obs)
        plt.title("Initial Observation (After Reset)")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.show(block=False) # Show plot without blocking execution

        print("\nTaking a few random steps...")
        for i in range(5):
            action = env.action_space.sample()
            print(f"Step {i+1}, Action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode ended early.")
                break
            time.sleep(0.1) # Small pause

        # --- Plot Observation After Steps ---
        print("\nPlotting observation after random steps...")
        plt.figure(2)
        plt.imshow(obs)
        plt.title(f"Observation After {step+1} Random Steps") # Use last step count
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.show() # Show plot and block until closed

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing environment.")
        env.close()