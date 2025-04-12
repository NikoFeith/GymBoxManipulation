# scripts/run_and_visualize_obs.py

import gymnasium as gym
import numpy as np
import time
import traceback
import cv2 # Import OpenCV

# Import your environment package
try:
    import physics_block_rearrangement_env
except ImportError:
    print("*"*50)
    print("ERROR: Could not import 'physics_block_rearrangement_env'.")
    print("Make sure the package is installed (e.g., 'pip install -e .')")
    print("or that the project directory is in your PYTHONPATH.")
    print("*"*50)
    exit(1)

# --- Test Configuration ---
ENV_ID = "PhysicsBlockRearrangement-v0"
TASK_CONFIG_FILE = None # Use default task

ACTION_SEQUENCE = [0, 3, 1, 4, 2, 5] # Example for place_3_line
# ACTION_SEQUENCE = None # Uncomment to use random actions

NUM_EPISODES = 2
MAX_STEPS_PER_EPISODE = len(ACTION_SEQUENCE) if ACTION_SEQUENCE else 20

RENDER_DELAY_SEC = 0.5 # Pause between steps

# --- ADD Display Size ---
DISPLAY_WIDTH = 336 # Choose a desired size (e.g., 6x the 56x56 obs)
DISPLAY_HEIGHT = 336
# -----------------------


# --- Main Test Function ---
def run_visual_test():
    print(f"--- Starting Visual Test with Observations for '{ENV_ID}' ---")
    # ... (print config info) ...

    env = None
    window_name = "Observation (Resized)" # Changed window name slightly
    # Still use WINDOW_NORMAL to allow manual resizing
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Optionally resize the window initially (may depend on OS/window manager)
    # cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    try:
        # Create Environment with GUI
        env_kwargs = {'use_gui': True}
        if TASK_CONFIG_FILE:
            env_kwargs['task_config_file'] = TASK_CONFIG_FILE

        print("Creating environment (with GUI)...")
        env = gym.make(ENV_ID, **env_kwargs)
        print("Environment created.")

        for episode in range(NUM_EPISODES):
            print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")
            obs, info = env.reset() # Get initial state

            # --- Display Initial Observation with OpenCV (Resized) ---
            print("Displaying initial observation...")
            if isinstance(obs, np.ndarray) and obs.size > 0:
                obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                # --- Resize the image ---
                obs_resized = cv2.resize(
                    obs_bgr,
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                    interpolation=cv2.INTER_NEAREST # Good for pixelated look
                    # interpolation=cv2.INTER_LINEAR # Alternative for smoother look
                )
                # ------------------------
                cv2.imshow(window_name, obs_resized)
            else:
                print("Warning: Initial observation invalid, showing blank.")
                blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                cv2.imshow(window_name, blank)
            cv2.waitKey(500) # Wait 500ms for window to show

            terminated = False
            truncated = False
            action_idx = 0

            for step_count in range(MAX_STEPS_PER_EPISODE):
                if terminated or truncated: break

                # --- Choose Action ---
                if ACTION_SEQUENCE:
                    if action_idx >= len(ACTION_SEQUENCE): break
                    action = ACTION_SEQUENCE[action_idx]; action_idx += 1
                else:
                    action = env.action_space.sample()

                print(f"\nStep {step_count + 1}: Executing Action: {action}")

                # --- Step Environment ---
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"  -> Reward: {reward:.3f}, Term: {terminated}, Trunc: {truncated}, Info: {info}")

                # --- Display Observation with OpenCV (Resized) ---
                if isinstance(obs, np.ndarray) and obs.size > 0:
                    obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                    # --- Resize the image ---
                    obs_resized = cv2.resize(
                        obs_bgr,
                        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                        interpolation=cv2.INTER_NEAREST
                    )
                    # ------------------------
                    cv2.imshow(window_name, obs_resized)
                else:
                     print("Warning: Observation invalid, showing blank.")
                     blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                     cv2.imshow(window_name, blank)

                key = cv2.waitKey(1)
                if key == 27: raise KeyboardInterrupt

                # --- Pause to see PyBullet GUI ---
                print(f"  Pausing for {RENDER_DELAY_SEC}s to observe GUI...")
                time.sleep(RENDER_DELAY_SEC)

            # ... (End of step loop, print episode end reason) ...
            time.sleep(1.0)

        # ... (End of episode loop) ...

    except KeyboardInterrupt:
         print("Test interrupted by user.")
    except Exception as e:
        print("\n!!!!!! ERROR during Visual Test !!!!!!")
        print(traceback.format_exc())

    finally:
        if env is not None:
            print("\nClosing environment.")
            env.close()
            print("Environment closed.")
        cv2.destroyAllWindows() # Close OpenCV window
        print("OpenCV windows closed.")

# --- Run the Test ---
if __name__ == "__main__":
    run_visual_test()