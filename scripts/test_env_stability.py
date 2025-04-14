# scripts/test_env_stability.py

import gymnasium as gym
import numpy as np
import time
import traceback
import cv2 # Import OpenCV

# Import your environment package to register it
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
NUM_EPISODES = 50
MAX_STEPS_PER_EPISODE = 200
TASK_CONFIG_FILE = "place_3_line.yaml" # Use default task

# --- Visualization Options ---
VISUALIZE_OBS = True  # <<< Set to True to see observations, False for max speed stability test
DISPLAY_WIDTH = 336   # Display size if VISUALIZE_OBS is True
DISPLAY_HEIGHT = 336
# ---------------------------



# --- Main Test Function ---
def run_stability_test():
    print(f"--- Starting Stability Test for '{ENV_ID}' ---")
    print(f"Running {NUM_EPISODES} episodes, max {MAX_STEPS_PER_EPISODE} steps each.")
    print(f"Visualize Observations: {VISUALIZE_OBS}")
    if TASK_CONFIG_FILE:
        print(f"Using Task Config: {TASK_CONFIG_FILE}")

    env = None
    total_steps = 0
    start_time = time.time()
    crashed = False
    window_name = "Observation (Stability Test)"

    if VISUALIZE_OBS:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    failure_counts = {
        'already_holding': 0,
        'invalid_index': 0,
        'pose_fetch_failed': 0,
        'grasp_calc_failed': 0,
        'both_orientations_failed': 0,
        'gripper_open_failed': 0,
        'pregrasp_move_failed': 0,
        'grasp_move_failed': 0,
        'gripper_close_failed': 0,
        'constraint_create_failed': 0,
        'lift_failed': 0,
        'ik_calculation_failed': 0,
        "home pose recovery": 0,
    }


    try:
        # Create the environment (always headless for stability/speed)
        env_kwargs = {'use_gui': False, 'render_mode': 'rgb_array'} # Must be rgb_array to get obs
        if TASK_CONFIG_FILE:
            env_kwargs['task_config_file'] = TASK_CONFIG_FILE

        print("Creating environment (headless)...")
        env = gym.make(ENV_ID, **env_kwargs)
        print("Environment created successfully.")

        assert isinstance(env.observation_space, gym.spaces.Box), "Observation space should be Box"
        assert isinstance(env.action_space, gym.spaces.Discrete), "Action space should be Discrete"
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")


        for episode in range(NUM_EPISODES):
            print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---", end="") # Use end="" for less verbose output
            start_ep_time = time.time()
            ep_crashed = False
            try:
                obs, info = env.reset()

                # --- Initial Obs Check & Display ---
                assert isinstance(obs, np.ndarray), f"Reset obs type: {type(obs)}"
                assert obs.shape == env.observation_space.shape, f"Reset obs shape: {obs.shape}"
                assert obs.dtype == env.observation_space.dtype, f"Reset obs dtype: {obs.dtype}"

                if VISUALIZE_OBS:
                    if isinstance(obs, np.ndarray) and obs.size > 0:
                        obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                        obs_resized = cv2.resize(obs_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow(window_name, obs_resized)
                        key = cv2.waitKey(1) # Need minimal waitKey to process window events
                        if key == 27: raise KeyboardInterrupt # Allow ESC quit
                    else: # Handle case where initial obs might be invalid
                         blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                         cv2.imshow(window_name, blank)
                         key = cv2.waitKey(1)
                         if key == 27: raise KeyboardInterrupt

                terminated = False
                truncated = False
                step_count = 0

                while not terminated and not truncated and step_count < MAX_STEPS_PER_EPISODE:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if not info.get("primitive_success", True):
                        reason = getattr(env.unwrapped, "last_failure_reason", "unknown_failure")
                        failure_counts[reason] = failure_counts.get(reason, 0) + 1

                    total_steps += 1
                    step_count += 1

                    # --- Step Sanity Checks ---
                    assert isinstance(obs, np.ndarray), f"Step obs type: {type(obs)}"
                    assert obs.shape == env.observation_space.shape, f"Step obs shape: {obs.shape}"
                    assert obs.dtype == env.observation_space.dtype, f"Step obs dtype: {obs.dtype}"
                    assert isinstance(reward, (float, int)), f"Step reward type: {type(reward)}"
                    assert isinstance(terminated, bool), f"Step term type: {type(terminated)}"
                    assert isinstance(truncated, bool), f"Step trunc type: {type(truncated)}"
                    assert isinstance(info, dict), f"Step info type: {type(info)}"

                    # --- Optional Visualization ---
                    if VISUALIZE_OBS:
                        if isinstance(obs, np.ndarray) and obs.size > 0:
                             obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                             obs_resized = cv2.resize(obs_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
                             cv2.imshow(window_name, obs_resized)
                        else: # Show blank if obs invalid
                             blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                             cv2.imshow(window_name, blank)
                        key = cv2.waitKey(1) # Need minimal waitKey
                        if key == 27: raise KeyboardInterrupt

                    # Print progress less often for speed
                    if step_count % 100 == 0: print(".", end="", flush=True)

                # --- End Step Loop ---
                ep_duration = time.time() - start_ep_time
                print(f" Finished Ep {episode + 1} ({step_count} steps) in {ep_duration:.2f}s.")
                # Don't print term/trunc reason unless verbose mode desired

            except KeyboardInterrupt: # Handle user exit cleanly
                 print("\nTest interrupted by user.")
                 crashed = True # Treat interruption as failure for summary
                 break
            except Exception as e:
                print(f"\n!!!!!! ERROR during Episode {episode + 1} !!!!!!")
                print(traceback.format_exc())
                crashed = True
                ep_crashed = True # Mark episode as crashed
                break # Stop test on first crash

            if ep_crashed: break # Exit outer loop too if inner loop broke due to error

        # --- End Episode Loop ---

    except KeyboardInterrupt: # Handle user exit during setup
         print("\nTest interrupted by user during setup.")
         crashed = True
    except Exception as e:
        print("\n!!!!!! ERROR during Environment Creation or Setup !!!!!!")
        print(traceback.format_exc())
        crashed = True

    finally:
        if env is not None:
            print("\nClosing environment...")
            env.close()
            print("Environment closed.")
        if VISUALIZE_OBS:
            cv2.destroyAllWindows()
            print("OpenCV windows closed.")

    # --- Final Report ---
    end_time = time.time()
    duration = end_time - start_time
    print("\n--- Stability Test Summary ---")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total Steps Taken: {total_steps}")

    print("\n--- Failure Summary ---")
    if failure_counts:
        for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"{reason:30s}: {count}")
    else:
        print("No primitive failures occurred.")

    if crashed:
        print("Result: FAILED (An error occurred or test interrupted)")
    else:
        print(f"Result: PASSED ({NUM_EPISODES} episodes completed without crashing)")
    print("------------------------------")

    return not crashed

# --- Run the Test ---
if __name__ == "__main__":
    success = run_stability_test()
    exit(0 if success else 1)