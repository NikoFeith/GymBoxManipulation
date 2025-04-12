# scripts/test_env_stability.py

import gymnasium as gym
import numpy as np
import time
import traceback

# Import your environment package to register it
import physics_block_rearrangement_env

# --- Test Configuration ---
ENV_ID = "PhysicsBlockRearrangement-v0"
NUM_EPISODES = 50  # How many episodes to run
MAX_STEPS_PER_EPISODE = 200 # Max steps before truncating an episode in the test
# Specify the task config you want to test stability with
# (or leave as None to use the environment's default)
# TASK_CONFIG_FILE = "place_3_line.yaml"
TASK_CONFIG_FILE = None # Use default "place_3_line.yaml" from env definition

# --- Main Test Function ---
def run_stability_test():
    print(f"--- Starting Stability Test for '{ENV_ID}' ---")
    print(f"Running {NUM_EPISODES} episodes, max {MAX_STEPS_PER_EPISODE} steps each.")
    if TASK_CONFIG_FILE:
        print(f"Using Task Config: {TASK_CONFIG_FILE}")

    env = None
    total_steps = 0
    start_time = time.time()
    crashed = False

    try:
        # Create the environment (headless is faster for stability tests)
        env_kwargs = {'use_gui': False, 'render_mode': 'rgb_array'}
        if TASK_CONFIG_FILE:
            env_kwargs['task_config_file'] = TASK_CONFIG_FILE

        print("Creating environment...")
        env = gym.make(ENV_ID, **env_kwargs)
        print("Environment created successfully.")

        # Validate spaces upon creation
        assert isinstance(env.observation_space, gym.spaces.Box), "Observation space should be Box"
        assert isinstance(env.action_space, gym.spaces.Discrete), "Action space should be Discrete"
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")


        for episode in range(NUM_EPISODES):
            print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")
            try:
                # Reset environment
                obs, info = env.reset()

                # --- Sanity Checks after Reset ---
                assert isinstance(obs, np.ndarray), f"Reset obs is not numpy array, type: {type(obs)}"
                assert obs.shape == env.observation_space.shape, \
                       f"Reset obs shape mismatch: {obs.shape} vs {env.observation_space.shape}"
                assert obs.dtype == env.observation_space.dtype, \
                       f"Reset obs dtype mismatch: {obs.dtype} vs {env.observation_space.dtype}"
                assert isinstance(info, dict), f"Reset info is not dict, type: {type(info)}"
                print("Reset successful. Initial obs/info checks passed.")

                terminated = False
                truncated = False
                step_count = 0

                while not terminated and not truncated and step_count < MAX_STEPS_PER_EPISODE:
                    # Sample random action
                    action = env.action_space.sample()

                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_steps += 1
                    step_count += 1

                    # --- Sanity Checks after Step ---
                    assert isinstance(obs, np.ndarray), f"Step obs is not numpy array, type: {type(obs)}"
                    assert obs.shape == env.observation_space.shape, \
                           f"Step obs shape mismatch: {obs.shape} vs {env.observation_space.shape}"
                    assert obs.dtype == env.observation_space.dtype, \
                           f"Step obs dtype mismatch: {obs.dtype} vs {env.observation_space.dtype}"
                    assert isinstance(reward, (float, int)), f"Step reward is not float/int, type: {type(reward)}"
                    assert isinstance(terminated, bool), f"Step terminated is not bool, type: {type(terminated)}"
                    assert isinstance(truncated, bool), f"Step truncated is not bool, type: {type(truncated)}"
                    assert isinstance(info, dict), f"Step info is not dict, type: {type(info)}"

                    # Optional: Print progress within episode sparingly
                    if step_count % 50 == 0:
                         print(f"  Step {step_count}... (Terminated={terminated}, Truncated={truncated})")


                print(f"Episode finished after {step_count} steps.")
                if terminated: print("  Reason: Terminated (Goal Reached?)")
                if truncated: print("  Reason: Truncated (Max Steps Reached)")


            except Exception as e:
                print(f"\n!!!!!! ERROR during Episode {episode + 1} !!!!!!")
                print(traceback.format_exc())
                crashed = True
                break # Stop test on first crash

        # End Episode Loop

    except Exception as e:
        print("\n!!!!!! ERROR during Environment Creation or Setup !!!!!!")
        print(traceback.format_exc())
        crashed = True

    finally:
        if env is not None:
            print("\nClosing environment...")
            env.close()
            print("Environment closed.")

    # --- Final Report ---
    end_time = time.time()
    duration = end_time - start_time
    print("\n--- Stability Test Summary ---")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total Steps Taken: {total_steps}")
    if crashed:
        print("Result: FAILED (An error occurred)")
    else:
        print(f"Result: PASSED ({NUM_EPISODES} episodes completed without crashing)")
    print("------------------------------")

    return not crashed # Return True if passed, False if failed

# --- Run the Test ---
if __name__ == "__main__":
    success = run_stability_test()
    exit(0 if success else 1) # Exit with code 0 on success, 1 on failure