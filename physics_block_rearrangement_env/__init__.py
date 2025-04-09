from gymnasium.envs.registration import register

register(
     id='PhysicsBlockRearrangement-v0',
     entry_point='physics_block_rearrangement_env.envs:PhysicsBlockRearrangementEnv',
     max_episode_steps=100,
)
