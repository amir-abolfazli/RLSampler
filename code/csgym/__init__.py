from gym.envs.registration import register


register(
    id='csgym-v1',
    entry_point='csgym.cs_env:ConfSamplingEnv'
)