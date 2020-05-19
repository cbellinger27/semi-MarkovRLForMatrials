install: 

pip install -e .

test:

import gym
import gym_PhaseChangeGridWorldEnv
env = gym.make('PhaseChangeGridWorld2Env_NoLanding_FreeObs-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

env.close()