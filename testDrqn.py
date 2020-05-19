
####################################################################################
# <one line to give the program's name and a brief idea of what it does.>
# Copyright (C) <2020>  <Colin Bellinger>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# Please direct any questions / comments to myself, Colin Bellinger, at
# colin.bellinger@gmail.com. For additional software and publications
# please see https://web.cs.dal.ca/~bellinger and researchgate
# https://www.researchgate.net/profile/Colin_Bellinger
#
# Relevant publications include: 
# 1. Colin Bellinger, Rory Coles, Mark Crowley and Isaac Tamblyn, Reinforcement Learning in a Physics-Inspired Semi-Markov Environment, Canadian Conference on Artificial Intelligence, 2020.
#
# https://arxiv.org/abs/2004.07333
#
#
####################################################################################

import random
import numpy as np
import gym
import gym_PhaseChangeGridWorldEnv
import matplotlib.pyplot as plt
import time
# from multiprocessing import Process
from collections import deque
import agentModels.BasicDRQNTf1 as bdrqn

import tensorflow as tf

numEpisodes = 10000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


### ORIGINAL
xDestination = 14
yDestination=29
xStart=17
yStart=1

histSz = 1

hist = deque(maxlen=histSz)
for i in range(histSz):
	hist.append([0,0])


trace_length = 1

epSteps = 0
episode = 0
env = gym.make('PhaseChangeGridWorld2Env_NoLanding_FreeObs-v0')
env.renderMode = None
env.yStart = yStart
env.xStart = xStart
env.xDestination = xDestination
env.yDestination = yDestination
goal = [env.xDestination, env.yDestination]
# env.stickyBarriers = [] #No sticky barries. Therefore the problem is Markovian
new_state = env.reset(xStart=xStart, yStart=yStart, xDestination=xDestination, yDestination=yDestination)
tmpState = np.ndarray(shape=(0,2)) 
for i in range(trace_length):
	tmpState = np.concatenate((tmpState, np.array([0,0]).reshape(1,2)))

agent = bdrqn.BasicDRQN(env=env, numStatesX=(env.xMax-env.xMin)+1, numStatesY=(env.xMax-env.yMin)+1, stateWithGoal=False, numActions=4,trace_length=trace_length)
episodeBuffer = []
agent.epsilon_delta = 0.00001
iterStepsPerEp = np.ndarray(shape=(0,int(numEpisodes/50)))
actionCount = np.zeros(shape=(4,int(numEpisodes/50)))
steps_per_episode = []
while episode <= numEpisodes:
	epSteps += 1
	old_state = new_state # Store current state
	tmpState = np.concatenate((tmpState[1:,:], np.array(old_state).reshape(1,2)))
	tmpOldState = tmpState
	action = agent.get_next_action(tmpState) # Query agent for the next action
	new_state, reward, done, info = env.step(action) # Take action, get new state and reward
	tmpNewState = np.concatenate((tmpState[1:,:], np.array(new_state).reshape(1,2)))
	episodeBuffer.append(np.reshape(np.array([tmpOldState,action,reward,tmpNewState,done]),[1,5]))
	if done or epSteps > 10000:
		# steps_per_episode = np.append(steps_per_episode, epSteps)
		print(episode)
		epSteps = 0
		episode +=1
		if done == True:
			agent.update(episodeBuffer=episodeBuffer, train=True, batch_size=128)
		elif len(agent.memory.buffer) > 5:
			agent.update(episodeBuffer=None, train=True, batch_size=128)
		if episode % 50 == 0: # TEST PERFORMANCE 
			agent.model.reset_states()
			done = False
			tmpEp = agent.epsilon
			agent.epsilon = 0.2
			tstStps = 0
			tmpState = np.ndarray(shape=(0,2)) 
			for i in range(trace_length):
				tmpState = np.concatenate((tmpState, np.array([0,0]).reshape(1,2)))
			new_state = env.reset(xStart=xStart, yStart=yStart, xDestination=xDestination, yDestination=yDestination)
			while not done and tstStps < 10000:
				tstStps += 1
				old_state = new_state # Store current state
				tmpState = np.concatenate((tmpState[1:,:], np.array(old_state).reshape(1,2)))
				action = agent.get_next_action(tmpState) # Query agent for the next action
				actionCount[action, int(episode / 50)-1] += 1 
				new_state, reward, done, info = env.step(action) # Take action, get new state and reward
			steps_per_episode = np.append(steps_per_episode, tstStps)
			agent.epsilon = tmpEp
		agent.model.reset_states()
		episodeBuffer = []
		new_state = env.reset(xStart=xStart, yStart=yStart, xDestination=xDestination, yDestination=yDestination)
		tmpState = np.ndarray(shape=(0,2)) 
		for i in range(trace_length):
			tmpState = np.concatenate((tmpState, np.array([0,0]).reshape(1,2)))


env.close()


print(actionCount)
print(steps_per_episode.reshape(1,int(numEpisodes/50)))

mean_steps_per_episode = steps_per_episode.reshape(1,int(numEpisodes/50))
plt.plot(np.arange(len(mean_steps_per_episode[0])), mean_steps_per_episode[0], color="blue", linestyle="--", label="DRQN")
plt.legend()
plt.show()

#TEST AGENTS

agent.model.reset_states()
done = False
tmpEp = agent.epsilon
agent.epsilon = 0.2
tstStps = 0
tmpState = np.ndarray(shape=(0,2)) 
for i in range(trace_length):
	tmpState = np.concatenate((tmpState, np.array([0,0]).reshape(1,2)))

new_state = env.reset(xStart=xStart, yStart=yStart, xDestination=xDestination, yDestination=yDestination)
agent.model.reset_states()
while not done and tstStps < 10000:
	tstStps += 1
	old_state = new_state # Store current state
	tmpState = np.concatenate((tmpState[1:,:], np.array(old_state).reshape(1,2)))
	action = agent.get_next_action(tmpState) # Query agent for the next action
	new_state, reward, done, info = env.step(action) # Take action, get new state and reward
	env.render()

env.close()


