
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
import agentModels.BasicDQNtf1 as bdqn

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
RND_GOAL_PROB = 0.95

hist = deque(maxlen=histSz)
for i in range(histSz):
	hist.append([0,0])



epSteps = 0
episode = 0
env = gym.make('PhaseChangeGridWorld2Env_NoLanding_FreeObs-v0')
env.renderMode = None
env.yStart = yStart
env.xStart = xStart
env.xDestination = xDestination
env.yDestination = yDestination
# env.stickyBarriers = [] #No sticky barries. Therefore the problem is Markovian
stateSpaceSize = (((env.yMax-env.yMin)+1) + ((env.yMax-env.yMin)+1))*(histSz+1)
new_state = env.reset(xStart=xStart, yStart=yStart, xDestination=xDestination, yDestination=yDestination)
goal = [env.xDestination, env.yDestination]
hist.append(new_state)
new_state = np.array(hist).flatten()
agent = bdqn.BasicDQN(env=env, numStatesX=(env.xMax-env.xMin)+1, numStatesY=(env.yMax-env.yMin)+1, stateSpaceSize=stateSpaceSize, numActions=4,stateHistSize=histSz+1) # 4 actions plus the 2-d goal state
agent.epsilon_delta = 0.00001
iterStepsPerEp = np.ndarray(shape=(0,int(numEpisodes/50)))
actionCount = np.zeros(shape=(4,int(numEpisodes/50)))
steps_per_episode = []
while episode < numEpisodes:
	epSteps += 1
	old_state = new_state # Store current state
	action = agent.get_next_action(np.concatenate((old_state,np.array(goal)))) # Query agent for the next action
	tmpState, reward, done, info = env.step(action) # Take action, get new state and reward
	hist.append(tmpState)
	new_state = np.array(hist).flatten()
	if done == False and epSteps < 10000:
		if np.random.uniform() > RND_GOAL_PROB: # and hist[len(hist)-1] != tmpState: # change the goal to where you are and train
			# print("random goal " + str(tmpState))
			agent.remember(np.concatenate((old_state,np.array(tmpState))), action, 1, np.concatenate((new_state,np.array(tmpState))), done) 
		else:
			agent.update(np.concatenate((old_state,np.array(goal))), np.concatenate((new_state,np.array(goal))), action, reward, done)
	else:
		agent.update(np.concatenate((old_state,np.array(goal))), np.concatenate((new_state,np.array(goal))), action, reward, done)
		print(episode)
		epSteps = 0
		episode += 1
		if episode % 50 == 0: # TEST PERFORMANCE 
			savedCurBufferActions = np.array([])
			savedCurTrajectory = np.ndarray(shape=(0,2))
			done = False
			tmpEp = agent.epsilon
			agent.epsilon = 0.2
			tstStps = 0
			new_state = env.reset(xStart=xStart, yStart=yStart, xDestination=xDestination, yDestination=yDestination)
			tstHist = deque(maxlen=histSz)
			for i in range(histSz):
				tstHist.append([0,0])
			tstHist.append(new_state)
			new_state = np.array(tstHist).flatten()
			while not done and tstStps < 10000:
				tstStps += 1
				old_state = new_state # Store current state
				action = agent.get_next_action(np.concatenate((old_state,np.array(goal)))) # Query agent for the next action
				actionCount[action, int(episode / 50)-1] += 1 
				new_state, reward, done, info = env.step(action) # Take action, get new state and reward
				tstHist.append(new_state)
				new_state = np.array(tstHist).flatten()
			agent.epsilon = tmpEp
			steps_per_episode = np.append(steps_per_episode, tstStps)
		new_state = env.reset(xStart=xStart, yStart=yStart, xDestination=xDestination, yDestination=yDestination)
		hist = deque(maxlen=histSz)
		for i in range(histSz):
			hist.append([0,0])
		hist.append(new_state)
		new_state = np.array(hist).flatten()

env.close()

print(actionCount)
print(steps_per_episode.reshape(1,int(numEpisodes/50)))

mean_steps_per_episode = steps_per_episode.reshape(1,int(numEpisodes/50))
plt.plot(np.arange(len(mean_steps_per_episode[0])), mean_steps_per_episode[0], color="blue", linestyle="--", label="DQN+HER")
plt.legend()
plt.show()

#TEST AGENTS

tstHist = deque(maxlen=histSz)
for i in range(histSz):
	tstHist.append([0,0])

new_state = env.reset(xStart=xStart, yStart=yStart, xDestination=xDestination, yDestination=yDestination)
env.render()
tstHist.append(new_state)
new_state = np.array(tstHist).flatten()
done = False
tmpEp = agent.epsilon
agent.epsilon = 0.2
tstStps = 0
while not done and tstStps < 10000:
	tstStps += 1
	old_state = new_state # Store current state
	action = agent.get_next_action(np.concatenate((old_state,np.array(goal)))) # Query agent for the next action
	new_state, reward, done, info = env.step(action) # Take action, get new state and reward
	tstHist.append(new_state)
	new_state = np.array(tstHist).flatten()
	env.render()

env.close()

