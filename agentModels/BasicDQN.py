
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
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense
from keras.models import Sequential
from keras.optimizers import Adam 
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
from collections import deque
import agentModels.ExplorationExloitationScheduler as epsilonSampler
from agentModels.Experience_buffer import experience_buffer
def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class BasicDQN:
	def __init__(self, env, numStatesX, numStatesY, numActions, stateHistSize=1, learning_rate=0.1, epsilon_min=0.01,discount=0.95, epsilon=1.0, alpha=.01, alpha_decay=.01, iterations=10000, update_target_freq=10):
		self.learning_rate = learning_rate
		self.epsilon_min = epsilon_min
		self.discount = discount # How much we appreciate future reward over current
		self.alpha = alpha
		self.alpha_decay = alpha_decay
		self.epsilon = 1.0 # Initial exploration rate
		self.epsilon_delta = 1/10000.0 # Shift from exploration to explotation
		self.numStatesX = numStatesX
		self.numStatesY = numStatesY
		self.action_size = numActions
		self.env = env
		self.memory = deque(maxlen=50000)
		self.stateHistSize = stateHistSize
		self.model = self._build_model()
		self.targetModel = self._build_model()
		self.targetModel.set_weights(self.model.get_weights())
		self.update_target_freq = update_target_freq
		self.updateCount = 0
		self.sampleEpsilon = epsilonSampler.ExplorationExploitationScheduler(self.model, numActions)

	def _build_model(self):
		model = Sequential()
		model.add(Dense(100, input_dim=(self.numStatesX + self.numStatesY)*self.stateHistSize, activation='relu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(self.action_size, activation='sigmoid'))
		model.compile(loss='mse',optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
		return model
	def update_target_model(self):
		"""
		After some time interval update the target model to be same with model
		"""
		self.targetModel.set_weights(self.model.get_weights())
	# Ask model to estimate Q value for specific state (inference)
	def get_Q(self, state, targetNet=False):
		# Model input: Single state represented by array of 5 items (state one-hot)
		# Model output: Array of Q values for single state
		if targetNet == False:
			return self.model.predict(self.to_one_hot(state))[0]
		else:
			return self.targetModel.predict(self.to_one_hot(state))[0]

	def get_next_action(self,state):
		# return self.sampleEpsilon.get_action(step_number, state)
		if random.random() > self.epsilon: # Explore (gamble) or exploit (greedy)
			return self.greedy_action(state)
		else:
			return self.random_action()

	# Which action (FORWARD or BACKWARD) has bigger Q-value, estimated by our model (inference).
	def greedy_action(self, state):
	# argmax picks the higher Q-value and returns the index (FORWARD=0, BACKWARD=1)
		return np.argmax(self.get_Q(state))

	def random_action(self):
		return self.env.action_space.sample()

	def to_one_hot(self, state):
		one_hot = np.ndarray(shape=(1,0))
		for i in range(self.stateHistSize):
			xOne_hot = np.zeros((1, self.numStatesX))
			yOne_hot = np.zeros((1, self.numStatesY))
			xOne_hot[0, state[i*2]] = 1
			yOne_hot[0, state[i*2+1]] = 1
			one_hot = np.concatenate((one_hot, xOne_hot), axis=1)
			one_hot = np.concatenate((one_hot, yOne_hot), axis=1)
		return one_hot

	def replay(self, batch_size):
		x_batch = np.ndarray(shape=(0,(self.numStatesX + self.numStatesY)*self.stateHistSize))
		y_batch = []
		minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
		for state, action, reward, next_state, done in minibatch:
			y_target = self.get_Q(state, targetNet=True)
			y_target[action] = reward if done else reward + self.discount * np.amax(self.get_Q(next_state, targetNet=True))
			x_batch = np.concatenate((x_batch, self.to_one_hot(state)))
			y_batch.append(y_target)       
		self.model.fit(np.array(x_batch), np.array(y_batch), epochs=1, batch_size=len(x_batch), verbose=0)
		# Finally shift our exploration_rate toward zero (less gambling)
		if self.epsilon > 0.01:
				self.epsilon -= self.epsilon_delta

	def remember(self, state, action, reward, next_state, done):      
		self.memory.append((state, action, reward, next_state, done))

	def train(self, old_state, action, reward, new_state):
		# Ask the model for the Q values of the old state (inference)
		old_state_Q_values = self.get_Q(old_state)

		# Ask the model for the Q values of the new state (inference)
		new_state_Q_values = self.get_Q(new_state, targetNet=True)

		# Real Q value for the action we took. This is what we will train towards.
		old_state_Q_values[action] = reward + self.discount * np.amax(new_state_Q_values)

		# Setup training data
		training_input = self.to_one_hot(old_state)
		target_output = [old_state_Q_values]
		# training_data = {self.model_input: training_input, self.target_output: target_output}

		# Train
		self.model.fit(training_input, np.array(target_output).reshape(1,self.action_size))

	def update(self, old_state, new_state, action, reward, done):
		# Train our model with new data
		self.remember(old_state, action, reward, new_state, done)
		if done:
			# self.remember(old_state, action, reward, new_state, done)
			if len(self.memory) > 127:
				self.replay(128)
				self.updateCount += 1
		if self.updateCount == self.update_target_freq:
			self.update_target_model()
			self.updateCount = 0

