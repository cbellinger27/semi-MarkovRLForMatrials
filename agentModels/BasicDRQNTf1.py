
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
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.optimizers import Adam 
import tensorflow as tf

import numpy as np
import gym
import matplotlib.pyplot as plt
import time
from collections import deque
import agentModels.ExplorationExloitationScheduler as epsilonSampler
from agentModels.Experience_buffer import experience_buffer


class BasicDRQN:
	def __init__(self, env, numStatesX, numStatesY, numActions, stateWithGoal=False, learning_rate=0.1, epsilon_min=0.01,discount=0.95, epsilon=1.0, alpha=.01, alpha_decay=.01, iterations=10000, trace_length=8, update_target_freq=1):
		self.learning_rate = learning_rate
		self.epsilon_min = epsilon_min
		self.discount = discount # How much we appreciate future reward over current
		self.alpha = alpha
		self.alpha_decay = alpha_decay
		self.epsilon = 1.0 # Initial exploration rate
		self.epsilon_delta = 1.0 / 1000.0 # Shift from exploration to explotation
		self.numStatesX = numStatesX
		self.numStatesY = numStatesY
		self.stateWithGoal = stateWithGoal
		self.stateSize = (self.numStatesX + self.numStatesY )
		if self.stateWithGoal:
			self.stateSize = self.stateSize * 2
		self.action_size = numActions
		self.env = env
		self.memory = experience_buffer(buffer_size = 150)
		self.trace_length = trace_length
		self.model = self._build_model()
		self.targetModel  = self._build_model()
		self.targetModel.set_weights(self.model.get_weights())
		self.update_target_freq = update_target_freq
		self.updateCount = 0
		self.sampleEpsilon = epsilonSampler.ExplorationExploitationScheduler(self.model, numActions)

	def _build_model(self):
		model = Sequential()
		model.add(GRU(128, return_sequences=False, activation='relu',input_shape=(None, self.stateSize)))
		# model.add(Dense(self.action_size, activation = "softmax"))
		model.add(Dense(self.action_size, activation = "linear"))
		model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
		# model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
		return model
	#
	def setModelWeights(self, weightsFile="drqnHer.h5"):
		self.model.load_weights(weightsFile)

	def saveModelWeights(self, fileName="drqnHer.h5"):
		self.model.save_weights(fileName)

	def update_target_model(self):
		"""
		After some time interval update the target model to be same with model
		"""
		self.targetModel.set_weights(self.model.get_weights())
	# Ask model to estimate Q value for specific state (inference)
	def get_Q(self, state, isTraining=False):
		# Model input: Single state represented by array of 5 items (state one-hot)
		# Model output: Array of Q values for single state
		oneHotState = np.ndarray(shape=(0,self.stateSize))
		if self.trace_length > 1:
			for i in range(self.trace_length):
				if self.stateWithGoal:
					tmpState = np.concatenate((self.to_one_hot(state[i,:2]), self.to_one_hot(state[i,2:])),axis=1)
					oneHotState = np.concatenate((oneHotState, tmpState))
				else: 
					tmpState = self.to_one_hot(state[i,:])
					oneHotState = np.concatenate((oneHotState, tmpState))
		else:
			if self.stateWithGoal:
				tmpState = np.concatenate((self.to_one_hot(state[0, :2]), self.to_one_hot(state[0,2:])),axis=1)
				oneHotState = np.concatenate((oneHotState, tmpState))
			else:
				tmpState = self.to_one_hot(state[0, :2])
				oneHotState = np.concatenate((oneHotState, tmpState))
		if isTraining == False:
			return self.model.predict(oneHotState.reshape(1,self.trace_length,self.stateSize))[0]
		else:
			return self.targetModel.predict(oneHotState.reshape(1,self.trace_length,self.stateSize))[0]

	def get_next_action(self,state):
		# return self.sampleEpsilon.get_action(step_number, state)
		if random.random() > self.epsilon: # Explore (gamble) or exploit (greedy)
			action = self.greedy_action(state)
			# print("greedy"+str(action))
			return action
		else:
			action = self.random_action()
			# print("random"+str(action))
			return action

	# Which action (FORWARD or BACKWARD) has bigger Q-value, estimated by our model (inference).
	def greedy_action(self, state):
	# argmax picks the higher Q-value and returns the index (FORWARD=0, BACKWARD=1)
		return np.argmax(self.get_Q(state))

	def random_action(self):
		return self.env.action_space.sample()

	def to_one_hot(self, state):
		one_hot = np.ndarray(shape=(1,0))
		xOne_hot = np.zeros((1, self.numStatesX))
		yOne_hot = np.zeros((1, self.numStatesY))
		xOne_hot[0, int(state[0])] = 1
		yOne_hot[0, int(state[1])] = 1
		one_hot = np.concatenate((one_hot, xOne_hot), axis=1)
		one_hot = np.concatenate((one_hot, yOne_hot), axis=1)
		return one_hot

	def replay(self, trainBatch):
		x_batch = np.ndarray(shape=(0,self.trace_length,self.stateSize))
		y_batch = []
		# for state, action, reward, next_state, done in trainBatch:
		for inst in range(0, trainBatch.shape[0],self.trace_length):
			endPoint = self.trace_length
			state = trainBatch[inst][0]
			next_state = trainBatch[inst][3]
			y_target = self.get_Q(state, isTraining=True)
			reward = trainBatch[inst][2]
			action = trainBatch[inst][1]
			done = trainBatch[inst][4]
			y_target[action] = reward if done else reward + self.discount * np.amax(self.get_Q(next_state, isTraining=True))
			oneHotState = np.ndarray(shape=(0,self.stateSize))
			if self.trace_length > 1:
				for i in range(self.trace_length):
					if self.stateWithGoal:
						tmpState = np.concatenate((self.to_one_hot(state[i,:2]), self.to_one_hot(state[i,2:])),axis=1)
						oneHotState = np.concatenate((oneHotState, tmpState))
					else: 
						tmpState = self.to_one_hot(state[i,:2])
						oneHotState = np.concatenate((oneHotState, tmpState))
			else:
				if self.stateWithGoal:
					tmpState = np.concatenate((self.to_one_hot(state[0, :2]), self.to_one_hot(state[0, 2:])),axis=1)
					oneHotState = np.concatenate((oneHotState, tmpState))
				else:
					tmpState = self.to_one_hot(state[0, :2])
					oneHotState = np.concatenate((oneHotState, tmpState))
			x_batch = np.concatenate((x_batch, oneHotState.reshape(1,self.trace_length,self.stateSize)))
			y_batch.append(y_target)       
		self.model.fit(np.array(x_batch), np.array(y_batch), epochs=1, batch_size=len(x_batch), verbose=0)
		# Finally shift our exploration_rate toward zero (less gambling)
		if self.epsilon > 0.01:
				self.epsilon -= self.epsilon_delta
	
	def train(self, trainBatch):
		# Ask the model for the Q values of the old state (inference)
		old_state_Q_values = self.get_Q(old_state)

		# Ask the model for the Q values of the new state (inference)
		new_state_Q_values = self.get_Q(new_state,isTraining=True)

		# Real Q value for the action we took. This is what we will train towards.
		old_state_Q_values[action] = reward + self.discount * np.amax(new_state_Q_values)

		# Setup training data
		training_input = self.to_one_hot(old_state)
		target_output = [old_state_Q_values]
		# training_data = {self.model_input: training_input, self.target_output: target_output}

		# Train
		self.model.fit(training_input, np.array(target_output).reshape(1,self.action_size))

	def update(self, episodeBuffer=None, train=False, batch_size=50):
		# Train our model with new data
		 #Add the episode to the experience buffer
		if episodeBuffer != None:
			bufferArray = np.array(episodeBuffer)
			episodeBuffer = list(zip(bufferArray))
			self.memory.add(episodeBuffer)
		if train==True and len(self.memory.buffer) > batch_size:
			trainBatch = self.memory.sample(batch_size,self.trace_length) #Get a random batch of experiences.
			self.replay(trainBatch)
			self.updateCount += 1
		if self.updateCount == self.update_target_freq:
			self.update_target_model()
			self.updateCount = 0
