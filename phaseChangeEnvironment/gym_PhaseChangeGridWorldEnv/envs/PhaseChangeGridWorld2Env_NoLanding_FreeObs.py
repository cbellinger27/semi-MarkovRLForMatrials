
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

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import gym
import gym.spaces
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import time
from matplotlib.collections import PatchCollection
from Plotter import Plotter


class PhaseChangeGridWorld2Env_NoLanding_FreeObs(gym.Env):
	metadata = {'render.modes': ['human', 'none']}
	NAMES_ACTIONS = { 'TD': 0,'TU': 1, 'PD': 2,'PU': 3}
	ACTIONS_NAMES = {0:'TD', 1:'TU', 2:'PD', 3:'PU'}

	def __init__(self, xStart=17, yStart=1, xDestination = 14, yDestination=29, renderMode='human'):
		self.xStart = xStart
		self.yStart = yStart
		self.xState = xStart
		self.yState = yStart
		self.xMin = 0
		self.xMax = 30
		self.yMin = 0
		self.yMax = 30
		self.stickyBarrierStps = 1
		self.xDestination = xDestination
		self.yDestination = yDestination
		self.observation_space = 4
		self.stickyBarriers = [[9,0],[10,1],[11,2],[12,3],[13,4],[14,5],[15,6],[16,7],[17,8],[18,9],[19,10],[20,11],[21,12],[22,13],[23,14],[24,15],[25,16],[26,17],[27,18],[28,19],[29,20],[30,21],[4,30],[5,29],[6,28],[7,27],[8,26],[9,25],[10,24],[11,23],[12,22],[13,21],[14,20],[15,19],[16,18],[17,17],[18,16],[19,15],[20,14],[21,13]]
		self.stickSteps = {'TD': 0,'TU': 0, 'PD': 0,'PU': 0}
		self.action_space  = spaces.Discrete(4)
		self.renderMode = renderMode
		self.p = Plotter(stickyBarriers=self.stickyBarriers,yMax=self.yMax,yMin=self.yMin,xMax=self.xMax,xMin=self.xMin,xDestination=self.xDestination,yDestination=self.yDestination,xStart=self.xStart,yStart=self.yStart)

	def step(self, action):
		done      = False
		info      = {}
		assert action >= 0 and action <= 3, 'Invalid action.'
		

		if [self.xState, self.yState] in self.stickyBarriers:			# AGENT IS AT A PHASE CHANGE BOUNDARY
			if action == 0 and self.xState > self.xMin:						# AGENT WANTS TO DECREASE THE TEMPERATURE AND HAS ROOM TO DO SO. TO DECREASE THE TEMPERATURE, IT MUST FIRST DECREASE THE PRESSURE self.stickyBarrierStps TIMES
				if self.stickSteps[PhaseChangeGridWorld2Env_NoLanding_FreeObs.ACTIONS_NAMES[2]] >= self.stickyBarrierStps:  # AGENT HAS DECREASED THE PRESSURE ENOUGH TO LOWER TEMPERATURE 
					self.xState -= 1											 # MOVE INTO NEXT PHASE
					self.stickSteps = {'TD': 0,'TU': 0, 'PD': 0,'PU': 0}		 # RESET ALL STICK STEPS 
				else:
					self.stickSteps[PhaseChangeGridWorld2Env_NoLanding_FreeObs.ACTIONS_NAMES[0]] += 1						 # INCREASE THE STICKY STEPS FOR ACTION 0  
			elif action == 1 and self.xState < self.xMax:					# AGENT WANTS TO INCREASES THE TEMPERATURE AND HAS ROOM TO DO SO. TO INCREASE THE TEMPERATURE, IT MUST FIRST INCREASE THE PRESSURE self.stickyBarrierStps TIMES
				if self.stickSteps[PhaseChangeGridWorld2Env_NoLanding_FreeObs.ACTIONS_NAMES[3]] >= self.stickyBarrierStps:  # AGENT HAS INCREASED THE PRESSURE ENOUGH TO INCREASE TEMPERATURE 
					self.xState += 1											 # MOVE INTO NEXT PHASE
					self.stickSteps = {'TD': 0,'TU': 0, 'PD': 0,'PU': 0}		 # RESET ALL STICK STEPS 
				else:
					self.stickSteps[PhaseChangeGridWorld2Env_NoLanding_FreeObs.ACTIONS_NAMES[1]] += 1						 # INCREASE THE STICKY STEPS FOR ACTION 1  
			elif action == 2 and self.yState > self.yMin:					# AGENT WANTS TO DECREASE THE PRESSURE AND HAS ROOM TO DO SO. TO DECREASE THE PRESSURE, IT MUST FIRST DECREASE THE TEMPERATURE self.stickyBarrierStps TIMES
				if self.stickSteps[PhaseChangeGridWorld2Env_NoLanding_FreeObs.ACTIONS_NAMES[0]] >= self.stickyBarrierStps:  # AGENT HAS DECREASED THE TEMPERATURE ENOUGH TO LOWER PRESSURE 
					self.yState -= 1											 # MOVE INTO NEXT PHASE
					self.stickSteps = {'TD': 0,'TU': 0, 'PD': 0,'PU': 0}		 # RESET ALL STICK STEPS 
				else:
					self.stickSteps[PhaseChangeGridWorld2Env_NoLanding_FreeObs.ACTIONS_NAMES[2]] += 1						 # INCREASE THE STICKY STEPS FOR ACTION 2  
			elif action == 3 and self.yState < self.yMax:					# AGENT WANTS TO INCREASES THE PRESSURE AND HAS ROOM TO DO SO. TO INCREASE THE PRESSURE, IT MUST FIRST INCREASE THE TEMPERATURE self.stickyBarrierStps TIMES
				if self.stickSteps[PhaseChangeGridWorld2Env_NoLanding_FreeObs.ACTIONS_NAMES[1]] >= self.stickyBarrierStps:  # AGENT HAS INCREASED THE TEMPERATURE ENOUGH TO INCREASE PRESSURE 
					self.yState += 1											 # MOVE INTO NEXT PHASE
					self.stickSteps = {'TD': 0,'TU': 0, 'PD': 0,'PU': 0}		 # RESET ALL STICK STEPS 
				else:
					self.stickSteps[PhaseChangeGridWorld2Env_NoLanding_FreeObs.ACTIONS_NAMES[3]] += 1						 # INCREASE THE STICKY STEPS FOR ACTION 3  
		else:
			if action == 0 and self.xState > self.xMin:
				self.xState -= 1
			elif action == 1 and self.xState < self.xMax:
				self.xState += 1
			elif action == 2 and self.yState > self.yMin:
				self.yState -= 1
			elif action == 3 and self.yState < self.yMax:
				self.yState += 1

		reward = self.reward(action)
		
		if self.xState == self.xDestination and self.yState == self.yDestination:
			done = True
			self.render(mode=self.renderMode)	
			self.reset(xStart=self.xStart, yStart=self.yStart, xDestination=self.xDestination, yDestination=self.yDestination)
			return [self.xDestination, self.yDestination], reward, done, info
		
		self.render(mode=self.renderMode)	
	
		return [self.xState, self.yState], reward, done, info

	def reset(self, xStart=17, yStart=1, xDestination = 14, yDestination=29):
		self._first_render = True
		self.xStart = xStart
		self.yStart = yStart
		self.xState = xStart
		self.yState = yStart
		self.xDestination = xDestination
		self.yDestination = yDestination
		self.action_space = spaces.Discrete(4)
		return [self.xState, self.yState]

	def inStickBoundary(state):
		return [state[0], state[1]] in self.stickyBarriers

	def render(self, mode='human'):
		if mode == 'human':				
			self.p.updatePlot(self.xState, self.yState)

	def close(self):
		self.p.close()

	def reward(self, action):
		if self.xState == self.xDestination and self.yState == self.yDestination:
			return  1
		else: # self.positionInEnv > self.totalDistance
			return -0.5

