import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import numpy as np





class Plotter():
	def __init__(self, stickyBarriers,yMax=30,yMin=0,xMax=30,xMin=0,xDestination =14,yDestination=29,xStart=17,yStart=1):
		self.yMax = yMax
		self.yMin = yMin
		self.xMax = xMax
		self.xMin = xMin
		self.xStart = xStart
		self.yStart = yStart
		self.xState = xStart
		self.yState = yStart
		self.xDestination = xDestination
		self.yDestination = yDestination
		self.stickyBarriers = stickyBarriers
		self.ax, self.fig = self.initializePlot()

	def initializePlot(self):
		yAxisTicks = np.arange(self.yMin-1,self.yMax+2, step=1)
		xAxisTicks = np.arange(self.xMin-1,self.xMax+2, step=1)
		yAxisNames = [str(x) for x in yAxisTicks.tolist()]
		yAxisNames[0] = None
		yAxisNames[len(yAxisNames)-1] = None
		xAxisNames = [str(x) for x in xAxisTicks.tolist()]
		xAxisNames[0] = None
		xAxisNames[len(xAxisNames)-1] = None
		# scaler = MinMaxScaler()
		# print(q_values)
		fig, ax = plt.subplots(figsize=(8,8))
		plt.xticks(np.round(xAxisTicks),xAxisNames)
		plt.yticks(np.round(yAxisTicks),yAxisNames)
		plt.xlabel("Temperature")
		plt.ylabel("Pressure")
		plt.tight_layout()
		plt.grid(True)
		patches = []
		colors = []
	
		ax.add_patch(plt.Circle((self.xDestination, self.yDestination), 0.5, color='red', alpha=0.95))
		ax.add_patch(plt.Circle((self.xStart, self.yStart), 0.5, color='blue', alpha=0.95))
		for b in self.stickyBarriers:
			ax.add_patch(plt.Rectangle((b[0]-0.5,b[1]-0.5),1, 1, color='grey', alpha=0.5))
		return ax, fig

	def updatePlot(self, xNewState, yNewState):
		if self.xState == self.xStart and self.yState == self.yStart:
			self.ax.add_patch(plt.Circle((self.xStart, self.yStart), 0.5, color='blue', alpha=0.95))
		elif [self.xState, self.yState] in self.stickyBarriers:
			self.ax.add_patch(plt.Circle((self.xState, self.yState), 0.5, color='white', alpha=0.95))
			self.ax.add_patch(plt.Rectangle((self.xState-0.5,self.yState-0.5),1, 1, color='grey', alpha=0.5))
		else:
			self.ax.add_patch(plt.Circle((self.xState, self.yState), 0.5, color='white', alpha=0.95))
		self.xState = xNewState
		self.yState = yNewState
		self.ax.add_patch(plt.Circle((self.xState, self.yState), 0.5, color='#00ffff', alpha=0.5))
		if self.xState == self.xDestination and self.yState == self.yDestination:
			self.ax.add_patch(plt.Rectangle((self.xState-0.5, self.yState-0.5), 1, 1, edgecolor='green', linewidth='3', alpha=0.5))
			plt.text(self.xState, self.yState+0.75, 'Successfully\n   Landed', fontsize=12, color='red')
			plt.pause(1)
		self.render()
			
	def render(self):
		plt.show(block=False)
		plt.pause(0.1)

	def close(self):
		plt.close('all')
