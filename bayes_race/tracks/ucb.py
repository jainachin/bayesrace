"""	Track at UC Berkeley.
	Source: https://github.com/urosolia/RacingLMPC/blob/master/src/fnc/Track.py
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import os
import numpy as np
from numpy import loadtxt
from bayes_race.tracks import Track


class UCBTrack(Track):
	""" base class for UCB tracks"""

	def __init__(self, track_id, track_width, reference):

		loadstr = 'src/ucb'+track_id
		path = os.path.join(os.path.dirname(__file__),loadstr)
		self.inner = loadtxt(
			path+'_inner.txt', 
			comments='#', 
			delimiter=',', 
			unpack=False
			)
		self.center = loadtxt(
			path+'_center.txt', 
			comments='#', 
			delimiter=',', 
			unpack=False
			)
		self.outer = loadtxt(
			path+'_outer.txt', 
			comments='#', 
			delimiter=',', 
			unpack=False
			)
		self.x_inner, self.y_inner = self.inner[0,:], self.inner[1,:]
		self.x_center, self.y_center = self.center[0,:], self.center[1,:]
		self.x_outer, self.y_outer = self.outer[0,:], self.outer[1,:]
		self.track_width = track_width
		super(UCBTrack, self).__init__()
		
	def param_to_xy(self, theta):
		"""	convert distance along the track to x, y coordinates
		"""
		return self._param2xy(theta)

	def xy_to_param(self, x, y):
		"""	convert x, y coordinates to distance along the track
		"""
		theta = self._xy2param(x, y)
		return theta

	def plot(self, **kwargs):
		""" plot center, inner and outer track lines
		"""
		fig = self._plot(**kwargs)
		return fig


class UCB(UCBTrack):
	"""	UCB track
	"""

	def __init__(self, reference='center'):
		track_width = 0.8
		super(UCB, self).__init__(
			track_id='', 
			track_width=track_width, 
			reference=reference
			)

