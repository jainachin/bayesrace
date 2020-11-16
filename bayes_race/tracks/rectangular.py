"""	Rectangular track.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import os
import sys
import numpy as np
from bayes_race.tracks import Track
from bayes_race.utils import Spline2D
import matplotlib.pyplot as plt


class Rectangular(Track):
	"""	Rectangular with any length, breadth and width"""

	def __init__(self, length, breadth, width):
		self.length = length
		self.breadth = breadth
		self.width = width
		self.track_width = width

		n_samples = 500
		self.x_center, self.y_center = self._trajectory(
			n_waypoints=n_samples, 
			n_samples=n_samples
			)
		self.x_outer, self.y_outer = self._trajectory(
			n_waypoints=n_samples, 
			n_samples=n_samples, 
			length=self.length+self.width, 
			breadth=self.breadth+self.width
			)
		self.x_inner, self.y_inner = self._trajectory(
			n_waypoints=n_samples, 
			n_samples=n_samples, 
			length=self.length-self.width, 
			breadth=self.breadth-self.width
			)

		# do not call super init, parametric is faster
		self._parametric()
		self.load_raceline()

		self.psi_init = 0.
		self.x_init = self.x_raceline[0]
		self.y_init = self.y_raceline[0]
		self.vx_init = 0.	

	def _parametric(self):
		"""	calculate track length, center line and theta for center line
			alternate approach is to call __init__ in superclass Track
			but this is much faster since we mostly have straight lines
		"""
		l, b = self.length, self.breadth
		self.track_length = 2*l + 2*b
		self.center_line = np.array([
			[0, l/2, l/2, -l/2, -l/2], 
			[-b/2, -b/2, b/2, b/2, -b/2]
			])
		self.theta_track = np.array([0, l/2, l/2+b, l/2+l+b, l/2+l+2*b])

	def _trajectory(self, n_waypoints=25, n_samples=100, return_waypoints_only=True, **kwargs):
		""" center, inner and outer track lines
			n_waypoints	: no of points used to fit cubic splines
			n_samples 	: no of points finally sampled after fitting cubic splines
		"""
		if "length" not in kwargs:
			length = self.length
		else:
			length = kwargs["length"]
		if "breadth" not in kwargs:
			breadth = self.breadth
		else:
			breadth = kwargs["breadth"]

		s = np.linspace(0, 2*(length+breadth)-1e-2, n_waypoints)
		wx = np.empty([n_waypoints])
		wy = np.empty([n_waypoints])
		for ids, theta in enumerate(s):
			wx[ids], wy[ids] = self.param_to_xy(theta, **kwargs)

		if return_waypoints_only:
			return wx, wy

		sp = Spline2D(wx, wy)
		s = np.arange(0, sp.s[-1], self.track_length/n_samples)
		x, y = [], []
		for i_s in s:
			ix, iy = sp.calc_position(i_s)
			x.append(ix)
			y.append(iy)
		return x, y

	def param_to_xy(self, theta, **kwargs):
		"""	convert distance along the track to x, y coordinates
			alternate is to call self._param2xy(theta)
			this is much faster since we mostly have straight lines
		"""
		if "length" not in kwargs:
			length = self.length
		else:
			length = kwargs["length"]
		if "breadth" not in kwargs:
			breadth = self.breadth
		else:
			breadth = kwargs["breadth"]

		theta = theta%(2*(length+breadth))
		if theta<=length/2:
			x = theta
			y = -breadth/2
		elif theta>length/2 and theta<=length/2+breadth:
			x = length/2
			y = -breadth/2 + (theta - length/2)
		elif theta>length/2+breadth and theta<=3/2*length+breadth:
			x = length/2 - (theta - length/2 - breadth)
			y = breadth/2
		elif theta>3/2*length+breadth and theta<=3/2*length+2*breadth:
			x = -length/2
			y = breadth/2 - (theta - 3/2*length - breadth)
		elif theta>3/2*length+2*breadth and theta<=2*length+2*breadth:
			x = -length/2 + (theta - 3/2*length - 2*breadth)
			y = -breadth/2
		return x, y

	def xy_to_param(self, x, y):
		"""	convert x, y coordinates to distance along the track
		"""
		theta = self._xy2param(x, y)
		return theta

	def load_raceline(self):
		"""	load raceline stored in npz file with keys 'x' and 'y'
		"""
		file_name = 'rectangular_raceline.npz'
		file_path = os.path.join(os.path.dirname(__file__), 'src', file_name)
		raceline = np.load(file_path)
		n_samples = 500
		self._load_raceline(
			wx=raceline['x'],
			wy=raceline['y'],
			n_samples=n_samples
			)

	def plot(self, **kwargs):
		""" plot center, inner and outer track lines
		"""
		fig = self._plot(**kwargs)
		return fig