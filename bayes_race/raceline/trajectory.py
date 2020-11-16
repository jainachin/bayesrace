"""	Sample a new trajectory parametrized by deviation from center line
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import matplotlib.pyplot as plt

from bayes_race.utils import Spline2D
from bayes_race.tracks import MAP2
from bayes_race.params import ORCA, F110
from bayes_race.raceline import calcMinimumTime


class randomTrajectory:

	def __init__(self, track, n_waypoints):
		"""	track is an object with the following attributes/methods
				track_width (vector)	: width of the track in [m] (assumed constant at each way point)
				track_length (scaler)	: length of the track in [m]
				param_to_xy (function): converts arc length to x, y coordinates on the track
			n_waypoints is no of points used to fit cubic splines (equal to dim in BayesOpt)
		"""
		self.track = track
		self.n_waypoints = n_waypoints

	def sample_nodes(self, scale):
		""" sample width vector of length `n_waypoints`
		"""
		# shrink to prevent getting too close to corners
		track_width = self.track.track_width*scale
		width = -track_width/2 + track_width*np.random.rand(self.n_waypoints)
		return width

	def calculate_xy(self, width, last_index, theta=None):
		"""	compute x, y coordinates from sampled nodes (width) 
		"""
		track = self.track
		n_waypoints = width.shape[0]
		eps = 1/5/n_waypoints*track.track_length

		# starting and terminal points are fixed
		wx = np.zeros(n_waypoints+2)
		wy = np.zeros(n_waypoints+2)
		wx[0] = track.x_center[0]
		wy[0] = track.y_center[0]
		wx[-1] = track.x_center[last_index]
		wy[-1] = track.y_center[last_index]
		if theta is None:
			theta = np.linspace(0, track.track_length, n_waypoints+2)
		else:
			assert width.shape[0]==len(theta), 'dims not equal'
			theta_start = np.array([0])
			theta_end = np.array([self.track.theta_track[last_index]])
			theta = np.concatenate([theta_start, theta, theta_end])

		# compute x, y for every way point parameterized by arc length
		for idt in range(1,n_waypoints+1):
			x_, y_ = track.param_to_xy(theta[idt]+eps)
			_x, _y = track.param_to_xy(theta[idt]-eps)
			x, y = track.param_to_xy(theta[idt])
			norm = np.sqrt((y_-_y)**2 + (x_-_x)**2)
			wx[idt] = x - width[idt-1]*(y_-_y)/norm
			wy[idt] = y + width[idt-1]*(x_-_x)/norm
		return wx, wy

	def fit_cubic_splines(self, wx, wy, n_samples):
		"""	fit cubic splines on the waypoints
		"""
		sp = Spline2D(wx, wy)
		s = np.linspace(0, sp.s[-1]-1e-3, n_samples)
		x, y = [], []
		for i_s in s:
			ix, iy = sp.calc_position(i_s)
			x.append(ix)
			y.append(iy)
		return x, y


if __name__ == '__main__':
	"""	example how to use
	"""

	TRACK_NAME = 'MAP2'
	SCALE = 0.90

	# choose vehicle params and specify indices of the nodes
	params = F110()
	track = MAP2()
	NODES = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 5]
	LASTIDX = 1

	print(track.x_raceline.shape)
	# find cooresponding distance in path coordinates
	theta = track.theta_track[NODES]
	n_waypoints = len(theta)

	# sample a random trajectory
	rand_traj = randomTrajectory(
		track=track,
		n_waypoints=n_waypoints,
		)
	width_random = rand_traj.sample_nodes(scale=SCALE)

	# find corresponding x,y coordinates
	# here we choose terminal point to be the first point to prevent crashing before finishing
	wx_random, wy_random = rand_traj.calculate_xy(
		width_random,
		last_index=NODES[LASTIDX],
		theta=theta,
		)

	# resample after fitting cubic splines
	n_samples = 500
	x_random, y_random = rand_traj.fit_cubic_splines(
		wx=wx_random, 
		wy=wy_random, 
		n_samples=n_samples
		)

	# uncomment to calculate minimum time to traverse
	# t_random = calcMinimumTime(x_random, y_random, **params)
	# print('time to traverse random trajectory: {}'.format(t_random))

	# plot
	fig = track.plot(color='k', grid=False)
	x_center, y_center = track.x_center, track.y_center
	plt.plot(x_center, y_center, '--k', alpha=0.5, lw=0.5)
	plt.plot(x_random, y_random, label='splines', lw=1.5)
	plt.plot(wx_random, wy_random, 'x', label='way points')
	plt.plot(wx_random[0], wy_random[0], 'o', label='start')
	plt.plot(wx_random[-1], wy_random[-1], 'o', label='finish')
	plt.axis('equal')
	plt.xlabel('x [m]')
	plt.ylabel('y [m]')
	plt.legend(loc=0)
	plt.show()