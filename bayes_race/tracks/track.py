"""	Skeleton for all tracks.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import matplotlib.pyplot as plt
from bayes_race.utils import Projection, Spline, Spline2D

class Track:
	"""	Base class for all tracks"""

	def __init__(self):
		self._calc_center_line()
		self._calc_track_length()
		self._calc_theta_track()

	def _calc_center_line(self):
		"""	center line of dimension 2xn
		"""
		self.center_line = np.concatenate([
			self.x_center.reshape(1,-1), 
			self.y_center.reshape(1,-1)
			])

	def _calc_track_length(self):
		"""	calculate track length using (x,y) for center line
		"""
		center = self.center_line
		# connect first and last point
		center = np.concatenate([center, center[:,0].reshape(-1,1)], axis=1) 
		diff = np.diff(center)
		self.track_length = np.sum(np.linalg.norm(diff, 2, axis=0))

	def _calc_raceline_length(self, raceline):
		"""	calculate length of raceline
		"""
		# connect first and last point
		raceline = np.concatenate([raceline, raceline[:,0].reshape(-1,1)], axis=1) 
		diff = np.diff(raceline)
		return np.sum(np.linalg.norm(diff, 2, axis=0))

	def _calc_theta_track(self):
		"""	calculate theta for the center line
		"""
		diff = np.diff(self.center_line)
		theta_track = np.cumsum(np.linalg.norm(diff, 2, axis=0))
		self.theta_track = np.concatenate([np.array([0]), theta_track])

	def _load_raceline(self, wx, wy, n_samples, v=None):
		"""	load raceline and fit cubic splines
		"""
		x, y = self._fit_cubic_splines(
			wx=wx, 
			wy=wy, 
			n_samples=n_samples
			)
		theta = np.cumsum(np.linalg.norm(np.diff(np.array([x,y])), 2, axis=0))
		theta = np.concatenate([np.array([0]), theta])
		self.x_raceline = np.array(x)
		self.y_raceline = np.array(y)
		self.raceline = np.array([x, y])

		dy = np.diff(np.array(y + [y[0]]))
		dx = np.diff(np.array(x + [x[0]]))
		psi = np.arctan(dy/dx)
		for idr in range(1,psi.shape[0]):
			if psi[idr]-psi[idr-1]>1:
				psi[idr:] = psi[idr:] - np.pi
			if psi[idr]-psi[idr-1]<-1:
				psi[idr:] = psi[idr:] + np.pi
		self.psi_raceline = psi
		# plt.figure()
		# plt.plot(psi)
		# plt.show()
		# import pdb; pdb.set_trace()
		self.spline_psi = Spline(theta, psi)

		if v is not None:
			self.v_raceline = v
			self.spline_v = Spline(theta, v)
		
	def _fit_cubic_splines(self, wx, wy, n_samples):
		"""	fit cubic splines on waypoints
		"""
		sp = Spline2D(wx, wy)
		self.spline = sp
		raceline = np.concatenate([[wx],[wy]])
		raceline_length = self._calc_raceline_length(raceline)
		s = np.arange(0, sp.s[-1], raceline_length/n_samples)
		# s = np.linspace(0, sp.s[-1], n_samples)
		x, y = [], []
		for i_s in s:
			ix, iy = sp.calc_position(i_s)
			x.append(ix)
			y.append(iy)
		return x, y

	def _param2xy(self, theta):
		"""	finds (x,y) coordinate on center line for a given theta
		"""
		theta_track = self.theta_track
		idt = 0
		while idt<theta_track.shape[0]-1 and theta_track[idt]<=theta:
			idt+=1
		deltatheta = (theta-theta_track[idt-1])/(theta_track[idt]-theta_track[idt-1])
		x = self.x_center[idt-1] + deltatheta*(self.x_center[idt]-self.x_center[idt-1])
		y = self.y_center[idt-1] + deltatheta*(self.y_center[idt]-self.y_center[idt-1])
		return x, y

	def _xy2param(self, x, y):
		"""	finds theta on center line for a given (x,y) coordinate
		"""
		center_line = self.center_line
		theta_track = self.theta_track

		optxy, optidx = self.project(x, y, center_line)
		distxy = np.linalg.norm(optxy-center_line[:,optidx],2)
		dist = np.linalg.norm(center_line[:,optidx+1]-center_line[:,optidx],2)
		deltaxy = distxy/dist
		if optidx==-1:
			theta = theta_track[optidx] + deltaxy*(self.track_length-theta_track[optidx])
		else:
			theta = theta_track[optidx] + deltaxy*(theta_track[optidx+1]-theta_track[optidx])
		theta = theta % self.track_length
		return theta

	def project(self, x, y, raceline):
		"""	finds projection for (x,y) on a raceline
		"""
		point = [(x, y)]
		n_waypoints = raceline.shape[1]

		proj = np.empty([2,n_waypoints])
		dist = np.empty([n_waypoints])
		for idl in range(-1, n_waypoints-1):
			line = [raceline[:,idl], raceline[:,idl+1]]
			proj[:,idl], dist[idl] = Projection(point, line)
		optidx = np.argmin(dist)
		if optidx == n_waypoints-1:
			optidx = -1
		optxy = proj[:,optidx]
		return optxy, optidx

	def _plot(self, color='g', grid=True, figsize=(6.4, 4.8)):
		""" plot center, inner and outer track lines
		"""
		fig = plt.figure(figsize=figsize)
		plt.grid(grid)
		# plt.plot(self.x_center, self.y_center, '--'+color, lw=0.75, alpha=0.5)
		plt.plot(self.x_outer, self.y_outer, color, lw=0.75, alpha=0.5)
		plt.plot(self.x_inner, self.y_inner, color, lw=0.75, alpha=0.5)
		plt.scatter(0, 0, color='k', alpha=0.2)
		plt.axis('equal')
		return fig

	def plot_raceline(self):
		""" plot center, inner and outer track lines
		"""
		fig = self._plot()
		plt.plot(self.x_raceline, self.y_raceline, 'b', lw=1)
		plt.show()

	def param_to_xy(self, theta, **kwargs):
		"""	convert distance along the track to x, y coordinates
		"""
		raise NotImplementedError

	def xy_to_param(self, x, y):
		"""	convert x, y coordinates to distance along the track
		"""
		raise NotImplementedError		