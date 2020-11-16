"""	Randomly generated tracks.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayes_race.tracks import Track
from bayes_race.tracks.compute_io import ComputeIO


class RandomTrack(Track):
	""" base class for Random tracks"""

	def __init__(self, track_id, track_width, reference):

		loadstr = 'src/'+track_id
		path = os.path.join(os.path.dirname(__file__),loadstr)
		self.center = pd.read_csv(path+'.csv', header=None).to_numpy().T
		self.x_center, self.y_center = self.center[0,:], self.center[1,:]
		self.track_width = track_width
		super(RandomTrack, self).__init__()
		self.load_raceline(reference, track_id)
		self.psi_init = np.arctan2(
			self.y_raceline[1] - self.y_raceline[0],
			self.x_raceline[1] - self.x_raceline[0]
			)
		self.x_init = self.x_raceline[0]
		self.y_init = self.y_raceline[0]
		self.vx_init = 0.
		self.x_inner, self.y_inner, self.x_outer, self.y_outer = ComputeIO(self)
		
	def param_to_xy(self, theta):
		"""	convert distance along the track to x, y coordinates
		"""
		return self._param2xy(theta)

	def xy_to_param(self, x, y):
		"""	convert x, y coordinates to distance along the track
		"""
		theta = self._xy2param(x, y)
		return theta

	def load_raceline(self, reference, track_id):
		"""	load raceline stored in npz file with keys 'x', 'y', 'speed', 'inputs'
		"""
		if reference is 'center':
			n_samples = 2*self.x_center.shape[0]-1
			self._load_raceline(
				wx=self.x_center,
				wy=self.y_center,
				n_samples=n_samples
				)
		elif reference is 'optimal':
			file_name = '{}_raceline.npz'.format(track_id)
			file_path = os.path.join(os.path.dirname(__file__), 'src', file_name)
			raceline = np.load(file_path)
			n_samples = raceline['x'].size
			self._load_raceline(
				wx=raceline['x'],
				wy=raceline['y'],
				n_samples=n_samples,
				v=raceline['speed'],
				t=raceline['time'],
				)
		else:
			raise NotImplementedError

	def plot(self, **kwargs):
		""" plot center, inner and outer track lines
		"""
		fig = self._plot(**kwargs)
		return fig	


class MAP2(RandomTrack):
	"""	Random track
	"""

	def __init__(self, reference='center'):
		track_width = 1
		super(MAP2, self).__init__(
			track_id='map2', 
			track_width=track_width, 
			reference=reference,
			)


class MAP3(RandomTrack):
	"""	Random track
	"""

	def __init__(self, reference='center'):
		track_width = 1
		super(MAP3, self).__init__(
			track_id='map3', 
			track_width=track_width, 
			reference=reference,
			)


class MAP8(RandomTrack):
	"""	Random track
	"""

	def __init__(self, reference='center'):
		track_width = 1
		super(MAP8, self).__init__(
			track_id='map8', 
			track_width=track_width, 
			reference=reference,
			)


class MAP16(RandomTrack):
	"""	Random track
	"""

	def __init__(self, reference='center'):
		track_width = 1
		super(MAP16, self).__init__(
			track_id='map16', 
			track_width=track_width, 
			reference=reference,
			)	