"""	Constraints for MPC.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np


def Boundary(x0, track, eps=0.01):
	"""	Compute linear boundary constraints given current position.
		Ain x >= bin and Aout x <= bout are both returned as A x <= b

		x			: current position [2x1]
		track 		: see bayes_race.tracks, example Rectangular etc
		eps 		: offset long center line required to compute direction

	"""
	theta = track.xy_to_param(x0[0], x0[1])
	x_, y_ = track.param_to_xy(theta+eps)
	_x, _y = track.param_to_xy(theta-eps)
	x, y = track.param_to_xy(theta)
	
	_x_ = np.array([x,y])
	# if np.linalg.norm(_x_-x0,2)>1e-3:
	# 	print('WARNING: current position too far from reference...')

	norm = np.sqrt((y_-_y)**2 + (x_-_x)**2)

	width = track.track_width/2
	xin = x - width*(y_-_y)/norm
	yin = y + width*(x_-_x)/norm
	Ain = np.array([(y_-_y), -(x_-_x)])
	bin = (y_-_y)*xin - (x_-_x)*yin

	width = -track.track_width/2
	xout = x - width*(y_-_y)/norm
	yout = y + width*(x_-_x)/norm
	Aout = np.array([(y_-_y), -(x_-_x)])
	bout = (y_-_y)*xout - (x_-_x)*yout

	A = np.concatenate([[-Ain],[Aout]])
	b = np.concatenate([[-bin],[bout]]).reshape(-1,1)
	return A, b