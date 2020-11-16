"""	Generate reference for MPC using a trajectory generator.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from bayes_race.utils import Spline2D


def ConstantSpeed(x0, v0, track, N, Ts, projidx, scale=0.9):
	"""	generate a reference trajectory of size 2x(N+1)
		first column is x0

		x0 		: current position (2x1)
		v0 		: current velocity (scaler)
		track	: see bayes_race.tracks, example Rectangular
		N 		: no of reference points, same as horizon in MPC
		Ts 		: sampling time in MPC
		projidx : hack required when raceline is longer than 1 lap

	"""
	# project x0 onto raceline
	raceline = track.raceline
	xy, idx = track.project_fast(x=x0[0], y=x0[1], raceline=raceline[:,projidx:projidx+10])
	projidx = idx+projidx

	# start ahead of the current position
	start = track.raceline[:,:projidx+2]

	xref = np.zeros([2,N+1])
	xref[:2,0] = x0

	# use splines to sample points based on max acceleration
	dist0 = np.sum(np.linalg.norm(np.diff(start), 2, axis=0))
	dist = dist0
	v = v0
	for idh in range(1,N+1):
		dist += scale*v*Ts
		dist = dist % track.spline.s[-1]
		xref[:2,idh] = track.spline.calc_position(dist)
		v = track.spline_v.calc(dist)
	return xref, projidx