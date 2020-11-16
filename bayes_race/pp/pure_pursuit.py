"""	Pure pursuit controller.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np


def purePursuit(x0, LD, KP, track, params):
	"""	x0 			: current state
		LD 			: lookahead distance
		KP 			: proportional gain for speed control
		params 		: see bayes_race.params, example ORCA
		track 		: see bayes_race.tracks, example ETHZ
	"""
	# search for nearest point at lookahead distance
	xyproj, ind = track.project(x=x0[0], y=x0[1], raceline=track.raceline)
	dist = np.linalg.norm(x0[:2]-track.raceline[:,ind])
	while dist<LD:
		if ind==track.raceline.shape[1]-1:
			ind = -1
		ind += 1
		dist = np.linalg.norm(x0[:2]-track.raceline[:,ind])

	# control action
	goal = track.raceline[:,ind]
	vref = track.v_raceline[ind-1]
	acc = KP * (vref - x0[3])
	alpha = np.arctan2(goal[1] - x0[1], goal[0] - x0[0]) - x0[2]
	steer = np.arctan2(2.0 * (params['lf']+params['lr']) * np.sin(alpha) / LD, 1.0)
	return [acc, steer]