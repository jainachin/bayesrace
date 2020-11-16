"""	Plot optimal racing lines from saved results.
	See generate_raceline_ucb.py, generate_raceline_ethz.py, generate_raceline_ethzmobil.py
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np

from bayes_race.tracks import UCB, ETHZ, ETHZMobil
from bayes_race.params import F110, ORCA
from bayes_race.raceline import randomTrajectory
from bayes_race.raceline import calcMinimumTimeSpeedInputs
from bayes_race.utils import Spline2D

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

#####################################################################
# which data

SAVE_RESULTS = False

# saved results 1
savestr = '20200207165005' 
TRACK_NAME = 'ETHZ'

# saved results 2
# savestr = '20200207182156'
# TRACK_NAME = 'ETHZMobil' 

# saved results 3
# savestr = '20200206181354'
# TRACK_NAME = 'UCB'

if savestr is '' or TRACK_NAME is '':
	sys.exit('\nspecify which file to load... \n')

# choose vehicle params and specify indices of the nodes
if TRACK_NAME is 'ETHZ':
	params = ORCA()
	track = ETHZ()
	NODES = [33, 67, 116, 166, 203, 239, 274, 309, 344, 362, 382, 407, 434, 448, 470, 514, 550, 586, 622, 657, 665]
	LASTIDX = 0

elif TRACK_NAME is 'ETHZMobil':
	params = ORCA()
	track = ETHZMobil()
	NODES = [7, 21, 37, 52, 66, 81, 97, 111, 136, 160, 175, 191, 205, 220, 236, 250, 275, 299, 337, 376]
	LASTIDX = 0

elif TRACK_NAME is 'UCB':
	params = F110()
	track = UCB()
	NODES = [10, 32, 44, 67, 83, 100, 113, 127, 144, 160, 175, 191]
	LASTIDX = 0

theta = track.theta_track[NODES]

#####################################################################
# load saved data

data = np.load('results/{}_raceline_data-{}.npz'.format(TRACK_NAME, savestr))
y_ei = data['y_ei']
y_nei = data['y_nei']
y_rnd = data['y_rnd']
iters = data['iters']
train_x_all_ei = data['train_x_all_ei']
train_x_all_nei = data['train_x_all_nei']
train_x_all_random = data['train_x_all_random']
train_y_all_ei = data['train_y_all_ei'].squeeze(-1)
train_y_all_nei = data['train_y_all_nei'].squeeze(-1)
train_y_all_random = data['train_y_all_random'].squeeze(-1)
N_TRIALS = train_x_all_ei.shape[0]
N_DIMS = train_x_all_ei.shape[-1]

#####################################################################
# plot best lap times
filepath = 'results/{}_convergence.png'.format(TRACK_NAME)

def ci(y):
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

plt.figure()
plt.grid(True)

plt.gca().set_prop_cycle(None)
plt.plot(iters, y_rnd.mean(axis=0), linewidth=1.5)
plt.plot(iters, y_ei.mean(axis=0), linewidth=1.5)
plt.plot(iters, y_nei.mean(axis=0), linewidth=1.5)

plt.gca().set_prop_cycle(None)
plt.fill_between(iters, y_rnd.mean(axis=0)-ci(y_rnd), y_rnd.mean(axis=0)+ci(y_rnd), label="random", alpha=0.2)
plt.fill_between(iters, y_ei.mean(axis=0)-ci(y_ei), y_ei.mean(axis=0)+ci(y_ei), label="EI", alpha=0.2)
plt.fill_between(iters, y_nei.mean(axis=0)-ci(y_nei), y_nei.mean(axis=0)+ci(y_nei), label="NEI", alpha=0.2)

plt.xlabel('# number of observations (beyond initial points)')
plt.ylabel('best lap times [s]')
plt.xlim([0, 50])
plt.legend(loc='lower left')

if SAVE_RESULTS:
	plt.savefig(filepath, dpi=600, bbox_inches='tight')

#####################################################################
# plot best trajectory
filepath = 'results/{}_bestlap.png'.format(TRACK_NAME)

n_waypoints = N_DIMS
n_samples = 500

x_inner, y_inner = track.x_inner, track.y_inner
x_center, y_center = track.x_center, track.y_center
x_outer, y_outer = track.x_outer, track.y_outer

rand_traj = randomTrajectory(track=track, n_waypoints=n_waypoints)

def gen_traj(x_all, idx, sim):
	w_idx = x_all[sim][idx]
	wx, wy = rand_traj.calculate_xy(
        width=w_idx,
        last_index=NODES[LASTIDX],
        theta=theta,
        )
	sp = Spline2D(wx, wy)
	s = np.linspace(0, sp.s[-1]-0.001, n_samples)
	x, y = [], []
	for i_s in s:
		ix, iy = sp.calc_position(i_s)
		x.append(ix)
		y.append(iy)
	return wx, wy, x, y

fig = plt.figure()
ax = plt.gca()
ax.axis('equal')
plt.plot(x_center, y_center, '--k', lw=0.5, alpha=0.5)
plt.plot(x_outer, y_outer, 'k', lw=0.5, alpha=0.5)
plt.plot(x_inner, y_inner, 'k', lw=0.5, alpha=0.5)

# best trajectory
sim, pidx = np.unravel_index(np.argmin(train_y_all_nei), train_y_all_nei.shape)
wx_nei, wy_nei, x_nei, y_nei = gen_traj(train_x_all_nei, pidx, sim)
plt.plot(wx_nei[:-1], wy_nei[:-1], linestyle='', marker='D', ms=5)
time, speed, inputs = calcMinimumTimeSpeedInputs(x_nei, y_nei, **params)
x = np.array(x_nei)
y = np.array(y_nei)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(speed.min(), speed.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(speed)
lc.set_linewidth(2)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')

if SAVE_RESULTS:
	np.savez('results/{}_optimalxy-{}.npz'.format(track_name, savestr), x=x, y=y)
	np.savez('results/{}_raceline-{}.npz'.format(track_name, savestr), x=x, y=y, time=time, speed=speed, inputs=inputs)
	plt.savefig(filepath, dpi=600, bbox_inches='tight')

#####################################################################

plt.show()
