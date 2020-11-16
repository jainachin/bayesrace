""" Plot optimal racing lines from saved results.
    See generate_raceline_f110.py.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np

from bayes_race.tracks import MAP2, ComputeIO
from bayes_race.params import F110
from bayes_race.raceline import randomTrajectory
from bayes_race.raceline import calcMinimumTimeSpeedInputs
from bayes_race.utils import Spline2D

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

#####################################################################
# which data

SAVE_RESULTS = False

params = F110()
track_name = 'MAP2'

if track_name is 'MAP2':
    savestr = '20201103033427'
    track = MAP2()
    NODES = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 5]
    LASTIDX = 1

theta = track.theta_track[NODES]


#####################################################################
# inner outer lines
x_inner, y_inner, x_outer, y_outer = ComputeIO(track)

#####################################################################
# load saved data

data = np.load('results/{}_raceline_data-{}.npz'.format(track_name, savestr))
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
filepath = 'results/{}_convergence.png'.format(track_name)

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

plt.xlabel('\# number of observations (beyond initial points)')
plt.ylabel('best lap times [s]')
plt.xlim([0, 50])
plt.legend(loc='lower left')

if SAVE_RESULTS:
    plt.savefig(filepath, dpi=600, bbox_inches='tight')

#####################################################################
# plot best trajectory
filepath = 'results/{}_bestlap.png'.format(track_name)

n_waypoints = N_DIMS
n_samples = 500
sim = 0

x_center, y_center = track.x_center, track.y_center

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

EI = True
if EI:
  train_x_all_nei = train_x_all_ei
  train_y_all_nei = train_y_all_ei

# best trajectory
sim, pidx = np.unravel_index(np.argmin(train_y_all_nei), train_y_all_nei.shape)
wx_nei, wy_nei, x_nei, y_nei = gen_traj(train_x_all_nei, pidx, sim)
plt.plot(wx_nei[:-2], wy_nei[:-2], linestyle='', marker='D', ms=5)
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
