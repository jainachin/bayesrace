"""	Plot MPC results.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from bayes_race.tracks import MAP2
from bayes_race.tracks import ComputeIO

#####################################################################
# settings

SAVE_RESULTS = False

SAMPLING_TIME = 0.02
HORIZON = 20

#####################################################################
# load track

MAP = 2
TRACK_NAME = 'MAP{}'.format(MAP)
SIM_TIME = 4.95
SIM_TIME2 = 8.35
N_SAMPLES = int(SIM_TIME/SAMPLING_TIME)
N_SAMPLES2 = int(SIM_TIME2/SAMPLING_TIME)

track = MAP2(reference='optimal')

#####################################################################
# compute inner and outer lines

x_inner, y_inner, x_outer, y_outer = ComputeIO(track)

#####################################################################
# load data

data = np.load('../data/DYN-NMPC-GP-NOCONS-{}.npz'.format(TRACK_NAME))
time_gp = data['time'][:N_SAMPLES+1]
states_gp = data['states'][:,:N_SAMPLES+1]
inputs_gp = data['inputs'][:,:N_SAMPLES]

data = np.load('../data/DYN-NMPC-NOCONS-{}.npz'.format(TRACK_NAME))
time_kin= data['time'][:N_SAMPLES2+1]
states_kin= data['states'][:,:N_SAMPLES2+1]
inputs_kin= data['inputs'][:,:N_SAMPLES2]

# speed
# plt.figure(figsize=(6,4))
# plt.plot(time_gp, states_gp[3], '-b', lw=1, label='vx')
# plt.plot(time_gp, states_gp[4], '-g', lw=1, label='vy')
# plt.plot(time_gp, np.sqrt(states_gp[3]**2 + states_gp[4]**2), '--r', lw=1, label='vabs')
# plt.plot(track.t_raceline, track.v_raceline, '-k', lw=1, label='vref')
# plt.legend()

# trajectory
plt.figure(figsize=(6,4))
plt.axis('equal')
# plt.plot(track.x_raceline, track.y_raceline, '--k', lw=1, label='raceline')
plt.plot(x_inner, y_inner, '-k', lw=0.75, alpha=0.5)
plt.plot(x_outer, y_outer, '-k', lw=0.75, alpha=0.5)
plt.plot(states_gp[0,:N_SAMPLES], states_gp[1,:N_SAMPLES], '-r', lw=1, label='MPC (GP)')
plt.plot(states_kin[0,:N_SAMPLES2], states_kin[1,:N_SAMPLES2], '--k', lw=1, label='MPC (without GP)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.15), frameon=False)

INDEX = 0
plt.scatter(states_gp[0,INDEX], states_gp[1,INDEX], color='r', marker='o', alpha=0.8, s=15)
for INDEX in range(HORIZON+5,N_SAMPLES,HORIZON+5):
	plt.scatter(states_gp[0,INDEX], states_gp[1,INDEX], color='r', marker='o', alpha=0.8, s=15)
	plt.text(states_gp[0,INDEX]-0.2, states_gp[1,INDEX]-0.2, '{}'.format(INDEX*SAMPLING_TIME), color='r', fontsize=10, ha='right', va='top')

for INDEX in range(HORIZON+5,N_SAMPLES2,HORIZON+5):
	plt.scatter(states_kin[0,INDEX], states_kin[1,INDEX], color='k', marker='o', alpha=0.8, s=15)
	plt.text(states_kin[0,INDEX]+0.5, states_kin[1,INDEX]+0.5, '{}'.format(INDEX*SAMPLING_TIME), color='k', fontsize=10, ha='right', va='top')

filepath = 'track_mpc_f110.png'
if SAVE_RESULTS:
	plt.savefig(filepath, dpi=600, bbox_inches='tight')

# inputs
plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(2,1)

plt.subplot(gs[0,:])
plt.plot(time_gp[:-1], inputs_gp[1], '-r', lw=1, label='MPC (GP)')
plt.plot(time_kin[:-1], inputs_kin[1], '--k', lw=1, label='MPC (without GP)')
plt.ylabel('steering [rad]')
plt.xlim([0, N_SAMPLES2*SAMPLING_TIME])
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.31), frameon=False)

plt.subplot(gs[1,:])
plt.plot(time_gp[:-1], inputs_gp[0], '-r', lw=1, label='MPC (GP)')
plt.plot(time_kin[:-1], inputs_kin[0], '--k', lw=1, label='MPC (without GP)')
plt.ylabel('acc [m/s$^2$]')
plt.xlabel('time [s]')
plt.xlim([0, N_SAMPLES2*SAMPLING_TIME])

filepath = 'inputs_mpc_f110.png'
if SAVE_RESULTS:
	plt.savefig(filepath, dpi=600, bbox_inches='tight')

plt.show()
