"""	Plot offline data.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from bayes_race.tracks import ETHZ
from bayes_race.models import Dynamic

#####################################################################
# settings

SAVE_RESULTS = False

SAMPLING_TIME = 0.02
HORIZON = 20

#####################################################################
# load track

N_SAMPLES = 390
TRACK_NAME = 'ETHZ'
LAPS_DONE = 1

track = ETHZ(reference='optimal', longer=True)

#####################################################################
# load inputs used to simulate Dynamic model

SUFFIX = 'NOCONS-'

data = np.load('../data/DYN-NMPC-{}{}.npz'.format(SUFFIX, TRACK_NAME))
time_dyn = data['time'][:N_SAMPLES+1]
states_dyn = data['states'][:,:N_SAMPLES+1]
inputs_dyn = data['inputs'][:,:N_SAMPLES]

data = np.load('../data/DYN-GPMPC-{}{}.npz'.format(SUFFIX, TRACK_NAME))
time_gp = data['time'][:N_SAMPLES+1]
states_gp = data['states'][:,:N_SAMPLES+1]
inputs_gp = data['inputs'][:,:N_SAMPLES]

data = np.load('../data/DYN-GPMPC{}-{}{}.npz'.format(LAPS_DONE, SUFFIX, TRACK_NAME))
time_gp_updated = data['time'][:N_SAMPLES+1]
states_gp_updated = data['states'][:,:N_SAMPLES+1]
inputs_gp_updated = data['inputs'][:,:N_SAMPLES]

#####################################################################
# plots

plt.figure(figsize=(6,4))
plt.axis('equal')
plt.plot(-track.y_outer, track.x_outer, 'k', lw=0.5, alpha=0.5)
plt.plot(-track.y_inner, track.x_inner, 'k', lw=0.5, alpha=0.5)
# plt.plot(-states_gp[1], states_gp[0], 'r', lw=1, label='$f_{\mathrm{corr}}$')
plt.plot(-states_gp_updated[1], states_gp_updated[0], 'r', lw=1, label='MPC (updated GP)')
plt.plot(-states_dyn[1], states_dyn[0], '--k', lw=1, label='MPC (exact)')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.15), frameon=False)

INDEX = 0
plt.scatter(-states_gp_updated[1,INDEX], states_gp_updated[0,INDEX], color='r', marker='o', alpha=0.8, s=15)
plt.text(-states_gp[1,INDEX], states_gp[0,INDEX]+0.05, '0', color='k', fontsize=10, ha='center', va='bottom')
plt.scatter(-states_dyn[1,INDEX], states_dyn[0,INDEX], color='k', marker='o', alpha=0.8, s=15)
for INDEX in range(HORIZON+5,N_SAMPLES,HORIZON+5):
	plt.scatter(-states_gp_updated[1,INDEX], states_gp_updated[0,INDEX], color='r', marker='o', alpha=0.8, s=15)
	plt.text(-states_gp_updated[1,INDEX]+0.05, states_gp_updated[0,INDEX]+0.05, '{}'.format(INDEX*SAMPLING_TIME), color='r', fontsize=10)	
	plt.scatter(-states_dyn[1,INDEX], states_dyn[0,INDEX], color='k', marker='o', alpha=0.8, s=15)
	plt.text(-states_dyn[1,INDEX]-0.05, states_dyn[0,INDEX]-0.05, '{}'.format(INDEX*SAMPLING_TIME), color='k', fontsize=10, ha='right', va='top')

filepath = 'track_mpc_lap{}.png'.format(LAPS_DONE)
if SAVE_RESULTS:
	plt.savefig(filepath, dpi=600, bbox_inches='tight')

plt.figure(figsize=(6,4.3))
gs = gridspec.GridSpec(2,1)

plt.subplot(gs[0,:])
ax = plt.gca()
plt.plot(time_gp_updated[:-1], inputs_gp_updated[1], 'r', lw=1, label='MPC (updated GP)')
plt.plot(time_dyn[:-1], inputs_dyn[1], '--k', lw=1, label='MPC (exact)')
plt.ylabel('steering $\delta$ [rad]')
plt.xlim([0, N_SAMPLES*SAMPLING_TIME])
ax.set_xticks([])
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.31), frameon=False)

plt.subplot(gs[1,:])
plt.plot(time_gp_updated[:-1], inputs_gp_updated[0], 'r', lw=1, label='MPC (updated GP)')
plt.plot(time_dyn[:-1], inputs_dyn[0], '--k', lw=1, label='MPC (exact)')
plt.ylabel('PWM $d$ [-]')
plt.xlabel('time [s]')
plt.xlim([0, N_SAMPLES*SAMPLING_TIME])

filepath = 'inputs_mpc_lap{}.png'.format(LAPS_DONE)
if SAVE_RESULTS:
	plt.savefig(filepath, dpi=600, bbox_inches='tight')

plt.show()