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

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)

#####################################################################
# load inputs used to simulate Dynamic model

suffix = '-ALL'

for prefix in ['GP', 'KIN']:

	if prefix is 'GP':
		N_SAMPLES = 400
		label = 'predicted ($f_{\mathrm{corr}}$)'
		index_range = range(HORIZON+5,N_SAMPLES-10,HORIZON+5)
	elif prefix is 'KIN':
		N_SAMPLES = 425
		label = 'predicted ($f_{\mathrm{kin}}$)'
		index_range = range(HORIZON+5,N_SAMPLES,HORIZON+5)
	else:
		raise NotImplementedError

	data = np.load('../data/DYN-{}MPC-{}{}.npz'.format(prefix, TRACK_NAME, suffix))
	time_dyn = data['time'][:N_SAMPLES+1]
	states_model_dyn = data['states_model'][:,:,:N_SAMPLES+1]
	states_mpc_dyn = data['states_mpc'][:,:,:N_SAMPLES+1]

	#####################################################################
	# plots

	plt.figure(figsize=(6,4))
	plt.axis('equal')
	plt.plot(-track.y_outer, track.x_outer, 'k', lw=0.5, alpha=0.5)
	plt.plot(-track.y_inner, track.x_inner, 'k', lw=0.5, alpha=0.5)
	plt.plot(-states_mpc_dyn[1,0,:], states_mpc_dyn[0,0,:], '-k', lw=0.5)

	INDEX = 0
	plt.plot(-states_model_dyn[1,:,INDEX], states_model_dyn[0,:,INDEX], '-g', marker='o', markersize=1, lw=0.5, label='simulated ($f_{\mathrm{dyn}}$)')
	plt.plot(-states_mpc_dyn[1,:,INDEX], states_mpc_dyn[0,:,INDEX], '-r', marker='o', markersize=1, lw=0.5, label=label)
	plt.scatter(-states_model_dyn[1,0,INDEX], states_model_dyn[0,0,INDEX], color='b', marker='o', alpha=0.8, s=15)
	plt.text(-states_model_dyn[1,0,INDEX], states_model_dyn[0,0,INDEX]+0.05, '0', color='k', fontsize=10, ha='center', va='bottom')

	for INDEX in index_range:
		plt.plot(-states_model_dyn[1,:,INDEX], states_model_dyn[0,:,INDEX], '-g', marker='o', markersize=1, lw=0.5)
		plt.plot(-states_mpc_dyn[1,:,INDEX], states_mpc_dyn[0,:,INDEX], '-r', marker='o', markersize=1, lw=0.5)
		plt.scatter(-states_model_dyn[1,0,INDEX], states_model_dyn[0,0,INDEX], color='b', marker='o', alpha=0.8, s=15)
		plt.text(-states_model_dyn[1,0,INDEX]+0.05, states_model_dyn[0,0,INDEX]+0.05, '{}'.format(INDEX*SAMPLING_TIME), color='k', fontsize=10)

	plt.xlabel('$x$ [m]')
	plt.ylabel('$y$ [m]')
	plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.15), frameon=False)

	filepath = 'error_{}_mpc.png'.format(prefix.lower())
	if SAVE_RESULTS:
		plt.savefig(filepath, dpi=600, bbox_inches='tight')

plt.show()