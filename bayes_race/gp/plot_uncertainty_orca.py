"""	Plot offline data.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import matplotlib.pyplot as plt

from bayes_race.params import ORCA
from bayes_race.tracks import ETHZ, ETHZMobil
from bayes_race.models import Dynamic

#####################################################################
# settings

SAVE_RESULTS = False

CTYPE = 'NMPC'
SAMPLING_TIME = 0.02
TRACK_NAME = 'ETHZ'

LAPS_DONE = 0
if LAPS_DONE == 0:
	LAPS_DONE = ''

filepath = 'track_validation.png'
label = 'MPC (exact)'
N_SAMPLES = 400
THRESHOLD = 0.25

#####################################################################
# load vehicle parameters

params = ORCA(control='pwm')
model = Dynamic(**params)

#####################################################################
# load track

if TRACK_NAME == 'ETHZ':
	track = ETHZ(reference='optimal')  		# ETHZ() or ETHZMobil()
elif TRACK_NAME == 'ETHZMobil':
	track = ETHZMobil(reference='optimal')  # ETHZ() or ETHZMobil()

#####################################################################
# load inputs used to simulate Dynamic model

data = np.load('../data/DYN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
time = data['time'][:N_SAMPLES+1]

states = np.zeros([7,N_SAMPLES+1])
states[:6,:] = data['states'][:6,:N_SAMPLES+1]
states[6,1:] = data['inputs'][1,:N_SAMPLES]

inputs = data['inputs'][:,:N_SAMPLES]
inputs[1,0] = inputs[1,0]
inputs[1,1:] = np.diff(inputs[1])
inputs[1] /= SAMPLING_TIME

#####################################################################
# plots

plt.figure(figsize=(6,4))
plt.axis('equal')

plt.plot(-track.y_outer, track.x_outer, 'k', lw=0.5, alpha=0.5)
plt.plot(-track.y_inner, track.x_inner, 'k', lw=0.5, alpha=0.5)
plt.plot(-states[1], states[0], 'k', lw=1, label=label)

data = np.load('../data/SIGMA-{}{}-{}.npz'.format(CTYPE, LAPS_DONE, TRACK_NAME))
std = data['std']
std = np.concatenate([np.array([0]),std])
threshold = std>THRESHOLD
plt.scatter(-states[1,threshold], states[0,threshold], label='high uncertainty', c='r', alpha=0.2)

plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5,1.15), frameon=False)

if SAVE_RESULTS:
	plt.savefig(filepath, dpi=600, bbox_inches='tight')

plt.show()