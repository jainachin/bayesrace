"""	Generate data by simulating kinematic model.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import matplotlib.pyplot as plt

from bayes_race.params import F110
from bayes_race.models import Kinematic6

#####################################################################
# settings

SAVE_RESULTS = False

SAMPLING_TIME = 0.02
CTYPE = 'PP'
MAP = 2
TRACK_NAME = 'MAP{}'.format(MAP)

#####################################################################
# load vehicle parameters

params = F110()
model = Kinematic6(input_acc=True, **params)

#####################################################################
# load inputs used to simulate Dynamic model

data = np.load('DYN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
N_SAMPLES = data['inputs'].shape[1]

time = data['time'][:N_SAMPLES+1]

# add steering angle as 7th state
states_dyn = np.zeros([7,N_SAMPLES+1])
states_dyn[:6,:] = data['states'][:6,:N_SAMPLES+1]
states_dyn[6,1:] = data['inputs'][1,:N_SAMPLES]

# change 2nd input to rate of change in steering
inputs = data['inputs'][:,:N_SAMPLES]
inputs[1,0] = inputs[1,0]
inputs[1,1:] = np.diff(inputs[1])
inputs[1] /= SAMPLING_TIME

#####################################################################
# open-loop simulation

states_kin = np.zeros([7,N_SAMPLES+1])
states_kin[:,0] = states_dyn[:,0]
for idn in range(N_SAMPLES):
	x_next, dxdt_next = model.sim_continuous(states_dyn[:,idn], inputs[:,idn].reshape(-1,1), [0, SAMPLING_TIME])
	states_kin[:,idn+1] = x_next[:,-1]

#####################################################################
# visualize the difference

plt.figure()
plt.plot(time, states_kin[3,:], label='vx kinematic')
plt.plot(time, states_dyn[3,:], '--', label='vx dynamic')
plt.plot(time, states_kin[4,:], label='vy kinematic')
plt.plot(time, states_dyn[4,:], '--', label='vy dynamic')
plt.ylabel('speed [m/s]')
plt.xlabel('time [s]')
plt.legend()

plt.figure()
plt.plot(time, states_kin[5,:], label='kinematic')
plt.plot(time, states_dyn[5,:], label='dynamic')
plt.ylabel('yaw rate')
plt.legend()

plt.show()

#####################################################################
# save data

if SAVE_RESULTS:
	np.savez(
		'../data/DYN-{}-{}.npz'.format(CTYPE, TRACK_NAME),
		time=time,
		states=states_dyn,
		inputs=inputs,
		)
	np.savez(
		'../data/KIN-{}-{}.npz'.format(CTYPE, TRACK_NAME),
		time=time,
		states=states_kin,
		inputs=inputs,
		)
