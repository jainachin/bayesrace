"""	Generate data by simulating dynamic model using pure pursuit control.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time as tm
import numpy as np
import matplotlib.pyplot as plt

from bayes_race.params import ORCA
from bayes_race.models import Dynamic
from bayes_race.tracks import ETHZ, ETHZMobil
from bayes_race.pp import purePursuit

#####################################################################
# settings

SAVE_RESULTS = False

SAMPLING_TIME = 0.02				# in [s]
SIM_TIME = 10						# in [s]
LD = 0.1
KP = 2

#####################################################################
# load vehicle parameters

params = ORCA(control='pwm')
model = Dynamic(**params)

#####################################################################
# load track

TRACK_NAME = 'ETHZMobil'
if TRACK_NAME == 'ETHZ':
	track = ETHZ(reference='optimal')  		# ETHZ() or ETHZMobil()
elif TRACK_NAME == 'ETHZMobil':
	track = ETHZMobil(reference='optimal')  # ETHZ() or ETHZMobil()

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs

#####################################################################
# closed-loop simulation

# initialize
states = np.zeros([n_states, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])

x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))

# dynamic plot
fig = track.plot(color='k', grid=False)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], 'r', alpha=0.8, label='pure pursuit')
xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(xyproj[0], xyproj[1], 'g', marker='o', alpha=0.5, markersize=5)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()

plt.figure()
plt.grid(True)
ax2 = plt.gca()
LnFfy, = ax2.plot(0, 0, label='Ffy')
LnFrx, = ax2.plot(0, 0, label='Frx')
LnFry, = ax2.plot(0, 0, label='Fry')
plt.xlim([0, SIM_TIME])
plt.ylim([-params['mass']*9.81, params['mass']*9.81])
plt.xlabel('time [s]')
plt.ylabel('force [N]')
plt.legend()
plt.ion()
plt.show()

# main simulation loop
for idt in range(n_steps):
		
	uprev = inputs[:,idt-1]
	x0 = states[:,idt]

	start = tm.time()
	upp = purePursuit(x0, LD, KP, track, params)
	end = tm.time()
	inputs[:,idt] = upp
	print("iteration: {}, time to solve: {:.2f}".format(idt, end-start))

	# update current position with numerical integration (exact model)
	x_next, dxdt_next = model.sim_continuous(states[:,idt], inputs[:,idt].reshape(-1,1), [0, Ts])
	states[:,idt+1] = x_next[:,-1]
	dstates[:,idt+1] = dxdt_next[:,-1]
	Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])

	# update plot
	xyproj, _ = track.project(x=x0[0], y=x0[1], raceline=track.raceline)

	LnS.set_xdata(states[0,:idt+1])
	LnS.set_ydata(states[1,:idt+1])

	LnP.set_xdata(xyproj[0])
	LnP.set_ydata(xyproj[1])

	LnFfy.set_xdata(time[:idt+1])
	LnFfy.set_ydata(Ffy[:idt+1])

	LnFrx.set_xdata(time[:idt+1])
	LnFrx.set_ydata(Frx[:idt+1])

	LnFry.set_xdata(time[:idt+1])
	LnFry.set_ydata(Fry[:idt+1])

	plt.pause(Ts/100)

plt.ioff()

#####################################################################
# save data

if SAVE_RESULTS:
	np.savez(
		'../data/DYN-PP-{}.npz'.format(TRACK_NAME),
		time=time,
		states=states,
		dstates=dstates,
		inputs=inputs,
		)

#####################################################################
# plots

# plot speed
plt.figure()
vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
plt.plot(time[:n_steps], vel[:n_steps], label='abs')
plt.plot(time[:n_steps], states[3,:n_steps], label='v')
plt.plot(track.t_raceline, track.v_raceline, label='ref')
plt.xlabel('time [s]')
plt.ylabel('speed [m/s]')
plt.grid(True)
plt.legend()

# plot acceleration
plt.figure()
plt.plot(time[:n_steps], inputs[0,:n_steps])
plt.xlabel('time [s]')
plt.ylabel('acceleration [m/s^2]')
plt.grid(True)

# plot steering angle
plt.figure()
plt.plot(time[:n_steps], inputs[1,:n_steps])
plt.xlabel('time [s]')
plt.ylabel('steering [rad]')
plt.grid(True)

# plot inertial heading
plt.figure()
plt.plot(time[:n_steps], states[2,:n_steps])
plt.xlabel('time [s]')
plt.ylabel('orientation [rad]')
plt.grid(True)

plt.show()