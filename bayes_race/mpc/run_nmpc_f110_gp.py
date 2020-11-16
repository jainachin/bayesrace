"""	Nonlinear MPC using Kinematic6 and GPs for model correction.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time as tm
import numpy as np
import casadi
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bayes_race.params import F110
from bayes_race.models import DynamicsST
from bayes_race.gp.utils import loadGPModel
from bayes_race.tracks import MAP2
from bayes_race.mpc.planner import ConstantSpeed
from bayes_race.mpc.gpmpc import setupNLP

#####################################################################
# CHANGE THIS

SAVE_RESULTS = False
ERROR_CORR = True
TRACK_CONS = False

#####################################################################
# default settings

SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

SUFFIX = ''
if ERROR_CORR:
	SUFFIX += 'GP-'

if not TRACK_CONS:
	SUFFIX += 'NOCONS-'

#####################################################################
# load vehicle parameters

params = F110()

#####################################################################
# load track

MAP = 2
TRACK_NAME = 'MAP{}'.format(MAP)
track = MAP2(reference='optimal')
if ERROR_CORR:
	SIM_TIME = 5.5
else:
	SIM_TIME = 9

#####################################################################
# load GP models

with open('../gp/f110/vxgp.pickle', 'rb') as f:
	(vxmodel, vxxscaler, vxyscaler) = pickle.load(f)
vxgp = loadGPModel('vx', vxmodel, vxxscaler, vxyscaler, kernel='RBF')
with open('../gp/f110/vygp.pickle', 'rb') as f:
	(vymodel, vyxscaler, vyyscaler) = pickle.load(f)
vygp = loadGPModel('vy', vymodel, vyxscaler, vyyscaler, kernel='Matern')
with open('../gp/f110/omegagp.pickle', 'rb') as f:
	(omegamodel, omegaxscaler, omegayscaler) = pickle.load(f)
omegagp = loadGPModel('omega', omegamodel, omegaxscaler, omegayscaler, kernel='Matern')
gpmodels = {
	'vx': vxgp,
	'vy': vygp,
	'omega': omegagp,
	'xscaler': vxxscaler,
	'yscaler': vxyscaler,
	}

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = 6
n_inputs = 2
horizon = HORIZON

#####################################################################
# define controller

nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, gpmodels, track, 
	track_cons=TRACK_CONS, error_correction=ERROR_CORR, input_acc=True)

#####################################################################
# closed-loop simulation

# initialize
states = np.zeros([n_states+1, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
states[:n_states,0] = x_init
previous_slip = 0.

print('starting at ({:.1f},{:.1f},{:.2f})'.format(x_init[0], x_init[1], x_init[2]))

if ERROR_CORR:
	scale = 1
else:
	scale = 0.6 	# fails above this due to model mismatch

# dynamic plot
fig = track.plot(color='k', grid=False)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], 'r', alpha=0.8)
LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=1, lw=0.5, label="reference")
xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(xyproj[0], xyproj[1], 'g', marker='o', alpha=0.5, markersize=5, label="current position")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-r', marker='o', markersize=1, lw=0.5, label="prediction")
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.ion()
plt.show()

# main simulation loop
for idt in range(n_steps-horizon):

	uprev = inputs[:,idt-1]
	x0 = states[:,idt]

	# planner based on BayesOpt
	xref, projidx = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, scale=scale)

	# solve NLP
	start = tm.time()	
	umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)
	end = tm.time()
	inputs[:,idt] = np.array([umpc[0,0], states[n_states,idt] + Ts*umpc[1,0]])
	print("iter: {}, cost: {:.5f}, time: {:.2f}".format(idt, fval, end-start))

	# update current position with numerical integration (exact model)
	states_st = np.array([
		states[0,idt],
		states[1,idt],
		inputs[1,idt-1],
		np.sqrt(states[3,idt]**2 + states[4,idt]**2),
		states[2,idt],
		states[5,idt],
		previous_slip,
		])
	inputs_st = np.array([umpc[1,0], umpc[0,0]])
	dxdt = DynamicsST(states_st, inputs_st, params)
	states_next_st = states_st + Ts*np.array(dxdt)

	# cast to states used for dynamic model
	states[:,idt+1] = [
		states_next_st[0],
		states_next_st[1],
		states_next_st[4],
		states_next_st[3]*np.cos(states_next_st[6]),
		states_next_st[3]*np.sin(states_next_st[6]),
		states_next_st[5],
		states_next_st[2],
		]
	previous_slip = states_next_st[6]

	# forward sim to predict over the horizon
	hstates2[:,0] = x0[:n_states]
	for idh in range(horizon):
		hstates2[:,idh+1] = xmpc[:n_states,idh+1]

	# update plot
	LnS.set_xdata(states[0,:idt+1])
	LnS.set_ydata(states[1,:idt+1])

	LnR.set_xdata(xref[0,1:])
	LnR.set_ydata(xref[1,1:])

	LnP.set_xdata(states[0,idt])
	LnP.set_ydata(states[1,idt])

	LnH2.set_xdata(hstates2[0])
	LnH2.set_ydata(hstates2[1])

	plt.pause(Ts/100)

plt.ioff()

#####################################################################
# save data

if SAVE_RESULTS:
	np.savez(
		'../data/DYN-NMPC-{}{}.npz'.format(SUFFIX, TRACK_NAME),
		time=time,
		states=states,
		inputs=inputs,
		)

#####################################################################
# plots

# plot speed
plt.figure()
plt.plot(time[:n_steps-horizon], states[3,:n_steps-horizon], label='vx')
plt.plot(time[:n_steps-horizon], states[4,:n_steps-horizon], label='vy')
plt.xlabel('time [s]')
plt.ylabel('speed [m/s]')
plt.grid(True)
plt.legend()

# plot acceleration
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[0,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('acceleration [-]')
plt.grid(True)

# plot steering angle
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[1,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('steering [rad]')
plt.grid(True)

# plot inertial heading
plt.figure()
plt.plot(time[:n_steps-horizon], states[2,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('orientation [rad]')
plt.grid(True)

plt.show()
