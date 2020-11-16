"""	Friction circle model. See Boyd's paper for details.
	
	Use FrictionCircle class to:
	1. simulate continuous model

"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from bayes_race.models.model import Model
from bayes_race.params import F110


class FrictionCircle(Model):

	def __init__(self, mass, **kwargs):
		"""	specify model params here
		"""
		self.mass = mass
		self.n_states = 4
		self.n_inputs = 2		
		Model.__init__(self)

	def sim_continuous(self, x0, u, t):
		"""	simulates the nonlinear continuous model with given input vector
			by numerical integration using 6th order Runge Kutta method
			x0 is the initial state of size 4x1
			u is the input vector of size 2xn
			t is the time vector of size 1x(n+1)
		"""
		n_steps = u.shape[1]
		x = np.zeros([4, n_steps+1])
		dxdt = np.zeros([4, n_steps+1])
		dxdt[:,0] = self._diffequation(None, x0, [0, 0])
		x[:,0] = x0
		for ids in range(1, n_steps+1):
			x[:,ids] = self._integrate(x[:,ids-1], u[:,ids-1], t[ids-1], t[ids])
			dxdt[:,ids] = self._diffequation(None, x[:,ids], u[:,ids-1])
		return x, dxdt

	def _diffequation(self, t, x, u):
		"""	write dynamics as first order ODE: dxdt = f(x(t))
			x is a 4x1 vector: [x, y, vx, vy]^T
			u is a 2x1 vector: [Fx, Fy]^T
		"""
		phi = np.arctan2(x[3],x[2])
		R = np.zeros((2,2))
		R[0,0] = np.cos(phi)
		R[0,1] = -np.sin(phi)
		R[1,0] = np.sin(phi)
		R[1,1] = np.cos(phi)

		M = np.zeros((2,2))
		M[0,0] = self.mass
		M[1,1] = self.mass

		C = np.zeros((2,2))
		d = np.zeros((2))

		dxdt = np.zeros(4)
		dxdt[:2] = x[2:]
		dxdt[2:] = np.dot(np.linalg.inv(M), np.dot(R, u) - np.dot(C, x[2:]) - d)
		return dxdt


if __name__ == '__main__':
	"""	test cases 1-3 use 4 states continuous model
		these results are perfectly identical to test cases in kinematic.py
	"""

	# vehicle parameters for F1/10
	params = F110()
	model = FrictionCircle(**params)

	test_case = 3

	#####################################################################
	# CONTINUOUS MODEL 4 STATES

	# start at origin with init velocity [3, 3] m/s
	# apply constant acceleration 1 m/s^2 for 1s and then move at constant speed
	if test_case == 1:
		n_steps = 100
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, 2, n_steps+1)
		inputs[0,:50] = 1*mass
		x_init = np.array([0, 0, 3, 3])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs, friction_circle=True)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad
	if test_case == 2:
		n_steps = 200
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, 4, n_steps+1)
		dr = lr/(lf+lr)
		beta = np.arctan(dr*np.tan(0.2))
		radius = lr/np.sin(beta)
		inputs[1,:] = mass*(3**2)/radius
		x_init = np.array([0, 0, 3, 0])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs, friction_circle=True)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad after 2 sec
	if test_case == 3:
		n_steps = 400
		inputs = np.zeros([2, n_steps])
		lf, lr, mass = params['lf'], params['lr'], params['mass']
		dr = lr/(lf+lr)
		beta = np.arctan(dr*np.tan(0.2))
		radius = lr/np.sin(beta)
		inputs[1,100:] = mass*(3**2)/radius
		time = np.linspace(0, 8, n_steps+1)
		x_init = np.array([0, 0, 3, 0])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs, friction_circle=True)