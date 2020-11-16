"""	Kinematic bicycle model.
	
	Use Kinematic class to:
	1. simulate continuous model
	2. linearize continuous model
	3. discretize continuous model
	4. simulate continuously linearized discrete model
	5. compare continuous and discrete models

	Two implementations are included for continuous system:
	1. 4 states: preferred way of solving (default)
	2. 6 states: don't use unless you understand
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from bayes_race.models.model import Model
from bayes_race.params import F110


class Kinematic(Model):

	def __init__(self, lf, lr, **kwargs):
		"""	specify model params here
		"""
		self.lf = lf
		self.lr = lr
		self.dr = lr/(lf+lr)
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
			x is a 4x1 vector: [x, y, psi, v]^T
			u is a 2x1 vector: [acc, steer]^T
		"""
		acc = u[0]
		steer = u[1]
		psi = x[2]
		vel = x[3]
		beta = np.arctan(self.dr*np.tan(steer))

		dxdt = np.zeros(4)
		dxdt[0] = vel*np.cos(psi+beta)
		dxdt[1] = vel*np.sin(psi+beta)
		dxdt[2] = vel*np.sin(beta)/self.lr
		dxdt[3] = acc
		return dxdt

	def casadi(self, x, u, dxdt):
		"""	write dynamics as first order ODE: dxdt = f(x(t))
			x is a 4x1 vector: [x, y, psi, v]^T
			u is a 2x1 vector: [acc, steer]^T
			dxdt is a casadi.SX variable
		"""
		acc = u[0]
		steer = u[1]
		psi = x[2]
		vel = x[3]
		beta = cs.atan(self.dr*cs.tan(steer))

		dxdt[0] = vel*cs.cos(psi+beta)
		dxdt[1] = vel*cs.sin(psi+beta)
		dxdt[2] = vel*cs.sin(beta)/self.lr
		dxdt[3] = acc
		return dxdt			

	def sim_continuous6(self, x0, u, t, v0=None):
		"""	simulates the model with given input vector
			x0 is the initial state of size 3xn
			u is the input vector of size 2xn
			t is the time vector of size 1x(n+1)
		"""
		self.n_states = 6
		n_steps = u.shape[1]
		x = np.zeros([6, n_steps+1])
		x[:3,0] = x0
		if v0 is not None:
			x[3:5,0] = v0
		for ids in range(1, n_steps+1):
			x[:,ids-1] = self._reset_yaw_rate(x[:,ids-1], u[:,ids-1]) 	# only for integration
			x[:,ids] = self._integrate(x[:,ids-1], u[:,ids-1], t[ids-1], t[ids])
		return x[:3,:], x[3:,:]

	def _diffequation6(self, t, x, u):
		"""	write dynamics as first order ODE: dxdt = f(x(t))
			x is a 6x1 vector: [x, y, psi, dxdt, dydt, dpsidt]^T
			u is a 2x1 vector: [acc, steer]^T
		"""
		acc = u[0]
		steer = u[1]
		psi = x[2]
		vel = np.linalg.norm(x[3:5],2)

		dr = self.lr/(self.lf+self.lr)
		beta = np.arctan(dr*np.tan(steer))

		dxdt = np.zeros(6)
		# dxdt[0] = vel*np.cos(psi+beta)
		# dxdt[1] = vel*np.sin(psi+beta)
		dxdt[:3] = x[3:]
		dxdt[2] = vel*np.sin(beta)/self.lr
		A = acc * np.array([
			np.cos(psi+beta),
			np.sin(psi+beta),
			np.sin(beta)/self.lr
			])
		B = vel * dxdt[2] * np.array([
			-np.sin(psi+beta),
			np.cos(psi+beta),		
			0
			])
		dxdt[3:] = A + B	
		return dxdt

	def _reset_yaw_rate(self, x, u):
		"""	resets yaw rate to correspond to current velocity and steer
			x is a 6x1 vector: [x, y, psi, dxdt, dydt, dpsidt]^T
			u is a 2x1 vector: [acc, steer]^T
		"""
		vel = np.linalg.norm(x[3:5],2)
		dr = self.lr/(self.lf+self.lr)
		beta = np.arctan(dr*np.tan(u[1]))
		x[5] = vel * np.sin(beta) / self.lr
		return x

	def calc_forces(self, x, u, mass, **kwargs):
		"""	return lateral and longitudinal forces
		"""
		acc = u[0]
		steer = u[1]
		vel = x[3]
		beta = np.arctan(self.dr*np.tan(steer))
		radius = self.lr/np.sin(beta)
		Fx, Fy = mass*acc, mass*(vel**2)/radius
		return Fx, Fy

	def sim_discrete(self, x0, u, Ts):
		"""	simulates a continuously linearized discrete model
			u is the input vector of size 2xn
			Ts is the sampling time
		"""
		n_steps = u.shape[1]
		x = np.zeros([4, n_steps+1])
		dxdt = np.zeros([4, n_steps+1])
		dxdt[:,0] = self._diffequation(None, x0, [0, 0])
		x[:,0] = x0
		for ids in range(1, n_steps+1):
			_, _, g = self.linearize(x0=x[:,ids-1], u0=u[:,ids-1])
			x[:,ids] = x[:,ids-1] + g*Ts
			dxdt[:,ids] = self._diffequation(None, x[:,ids], u[:,ids-1])
		return x, dxdt

	def linearize(self, x0, u0):
		"""	linearize at a given x0, u0
			for a given continuous system dxdt = f(x(t))
			calculate A = ∂f/∂x, B = ∂f/∂u, g = f evaluated at x0, u0
			A is 4x4, B is 4x2, g is 4x1
		"""
		lr = self.lr
		dr = self.dr
		acc = u0[0]
		steer = u0[1]
		psi = x0[2]
		vel = x0[3]

		tandelta = np.tan(steer)
		cosdelta = np.cos(steer)
		rtandelta = dr*tandelta
		beta = np.arctan(rtandelta)
		cospsibeta = np.cos(psi + beta)
		sinpsibeta = np.sin(psi + beta)
		sinbeta = np.sin(beta)
		cosbeta = np.cos(beta)
		delarctan = 1/(1+(dr*tandelta)**2)
		sec2delta = 1/(cosdelta**2)

		A = np.array([
			[0, 0, -vel*sinpsibeta, 	 cospsibeta],
			[0, 0,  vel*cospsibeta, 	 sinpsibeta],
			[0, 0,  			 0, 	 sinbeta/lr],
			[0, 0, 				 0, 			  0]
			])
		B = np.array([
			[0, -vel*sinpsibeta*delarctan*dr*sec2delta],
			[0,  vel*cospsibeta*delarctan*dr*sec2delta],
			[0,  vel*cosbeta*delarctan*dr*sec2delta/lr],
			[1,   			 						  0],
			])
		g = np.array([
			[vel*cospsibeta],
			[vel*sinpsibeta],
			[vel*sinbeta/lr],
			[			acc],
			]).reshape(-1,)
		return A, B, g


if __name__ == '__main__':
	"""	test cases 1-3 use 4 states continuous model (preferred)
		test cases 4-6 use 4 states discrete model (preferred)
		test cases 3-5 use 6 states continuous model (avoid using)
		test pairs (1,4,7), (2,5,8) and (3,6,9) should give same results
	"""

	# vehicle parameters for F1/10
	params = F110()
	model = Kinematic(**params)
	
	test_case = 3

	#####################################################################
	# CONTINUOUS MODEL 4 STATES

	# start at origin with init velocity [3, 3] m/s
	# apply constant acceleration 1 m/s^2 for 1s and then move at constant speed
	if test_case == 1:
		n_steps = 100
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, 2, n_steps+1)
		inputs[0,:50] = 1
		x_init = np.array([0, 0, np.pi/4, 3*np.sqrt(2)])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad
	if test_case == 2:
		n_steps = 200
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, 4, n_steps+1)
		inputs[1,:] = 0.2
		x_init = np.array([0, 0, 0, 3])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad after 2 sec
	if test_case == 3:
		n_steps = 400
		inputs = np.zeros([2, n_steps])
		inputs[1,100:] = 0.2
		time = np.linspace(0, 8, n_steps+1)
		x_init = np.array([0, 0, 0, 3])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	#####################################################################
	# DISCRETE MODEL 4 STATES
	
	# start at origin with init velocity [3, 3] m/s
	# apply constant acceleration 1 m/s^2 for 1s and then move at constant speed
	if test_case == 4:
		Ts = 0.02
		n_steps = int(4/Ts)
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, n_steps+1, n_steps+1)*Ts
		inputs[0,:int(n_steps/2)] = 1
		x_init = np.array([0, 0, np.pi/4, 3*np.sqrt(2)])
		x_disc, dxdt_disc = model.sim_discrete(x_init, inputs, Ts)
		model.plot_results(time, x_disc, dxdt_disc, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad
	if test_case == 5:
		Ts = 0.2
		n_steps = int(4/Ts)
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, n_steps+1, n_steps+1)*Ts
		inputs[1,:] = 0.2
		x_init = np.array([0, 0, 0, 3])
		x_disc, dxdt_disc = model.sim_discrete(x_init, inputs, Ts)
		model.plot_results(time, x_disc, dxdt_disc, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad after 2 sec
	if test_case == 6:
		Ts = 0.02
		n_steps = int(8/Ts)
		inputs = np.zeros([2, n_steps])
		inputs[1,int(n_steps/4):] = 0.2
		time = np.linspace(0, n_steps+1, n_steps+1)*Ts
		x_init = np.array([0, 0, 0, 3])
		x_disc, dxdt_disc = model.sim_discrete(x_init, inputs, Ts)
		model.plot_results(time, x_disc, dxdt_disc, inputs)

	#####################################################################
	# CONTINUOUS MODEL 6 STATES

	# start at origin with init velocity [3, 3] m/s in orientation (psi) pi/4
	# apply constant acceleration 1 m/s^2 for 1s and then move at constant speed
	if test_case == 7:
		n_steps = 100
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, 2, n_steps+1)
		inputs[0,:50] = 1
		x_init = np.array([0, 0, np.pi/4])
		v_init = np.array([3, 3])
		x_cont, dxdt_cont = model.sim_continuous6(x_init, inputs, time, v0=v_init)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad
	if test_case == 8:
		n_steps = 200
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, 4, n_steps+1)
		inputs[1,:] = 0.2
		dr = lr/(lf+lr)
		beta = np.arctan(dr*np.tan(0.2))
		x_init = np.array([0, 0, -beta])
		v_init = np.array([3, 0])
		x_cont, dxdt_cont = model.sim_continuous6(x_init, inputs, time, v0=v_init)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad after 2 sec
	# to get perfect circles, set yaw to -beta when steering is turned first time
	if test_case == 9:
		n_steps = 400
		inputs = np.zeros([2, n_steps])
		inputs[1,100:] = 0.2
		time = np.linspace(0, 8, n_steps+1)
		x_init = np.array([0, 0, 0])
		v_init = np.array([3, 0])
		x_cont, dxdt_cont = model.sim_continuous6(x_init, inputs, time, v0=v_init)
		model.plot_results(time, x_cont, dxdt_cont, inputs)