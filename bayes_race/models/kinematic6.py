"""	Kinematic6 bicycle model.
	
	Use Kinematic6 class to:
	1. simulate continuous model
	2. linearize continuous model
	3. discretize continuous model
	4. simulate continuously linearized discrete model
	5. compare continuous and discrete models

"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import casadi as cs
from bayes_race.models.model import Model
from bayes_race.params import ORCA


class Kinematic6(Model):

	def __init__(self, lf, lr, mass, input_acc=False, **kwargs):
		"""	specify model params here
		"""
		self.lf = lf
		self.lr = lr
		self.dr = lr/(lf+lr)
		self.mass = mass
		self.input_acc = input_acc
		if not input_acc:
			self.Cm1 = kwargs['Cm1']
			self.Cm2 = kwargs['Cm2']

		self.n_states = 7
		self.n_inputs = 2
		Model.__init__(self)

	def sim_continuous(self, x0, u, t):
		"""	simulates the nonlinear continuous model with given input vector
			by numerical integration using 6th order Runge Kutta method
			x0 is the initial state of size 7x1
			u is the input vector of size 2xn
			t is the time vector of size 1x(n+1)
		"""
		n_steps = u.shape[1]
		x = np.zeros([self.n_states, n_steps+1])
		dxdt = np.zeros([self.n_states, n_steps+1])
		dxdt[:,0] = self._diffequation(None, x0, [0, 0])
		x[:,0] = x0
		for ids in range(1, n_steps+1):
			x[:,ids] = self._integrate(x[:,ids-1], u[:,ids-1], t[ids-1], t[ids])
			dxdt[:,ids] = self._diffequation(None, x[:,ids], u[:,ids-1])
		return x, dxdt

	def _diffequation(self, t, x, u):
		"""	write dynamics as first order ODE: dxdt = f(x(t))
			x is a 7x1 vector: [x, y, psi, vx, vy, omega, steer]^T
			u is a 2x1 vector: [pwm, change in steer]^T
		"""
		pwm = u[0]
		dsteer = u[1]
		psi = x[2]
		vx = x[3]
		vy = x[4]
		omega = x[5]
		steer = x[6]
		Frx = self.calc_forces(x, u)

		dxdt = np.zeros(self.n_states)
		dxdt[0] = vx*np.cos(psi) - vy*np.sin(psi)
		dxdt[1] = vx*np.sin(psi) + vy*np.cos(psi)
		dxdt[2] = omega
		dxdt[6] = dsteer
		dxdt[3] = 1/self.mass * Frx
		dxdt[4] = (dxdt[6]*vx + steer*dxdt[3]) * self.lr / (self.lf+self.lr)
		dxdt[5] = (dxdt[6]*vx + steer*dxdt[3]) * (self.lf+self.lr)
		return dxdt

	def casadi(self, x, u, dxdt):
		"""	write dynamics as first order ODE: dxdt = f(x(t))
			x is a 7x1 vector: [x, y, psi, vx, vy, omega, steer]^T
			u is a 2x1 vector: [pwm, change in steer]^T
			dxdt is a casadi.SX variable
		"""
		if self.input_acc:
			acc = u[0]
		else:
			pwm = u[0]
		dsteer = u[1]
		psi = x[2]
		vx = x[3]
		vy = x[4]
		omega = x[5]
		steer = x[6]

		dxdt[0] = vx*cs.cos(psi) - vy*cs.sin(psi)
		dxdt[1] = vx*cs.sin(psi) + vy*cs.cos(psi)
		dxdt[2] = omega
		dxdt[6] = dsteer
		if self.input_acc:
			dxdt[3] = acc
		else:
			dxdt[3] = 1/self.mass * (self.Cm1-self.Cm2*vx) * pwm
		dxdt[4] = (dxdt[6]*vx + steer*dxdt[3]) * self.lr / (self.lf+self.lr)
		dxdt[5] = (dxdt[6]*vx + steer*dxdt[3]) * (self.lf+self.lr)
		return dxdt		

	def calc_forces(self, x, u):
		if self.input_acc:
			acc = u[0]
			Frx = self.mass*acc
		else:
			pwm = u[0]
			vx = x[3]
			Frx = (self.Cm1-self.Cm2*vx)*pwm
		return Frx

	def sim_discrete(self, x0, u, Ts):
		"""	simulates a continuously linearized discrete model
			u is the input vector of size 2xn
			Ts is the sampling time
		"""
		n_steps = u.shape[1]
		x = np.zeros([self.n_states, n_steps+1])
		dxdt = np.zeros([self.n_states, n_steps+1])
		dxdt[:,0] = self._diffequation(None, x0, [0, 0])
		x[:,0] = x0
		for ids in range(1, n_steps+1):
			g = self._diffequation(None, x[:,ids-1], u[:,ids-1]).reshape(-1,)
			x[:,ids] = x[:,ids-1] + g*Ts
			dxdt[:,ids] = self._diffequation(None, x[:,ids], u[:,ids-1])
		return x, dxdt

	def linearize(self, x0, u0):
		"""	linearize at a given x0, u0
			for a given continuous system dxdt = f(x(t))
			calculate A = ∂f/∂x, B = ∂f/∂u, g = f evaluated at x0, u0
			A is 7x7, B is 7x2, g is 7x1
		"""
		dsteer = u0[1]
		psi = x0[2]
		vx = x0[3]
		vy = x0[4]
		omega = x0[5]
		steer = x[6]

		sinpsi = np.sin(psi)
		cospsi = np.cos(psi)

		if self.input_acc:
			raise NotImplementedError
		Frx = self.calc_forces(x0, u0)

		pwm = u0[0]
		dFrx_dvx = -self.Cm2*pwm
		dFrx_du1 = self.Cm1-self.Cm2*vx

		f1_psi = -vx*sinpsi-vy*cospsi
		f1_vx = cospsi
		f1_vy = -sinpsi

		f2_psi = vx*cospsi-vy*sinpsi
		f2_vx = sinpsi
		f2_vy = cospsi

		f3_omega = 1

		f4_vx = 1/self.mass * dFrx_dvx

		f5_vx = (dsteer + steer * 1/self.mass * dFrx_dvx) * self.lr / (self.lf+self.lr)
		f5_delta = 1/self.mass * Frx * self.lr / (self.lf+self.lr)

		f6_vx = (dsteer + steer * 1/self.mass * dFrx_dvx) / (self.lf+self.lr)
		f6_delta = 1/self.mass * Frx / (self.lf+self.lr)

		f4_u1 = dFrx_du1
		f5_u1 = steer * 1/self.mass * dFrx_du1 * self.lr / (self.lf+self.lr)
		f6_u1 = steer * 1/self.mass * dFrx_du1 / (self.lf+self.lr)

		f5_u2 = vx * self.lr / (self.lf+self.lr)
		f6_u2 = vx / (self.lf+self.lr)
		f7_u2 = 1

		A = np.array([
			[0, 0, f1_psi, f1_vx, f1_vy, 0, 0],
			[0, 0, f2_psi, f2_vx, f2_vy, 0, 0],
			[0, 0, 0, 0, 0, f3_omega, 0],
			[0, 0, 0, f4_vx, 0, 0, 0],
			[0, 0, 0, f5_vx, 0, 0, f5_delta],
			[0, 0, 0, f6_vx, 0, 0, f6_delta],
			[0, 0, 0, 0, 0, 0, 0],
			])
		B = np.array([
			[0, 0],
			[0, 0],
			[0, 0],
			[f4_u1, 0],
			[f5_u1, f5_u2],
			[f6_u1, f6_u2],
			[0, f7_u2],
			])
		g = self._diffequation(None, x0, u0).reshape(-1,)
		return A, B, g


if __name__ == '__main__':
	"""	test cases 1-3 use 4 states continuous model
		test cases 4-6 use 4 states discrete model
		test pairs (1,4), (2,5) and (3,6) should give same results
	"""

	# vehicle parameters for F1/10
	params = ORCA()
	model = Kinematic6(**params)

	test_case = 3

	Ts = 0.02

	#####################################################################
	# CONTINUOUS MODEL 6 STATES

	# start at origin with init velocity [3, 3] m/s
	# apply constant pwm for 1s and then move at constant speed
	if test_case == 1:
		n_steps = 100
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, n_steps*Ts, n_steps+1)
		inputs[0,:50] = 1
		x_init = np.array([0, 0, np.pi/4, 3*np.sqrt(2), 0, 0, 0])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad
	if test_case == 2:
		n_steps = 200
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, n_steps*Ts, n_steps+1)
		inputs[1,0] = 0.2/Ts
		x_init = np.array([0, 0, 0, 3, 0, 0, 0])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad after 2 sec
	if test_case == 3:
		n_steps = 400
		inputs = np.zeros([2, n_steps])
		inputs[1,int(n_steps/4)] = 0.2/Ts
		time = np.linspace(0, n_steps*Ts, n_steps+1)
		x_init = np.array([0, 0, 0, 3, 0, 0, 0])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	#####################################################################
	# DISCRETE MODEL 6 STATES
	
	# start at origin with init velocity [3, 3] m/s
	# apply constant acceleration 1 m/s^2 for 1s and then move at constant speed
	if test_case == 4:
		n_steps = int(2/Ts)
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, n_steps+1, n_steps+1)*Ts
		inputs[0,:int(n_steps/2)] = 1
		x_init = np.array([0, 0, np.pi/4, 3*np.sqrt(2), 0, 0, 0])
		x_disc, dxdt_disc = model.sim_discrete(x_init, inputs, Ts)
		model.plot_results(time, x_disc, dxdt_disc, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad
	if test_case == 5:
		n_steps = int(4/Ts)
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, n_steps+1, n_steps+1)*Ts
		inputs[1,0] = 0.2/Ts
		x_init = np.array([0, 0, 0, 3, 0, 0, 0])
		x_disc, dxdt_disc = model.sim_discrete(x_init, inputs, Ts)
		model.plot_results(time, x_disc, dxdt_disc, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad after 2 sec
	if test_case == 6:
		n_steps = int(8/Ts)
		inputs = np.zeros([2, n_steps])
		inputs[1,int(n_steps/4)] = 0.2/Ts
		time = np.linspace(0, n_steps+1, n_steps+1)*Ts
		x_init = np.array([0, 0, 0, 3, 0, 0, 0])
		x_disc, dxdt_disc = model.sim_discrete(x_init, inputs, Ts)
		model.plot_results(time, x_disc, dxdt_disc, inputs)