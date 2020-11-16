"""	Dynamic bicycle model.
	
	Use Dynamic class to:
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
from bayes_race.params import F110


class Dynamic(Model):

	def __init__(self, lf, lr, mass, Iz, Cf, Cr, 
				 Bf=None, Br=None, Df=None, Dr=None,
				 Cm1=None, Cm2=None, Cr0=None, Cr2=None, 
				 input_acc=False, **kwargs):
		"""	specify model params here
		"""
		self.lf = lf
		self.lr = lr
		self.dr = lr/(lf+lr)
		self.mass = mass
		self.Iz = Iz

		self.Cf = Cf
		self.Cr = Cr

		self.Bf = Bf
		self.Br = Br
		self.Df = Df
		self.Dr = Dr

		self.Cm1 = Cm1
		self.Cm2 = Cm2
		self.Cr0 = Cr0
		self.Cr2 = Cr2

		self.approx = False
		if Bf is None or Br is None or Df is None or Dr is None:
			self.approx = True
		self.input_acc = input_acc
		self.n_states = 6
		self.n_inputs = 2
		Model.__init__(self)

	def sim_continuous(self, x0, u, t):
		"""	simulates the nonlinear continuous model with given input vector
			by numerical integration using 6th order Runge Kutta method
			x0 is the initial state of size 6x1
			u is the input vector of size 2xn
			t is the time vector of size 1x(n+1)
		"""
		n_steps = u.shape[1]
		x = np.zeros([6, n_steps+1])
		dxdt = np.zeros([6, n_steps+1])
		dxdt[:,0] = self._diffequation(None, x0, [0, 0])
		x[:,0] = x0
		for ids in range(1, n_steps+1):
			x[:,ids] = self._integrate(x[:,ids-1], u[:,ids-1], t[ids-1], t[ids])
			dxdt[:,ids] = self._diffequation(None, x[:,ids], u[:,ids-1])
		return x, dxdt

	def _diffequation(self, t, x, u):
		"""	write dynamics as first order ODE: dxdt = f(x(t))
			x is a 6x1 vector: [x, y, psi, vx, vy, omega]^T
			u is a 2x1 vector: [acc/pwm, steer]^T
		"""
		steer = u[1]
		psi = x[2]
		vx = x[3]
		vy = x[4]
		omega = x[5]

		Ffy, Frx, Fry = self.calc_forces(x, u)

		dxdt = np.zeros(6)
		dxdt[0] = vx*np.cos(psi) - vy*np.sin(psi)
		dxdt[1] = vx*np.sin(psi) + vy*np.cos(psi)
		dxdt[2] = omega
		dxdt[3] = 1/self.mass * (Frx - Ffy*np.sin(steer)) + vy*omega
		dxdt[4] = 1/self.mass * (Fry + Ffy*np.cos(steer)) - vx*omega
		dxdt[5] = 1/self.Iz * (Ffy*self.lf*np.cos(steer) - Fry*self.lr)
		return dxdt

	def calc_forces(self, x, u, return_slip=False):
		steer = u[1]
		psi = x[2]
		vx = x[3]
		vy = x[4]
		omega = x[5]

		if self.approx:

			# rolling friction and drag are ignored
			acc = u[0]
			Frx = self.mass*acc

			# See Vehicle Dynamics and Control (Rajamani)
			alphaf = steer - (self.lf*omega + vy)/vx
			alphar = (self.lr*omega - vy)/vx
			Ffy = 2 * self.Cf * alphaf
			Fry = 2 * self.Cr * alphar

		else:
			
			if self.input_acc:
				# rolling friction and drag are ignored
				acc = u[0]
				Frx = self.mass*acc
			else:
				# rolling friction and drag are modeled
				pwm = u[0]
				Frx = (self.Cm1-self.Cm2*vx)*pwm - self.Cr0 - self.Cr2*(vx**2)

			alphaf = steer - np.arctan2((self.lf*omega + vy), abs(vx))
			alphar = np.arctan2((self.lr*omega - vy), abs(vx))
			Ffy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alphaf))
			Fry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alphar))
		if return_slip:
			return Ffy, Frx, Fry, alphaf, alphar
		else:
			return Ffy, Frx, Fry

	def casadi(self, x, u, dxdt):
		"""	write dynamics as first order ODE: dxdt = f(x(t))
			x is a 6x1 vector: [x, y, psi, vx, vy, omega]^T
			u is a 2x1 vector: [acc/pwm, steer]^T
			dxdt is a casadi.SX variable
		"""
		pwm = u[0]
		steer = u[1]
		psi = x[2]
		vx = x[3]
		vy = x[4]
		omega = x[5]

		vmin = 0.05
		vy = cs.if_else(vx<vmin, 0, vy)
		omega = cs.if_else(vx<vmin, 0, omega)
		steer = cs.if_else(vx<vmin, 0, steer)
		vx = cs.if_else(vx<vmin, vmin, vx)

		Frx = (self.Cm1-self.Cm2*vx)*pwm - self.Cr0 - self.Cr2*(vx**2)
		alphaf = steer - cs.atan2((self.lf*omega + vy), vx)
		alphar = cs.atan2((self.lr*omega - vy), vx)
		Ffy = self.Df * cs.sin(self.Cf * cs.arctan(self.Bf * alphaf))
		Fry = self.Dr * cs.sin(self.Cr * cs.arctan(self.Br * alphar))

		dxdt[0] = vx*cs.cos(psi) - vy*cs.sin(psi)
		dxdt[1] = vx*cs.sin(psi) + vy*cs.cos(psi)
		dxdt[2] = omega
		dxdt[3] = 1/self.mass * (Frx - Ffy*cs.sin(steer)) + vy*omega
		dxdt[4] = 1/self.mass * (Fry + Ffy*cs.cos(steer)) - vx*omega
		dxdt[5] = 1/self.Iz * (Ffy*self.lf*cs.cos(steer) - Fry*self.lr)
		return dxdt

	def sim_discrete(self, x0, u, Ts):
		"""	simulates a continuously linearized discrete model
			u is the input vector of size 2xn
			Ts is the sampling time
		"""
		n_steps = u.shape[1]
		x = np.zeros([6, n_steps+1])
		dxdt = np.zeros([6, n_steps+1])
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
			A is 6x6, B is 6x2, g is 6x1
		"""
		steer = u0[1]
		psi = x0[2]
		vx = x0[3]
		vy = x0[4]
		omega = x0[5]

		# numerical correction for low speeds
		vmin = 0.05
		if vx < vmin:
			vy = 0
			omega = 0
			steer = 0
			vx = vmin

		sindelta = np.sin(steer)
		cosdelta = np.cos(steer)
		sinpsi = np.sin(psi)
		cospsi = np.cos(psi)

		Ffy, Frx, Fry, alphaf, alphar = self.calc_forces(x0, u0, return_slip=True)

		if self.approx:

			dFfy_dvx = 2 * self.Cf * (self.lf*omega + vy)/((self.lf*omega + vy)**2 + vx**2)
			dFfy_dvy = -2 * self.Cf * vx/((self.lf*omega + vy)**2 + vx**2)
			dFfy_domega = -2 * self.Cf * self.lf * vx/((self.lf*omega + vy)**2 + vx**2)

			dFrx_dvx = 0
			dFrx_dvu1 = 1

			dFry_dvx = -2 * self.Cr * (self.lr*omega - vy)/((self.lr*omega - vy)**2 + vx**2)
			dFry_dvy = -2 * self.Cr * vx/((self.lr*omega - vy)**2 + vx**2)
			dFry_domega = 2 * self.Cr * self.lr * vx/((self.lr*omega - vy)**2 + vx**2)

			dFfy_delta = 2*self.Cf

		else:

			dFfy_dalphaf = self.Bf * self.Cf * self.Df * np.cos(self.Cf * np.arctan(self.Bf * alphaf))
			dFfy_dalphaf *= 1/(1+(self.Bf*alphaf)**2)

			dFry_dalphar = self.Br * self.Cr * self.Dr * np.cos(self.Cr * np.arctan(self.Br * alphar))
			dFry_dalphar *= 1/(1+(self.Br*alphar)**2)

			dFfy_dvx =  dFfy_dalphaf * (self.lf*omega + vy)/((self.lf*omega + vy)**2 + vx**2)
			dFfy_dvy = -dFfy_dalphaf * vx/((self.lf*omega + vy)**2 + vx**2)
			dFfy_domega = -dFfy_dalphaf * self.lf * vx/((self.lf*omega + vy)**2 + vx**2)

			if self.input_acc:
				raise NotImplementedError
				
			pwm = u0[0]
			dFrx_dvx = -self.Cm2*pwm - 2*self.Cr2*vx
			dFrx_dvu1 = self.Cm1-self.Cm2*vx

			dFry_dvx = -dFry_dalphar * (self.lr*omega - vy)/((self.lr*omega - vy)**2 + vx**2)
			dFry_dvy = -dFry_dalphar * vx/((self.lr*omega - vy)**2 + vx**2)
			dFry_domega = dFry_dalphar * self.lr * vx/((self.lr*omega - vy)**2 + vx**2)

			dFfy_delta = dFfy_dalphaf

		f1_psi = -vx*sinpsi-vy*cospsi
		f1_vx = cospsi
		f1_vy = -sinpsi

		f2_psi = vx*cospsi-vy*sinpsi
		f2_vx = sinpsi
		f2_vy = cospsi

		f4_vx = 1/self.mass * (dFrx_dvx -dFfy_dvx*sindelta)
		f4_vy = 1/self.mass * (-dFfy_dvy*sindelta + self.mass*omega)
		f4_omega = 1/self.mass * (-dFfy_domega*sindelta + self.mass*vy)

		f5_vx = 1/self.mass * (dFry_dvx + dFfy_dvx*cosdelta - self.mass*omega)
		f5_vy = 1/self.mass * (dFry_dvy + dFfy_dvy*cosdelta)
		f5_omega = 1/self.mass * (dFry_domega + dFfy_domega*cosdelta - self.mass*vx)

		f6_vx = 1/self.Iz * (dFfy_dvx*self.lf*cosdelta - dFry_dvx*self.lr)
		f6_vy = 1/self.Iz * (dFfy_dvy*self.lf*cosdelta - dFry_dvy*self.lr)
		f6_omega = 1/self.Iz * (dFfy_domega*self.lf*cosdelta - dFry_domega*self.lr)

		f4_u1 = dFrx_dvu1
		f4_delta = 1/self.mass * (-dFfy_delta*sindelta - Ffy*cosdelta)
		f5_delta = 1/self.mass * (dFfy_delta*cosdelta - Ffy*sindelta)
		f6_delta = 1/self.Iz * (dFfy_delta*self.lf*cosdelta - Ffy*self.lf*sindelta)

		A = np.array([
			[0, 0, f1_psi, f1_vx, f1_vy, 0],
			[0, 0, f2_psi, f2_vx, f2_vy, 0],
			[0, 0, 0, 0, 0, 1],
			[0, 0, 0, f4_vx, f4_vy, f4_omega],
			[0, 0, 0, f5_vx, f5_vy, f5_omega],
			[0, 0, 0, f6_vx, f6_vy, f6_omega],
			])
		B = np.array([
			[0, 0],
			[0, 0],
			[0, 0],
			[f4_u1, f4_delta],
			[0, f5_delta],
			[0, f6_delta],
			])
		g = self._diffequation(None, x0, u0).reshape(-1,)
		return A, B, g


if __name__ == '__main__':
	"""	test cases 1-3 use 4 states continuous model
		test cases 4-6 use 4 states discrete model
		test pairs (1,4), (2,5) and (3,6) should give same results
	"""

	# vehicle parameters for F1/10
	params = F110()
	model = Dynamic(**params)

	test_case = 3

	#####################################################################
	# CONTINUOUS MODEL 6 STATES

	# start at origin with init velocity [3, 3] m/s
	# apply constant acceleration 1 m/s^2 for 1s and then move at constant speed
	if test_case == 1:
		n_steps = 100
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, 2, n_steps+1)
		inputs[0,:50] = 1
		x_init = np.array([0, 0, np.pi/4, 3*np.sqrt(2), 0, 0])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad
	if test_case == 2:
		n_steps = 200
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, 4, n_steps+1)
		inputs[1,:] = 0.2
		x_init = np.array([0, 0, 0, 3, 0, 0])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad after 2 sec
	if test_case == 3:
		n_steps = 400
		inputs = np.zeros([2, n_steps])
		inputs[1,100:] = 0.2
		time = np.linspace(0, 8, n_steps+1)
		x_init = np.array([0, 0, 0, 3, 0, 0])
		x_cont, dxdt_cont = model.sim_continuous(x_init, inputs, time)
		model.plot_results(time, x_cont, dxdt_cont, inputs)

	#####################################################################
	# DISCRETE MODEL 6 STATES
	
	# start at origin with init velocity [3, 3] m/s
	# apply constant acceleration 1 m/s^2 for 1s and then move at constant speed
	if test_case == 4:
		Ts = 0.02
		n_steps = int(2/Ts)
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, n_steps+1, n_steps+1)*Ts
		inputs[0,:int(n_steps/2)] = 1
		x_init = np.array([0, 0, np.pi/4, 3*np.sqrt(2), 0, 0])
		x_disc, dxdt_disc = model.sim_discrete(x_init, inputs, Ts)
		model.plot_results(time, x_disc, dxdt_disc, inputs)

	# start at origin with init velocity [3, 0] m/s
	# steer at constant angle 0.2 rad
	if test_case == 5:
		Ts = 0.02
		n_steps = int(4/Ts)
		inputs = np.zeros([2, n_steps])
		time = np.linspace(0, n_steps+1, n_steps+1)*Ts
		inputs[1,:] = 0.2
		x_init = np.array([0, 0, 0, 3, 0, 0])
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
		x_init = np.array([0, 0, 0, 3, 0, 0])
		x_disc, dxdt_disc = model.sim_discrete(x_init, inputs, Ts)
		model.plot_results(time, x_disc, dxdt_disc, inputs)