"""	Base model class.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from bayes_race.utils.rk6 import odeintRK6
import matplotlib.pyplot as plt


class Model:

	def __init__(self):
		pass

	def _integrate(self, x_t, u_t, t_start, t_end):
		"""	integrates using an RK6 ODE solver
			returns x_t at t_end = ∫ dxdt dt given x_t at t_start
			x_t is either a 4x1 vector: [x, y, psi, v]^T
					   or a 6x1 vector: [x, y, psi, dxdt, dydt, dpsidt]^T
		"""
		fun=self._diffequation
		odesol = odeintRK6(
			fun=fun, 
			y0=x_t, 
			t=[t_start, t_end], 
			args=(u_t,))
		return odesol[-1,:]

	def plot_results(self, t, x, dxdt, u, friction_circle=False):
		"""	plot states and inputs
		"""
		# plot position
		plt.figure()
		plt.plot(x[0,:], x[1,:])
		plt.xlabel('x [m]')
		plt.ylabel('y [m]')
		plt.axis('equal')
		plt.grid(True)

		plt.figure()
		plt.plot(t, x[0,:], label='x')
		plt.plot(t, x[1,:], label='y')
		plt.xlabel('time [s]')
		plt.ylabel('position [m]')
		plt.grid(True)
		plt.legend()

		# plot velocity
		if dxdt is not None:
			plt.figure()
			plt.plot(t, dxdt[0,:], label='speed x')
			plt.plot(t, dxdt[1,:], label='speed y')
			if not friction_circle:
				plt.plot(t, dxdt[2,:], label='yaw rate')
			plt.plot(t, np.sqrt(dxdt[0,:]**2+dxdt[1,:]**2), '--', label='speed abs')
			plt.xlabel('time [s]')
			plt.ylabel('velocity [m/s]')
			plt.grid(True)
			plt.legend()

			plt.figure()
			plt.plot(dxdt[0,:], dxdt[1,:])
			plt.xlabel('speed x [m/s]')
			plt.ylabel('speed y [m/s]')
			plt.axis('equal')
			plt.grid(True)

		# plot inertial heading
		plt.figure()
		if friction_circle:
			plt.plot(t, np.arctan2(dxdt[1,:],dxdt[0,:]))
		else:
			plt.plot(t, x[2,:])
		plt.ylabel('yaw (heading) [rad]')
		plt.xlabel('time [s]')
		plt.grid(True)

		# plot inputs
		plt.figure()
		if friction_circle:
			plt.plot(t[1:], u[0,:], label='force x')
			plt.plot(t[1:], u[1,:], label='force y')
		else:
			plt.plot(t[1:], u[0,:], label='acceleration')
			plt.plot(t[1:], u[1,:], label='steering')
		plt.ylabel('inputs')
		plt.grid(True)
		plt.legend()
		plt.show()