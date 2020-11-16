""" Params for F1/10 1:10 scale car from University of Pennsylvania
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


def F110():

	lf = 0.15875			# front tyres from center of gravity [m]
	lr = 0.17145			# read tyres from center of gravity [m]
	mass = 3.74 			# vehicle mass [kg]
	Iz = 0.04712 			# moment of inertia [kgm^2]
	Cf = 2.3 				# front cornering stiffness
	Cr = 2.3 				# rear cornering stiffness
	Csf = 4.718
	Csr = 5.4562
	hcog = 0.074
	mu = 0.523
	min_v = 0.
	max_v = 20.
	switch_v = 1.
	max_acc = 9.51 			# max acceleration [m/s^2]
	min_acc = -13.26 		# max deceleration [m/s^2]
	max_steer = 0.4189 		# max steering angle [rad]
	min_steer = -0.4189 	# min steering angle [rad]
	max_steer_vel = 3.2 	# max steering velocity [rad/s]

	max_inputs = [max_acc, max_steer]
	min_inputs = [min_acc, min_steer]

	max_rates = [None, max_steer_vel]
	min_rates = [None, -max_steer_vel]	

	params = {
		'lf': lf,
		'lr': lr,
		'mass': mass,
		'Iz': Iz,
		'Cf': Cf,
		'Cr': Cr,
		'Csf': Csf,
		'Csr': Csr,
		'hcog': hcog,
		'mu': mu,
		'min_v': min_v,
		'max_v': max_v,
		'switch_v': switch_v,
		'max_acc': max_acc,
		'min_acc': min_acc,
		'max_steer': max_steer,
		'min_steer': min_steer,
		'max_steer_vel': max_steer_vel,
		'max_inputs': max_inputs,
		'min_inputs': min_inputs,
		'max_rates': max_rates,
		'min_rates': min_rates,
		}
	return params