""" Params for ORCA 1:43 scale car from ETH Zurich
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


def ORCA(control='pwm'):
	"""	choose "pwm" for Dynamic and "acc" for Kinematic model
		see bayes_race.models for details
	"""

	lf = 0.029
	lr = 0.033
	mass = 0.041
	Iz = 27.8e-6
	Bf = 2.579
	Cf = 1.2
	Df = 0.192
	Br = 3.3852
	Cr = 1.2691
	Dr = 0.1737

	Cm1 = 0.287
	Cm2 = 0.0545
	Cr0 = 0.0518
	Cr2 = 0.00035

	max_acc = 5. 			# max acceleration [m/s^2]
	min_acc = -5. 			# max deceleration [m/s^2]
	max_pwm = 1. 			# max PWM duty cycle
	min_pwm = -0.1 			# min PWM duty cycle
	max_steer = 0.35 		# max steering angle [rad]
	min_steer = -0.35 		# min steering angle [rad]
	max_steer_vel = 5. 		# max steering velocity [rad/s]

	if control is 'pwm':
		max_inputs = [max_pwm, max_steer]
		min_inputs = [min_pwm, min_steer]

		max_rates = [None, max_steer_vel]
		min_rates = [None, -max_steer_vel]

	elif control is 'acc':
		max_inputs = [max_acc, max_steer]
		min_inputs = [min_acc, min_steer]

		max_rates = [None, max_steer_vel]
		min_rates = [None, -max_steer_vel]

	else:
		raise NotImplementedError(
			'choose control as "pwm" for Dynamic model \
			and "acc" for Kinematic model'
			)

	params = {
		'lf': lf,
		'lr': lr,
		'mass': mass,
		'Iz': Iz,
		'Bf': Bf,
		'Br': Br,
		'Cf': Cf,
		'Cr': Cr,
		'Df': Df,
		'Dr': Dr,
		'Cm1': Cm1,
		'Cm2': Cm2,
		'Cr0': Cr0,
		'Cr2': Cr2,
		'max_acc': max_acc,
		'min_acc': min_acc,		
		'max_pwm': max_pwm,
		'min_pwm': min_pwm,
		'max_steer': max_steer,
		'min_steer': min_steer,
		'max_steer_vel': max_steer_vel,
		'max_inputs': max_inputs,
		'min_inputs': min_inputs,
		'max_rates': max_rates,
		'min_rates': min_rates,
		}
	return params