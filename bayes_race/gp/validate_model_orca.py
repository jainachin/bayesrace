"""	Validate a trained GP model in Casadi.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time
import numpy as np
import casadi as cs
import _pickle as pickle
from scipy.linalg import solve_triangular

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

import matplotlib.pyplot as plt
from bayes_race.utils.plots import plot_true_predicted_variance

#####################################################################
# load data

N_SAMPLES = 400
VARIDX = 3
state_names = ['x', 'y', 'yaw', 'vx', 'vy', 'omega']
filename = 'orca/{}gp.pickle'.format(state_names[VARIDX])

def load_data(CTYPE, TRACK_NAME, VARIDX, xscaler=None, yscaler=None):

	data_dyn = np.load('../data/DYN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
	data_kin = np.load('../data/KIN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
	y_all = data_dyn['states'][:,1:N_SAMPLES+1] - data_kin['states'][:6,1:N_SAMPLES+1]
	x = np.concatenate([
		data_kin['inputs'][:,:N_SAMPLES].T,
		data_kin['states'][6,:N_SAMPLES].reshape(1,-1).T,
		data_dyn['states'][3:6,:N_SAMPLES].T],
		axis=1)
	y = y_all[VARIDX].reshape(-1,1)

	if xscaler is None or yscaler is None:
		xscaler = StandardScaler()
		yscaler = StandardScaler()
		xscaler.fit(x)
		yscaler.fit(y)
		return xscaler.transform(x), yscaler.transform(y), xscaler, yscaler
	else:
		return xscaler.transform(x), yscaler.transform(y)

#####################################################################
# load GP model
print('###########################################################')

with open(filename, 'rb') as f:
	(model, xscaler, yscaler) = pickle.load(f)     
print('final kernel: %s' %(model.kernel_))

#####################################################################
# test GP model on training data
print('###########################################################')

N_SAMPLES = 400
x_train, y_train = load_data('PP', 'ETHZMobil', VARIDX, xscaler=xscaler, yscaler=yscaler)
y_train_mu, y_train_std = model.predict(x_train, return_std=True)
y_train = yscaler.inverse_transform(y_train)
y_train_mu = yscaler.inverse_transform(y_train_mu)
y_train_std *= yscaler.scale_

MSE = mean_squared_error(y_train, y_train_mu, multioutput='raw_values')
R2Score = r2_score(y_train, y_train_mu, multioutput='raw_values')
EV = explained_variance_score(y_train, y_train_mu, multioutput='raw_values')

print('root mean square error: %s' %(np.sqrt(MSE)))
print('normalized mean square error: %s' %(np.sqrt(MSE)/np.array(np.abs(y_train.mean()))))
print('R2 score: %s' %(R2Score))
print('explained variance: %s' %(EV))

#####################################################################
# test GP model on validation data
print('###########################################################')

N_SAMPLES = 400
x_test, y_test = load_data('NMPC', 'ETHZ', VARIDX, xscaler=xscaler, yscaler=yscaler)
y_test_mu, y_test_std = model.predict(x_test, return_std=True)
y_test = yscaler.inverse_transform(y_test)
y_test_mu = yscaler.inverse_transform(y_test_mu)
y_test_std *= yscaler.scale_

MSE = mean_squared_error(y_test, y_test_mu, multioutput='raw_values')
R2Score = r2_score(y_test, y_test_mu, multioutput='raw_values')
EV = explained_variance_score(y_test, y_test_mu, multioutput='raw_values')

print('root mean square error: %s' %(np.sqrt(MSE)))
print('normalized mean square error: %s' %(np.sqrt(MSE)/np.array(np.abs(y_test.mean()))))
print('R2 score: %s' %(R2Score))
print('explained variance: %s' %(EV))

#####################################################################
# plot results

plot_true_predicted_variance(
	y_train, y_train_mu, y_train_std, 
	ylabel='{} '.format(state_names[VARIDX]), xlabel='sample index'
	)

plot_true_predicted_variance(
	y_test, y_test_mu, y_test_std, 
	ylabel='{} '.format(state_names[VARIDX]), xlabel='sample index'
	)

plt.show()