"""	Validate a trained GP model for error discrepancy between kinematic and dynamic models.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from bayes_race.gp.load_data import loadData
from bayes_race.utils.plots import plot_true_predicted_variance

#####################################################################
# load data

N_SAMPLES = 400
VARIDX = 3
state_names = ['x', 'y', 'yaw', 'vx', 'vy', 'omega']
filename = 'f110/{}gp.pickle'.format(state_names[VARIDX])

#####################################################################
# load GP model

with open(filename, 'rb') as f:
	(model, xscaler, yscaler) = pickle.load(f)     
print('final kernel: %s' %(model.kernel_))

#####################################################################
# test GP model on training data

x_train, y_train, xscaler, yscaler = loadData('PP', ['MAP2'], VARIDX, N_SAMPLES)

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

N_SAMPLES = 325

x_test, y_test = loadData('PP', ['MAP3', 'MAP8', 'MAP16'], VARIDX, N_SAMPLES, xscaler=xscaler, yscaler=yscaler)

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