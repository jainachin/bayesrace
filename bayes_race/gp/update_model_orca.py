"""	Update a GP model after collecting new data.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from bayes_race.utils.plots import plot_true_predicted_variance

#####################################################################
# load data

SAVE_MODELS = False

N_SAMPLES = 400
VARIDX = 3
names = ['x', 'y', 'yaw', 'vx', 'vy', 'omega']
LAPS_DONE = 1

fileload = 'orca/{}gp.pickle'.format(names[VARIDX])

def load_data(CTYPE, TRACK_NAME, VARIDX, xscaler=None, yscaler=None):

	data_dyn = np.load('../data/DYN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
	data_kin = np.load('../data/KIN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
	y_all = data_dyn['states'][:6,1:N_SAMPLES+1] - data_kin['states'][:6,1:N_SAMPLES+1]
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

x_train, y_train, xscaler, yscaler = load_data('PP', 'ETHZMobil', VARIDX)
x_train_mpc, y_train_mpc = load_data('GPMPC', 'ETHZ', VARIDX, xscaler=xscaler, yscaler=yscaler)

update_method = 2

# variance based selection
if update_method == 1:
	filesave = 'orca/{}gp{}-var.pickle'.format(names[VARIDX], LAPS_DONE)
	data = np.load('../data/SIGMA-{}-{}.npz'.format('GPMPC', 'ETHZ'))
	std = data['std']
	update_indices = std>0.15
	x_train = np.concatenate([x_train, x_train_mpc[update_indices,:]])
	y_train = np.concatenate([y_train, y_train_mpc[update_indices,:]])

# use all data to update
else:
	filesave = 'orca/{}gp{}.pickle'.format(names[VARIDX], LAPS_DONE)
	# adding first few observations to omega, increases error at low speed
	if VARIDX==5:
		x_train = np.concatenate([x_train, x_train_mpc[20:,:]])
		y_train = np.concatenate([y_train, y_train_mpc[20:,:]])
	else:
		x_train = np.concatenate([x_train, x_train_mpc])
		y_train = np.concatenate([y_train, y_train_mpc])

#####################################################################
# train GP model

k1 = 1.0*RBF(
	length_scale=np.ones(x_train.shape[1]),
	length_scale_bounds=(1e-5, 1e5),
	)
k2 = ConstantKernel(0.1)
kernel = k1 + k2
model = GaussianProcessRegressor(
	alpha=1e-6, 
	kernel=kernel, 
	normalize_y=True,
	n_restarts_optimizer=10,
	)
start = time.time()
model.fit(x_train, y_train)
end = time.time()
print('training time: %ss' %(end - start))        
print('final kernel: %s' %(model.kernel_))

if SAVE_MODELS:
	with open(filesave, 'wb') as f:
		pickle.dump((model, xscaler, yscaler), f)

#####################################################################
# test GP model on training data

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
	ylabel='{} '.format(names[VARIDX]), xlabel='sample index'
	)

plot_true_predicted_variance(
	y_test, y_test_mu, y_test_std, 
	ylabel='{} '.format(names[VARIDX]), xlabel='sample index'
	)

plt.show()