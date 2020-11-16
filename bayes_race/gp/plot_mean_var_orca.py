"""	Validate a trained GP model in Casadi.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time
import numpy as np
import _pickle as pickle
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

#####################################################################
# load data

SAVE_RESULTS = False
N_SAMPLES = 400

CTYPE = 'NMPC'

TRACK_NAME = 'ETHZ'

LAPS_DONE = 0
if LAPS_DONE == 0:
	LAPS_DONE = ''
	SUFFIX = ''
else:
	SUFFIX = '_lap{}'.format(LAPS_DONE)

names = ['x', 'y', 'psi', 'vx', 'vy', 'omega']
names_latex = ['x', 'y', '\psi', 'v_x', 'v_y', '\omega']

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

#####################################################################
# test GP model on validation data

y_test_std_max = np.zeros([N_SAMPLES])

plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(3,1)

for VARIDX in [3,4,5]:

	# load GP model
	filename = 'orca/{}gp{}.pickle'.format(names[VARIDX], LAPS_DONE)
	with open(filename, 'rb') as f:
		(model, xscaler, yscaler) = pickle.load(f)  

	# load data
	x_test, y_test = load_data(CTYPE, TRACK_NAME, VARIDX, xscaler=xscaler, yscaler=yscaler)

	# predict
	y_test_mu, y_test_std = model.predict(x_test, return_std=True)
	y_test = yscaler.inverse_transform(y_test)
	y_test_mu = yscaler.inverse_transform(y_test_mu)
	y_test_std *= yscaler.scale_
	y_test_std_max = np.max([y_test_std_max,y_test_std], axis=0)

	# error stats
	MSE = mean_squared_error(y_test, y_test_mu, multioutput='raw_values')
	R2Score = r2_score(y_test, y_test_mu, multioutput='raw_values')
	EV = explained_variance_score(y_test, y_test_mu, multioutput='raw_values')
	print('root mean square error: %s' %(np.sqrt(MSE)))
	print('normalized mean square error: %s' %(np.sqrt(MSE)/np.array(np.abs(y_test.mean()))))
	print('R2 score: %s' %(R2Score))
	print('explained variance: %s' %(EV))

	# plot results
	y_true = y_test.flatten()
	y_mu = y_test_mu.flatten()
	y_std = y_test_std.flatten()

	l = y_true.shape[0]
	x = range(l)

	# mean variance
	plt.subplot(gs[VARIDX-3,:])
	ax = plt.gca()
	if VARIDX==3:
		label1 = '$\mu_{v_x}$'
		label2 = '$\pm 2 \sigma_{v_x}$'
		ylabel = '$\mathbf{e}_4$ (error in $v_x$)'
		loc = 'lower left'
		ax.set_xticks([])
		ylim = [-0.2, 0]
	elif VARIDX==4:
		label1 = '$\mu_{v_y}$'
		label2 = '$\pm 2 \sigma_{v_y}$'
		ylabel = '$\mathbf{e}_5$ (error in $v_y$)'
		loc = 'upper left'

		ax.set_xticks([])
		ylim = [-0.27, 0.37]
	elif VARIDX==5:
		label1 = '$\mu_{\omega}$'
		label2 = '$\pm 2 \sigma_{\omega}$'
		ylabel = '$\mathbf{e}_6$ (error in $\omega$)'
		loc = 'upper left'
		ylim = [-3, 3.5]
	plt.plot(x, y_mu, 'r', ls='-', lw=1, zorder=9, label=label1)
	plt.fill_between(x, (y_mu+2*y_std), (y_mu-2*y_std), alpha=0.2, color='r', label=label2)
	plt.plot(x, y_true, '--k', lw=1, zorder=9, label='true')
	plt.ylabel(ylabel)
	plt.xlim([0, N_SAMPLES])
	plt.ylim(ylim)
	plt.legend(loc=loc, ncol=3)

plt.xlabel('k (sample index)')
plt.tight_layout()

filepath = 'error_validation{}.png'.format(SUFFIX)
if SAVE_RESULTS:
	plt.savefig(filepath, dpi=600, bbox_inches='tight')

plt.show()