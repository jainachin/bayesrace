"""	Load data for training and testing GP models.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from sklearn.preprocessing import StandardScaler


def loadSingleTrack(data_dyn, data_kin, varidx, n_samples):

	y_all = data_dyn['states'][:6,1:n_samples+1] - data_kin['states'][:6,1:n_samples+1]
	x = np.concatenate([
		data_kin['inputs'][:,:n_samples].T,
		data_kin['states'][6,:n_samples].reshape(1,-1).T,
		data_dyn['states'][3:6,:n_samples].T],
		axis=1)
	y = y_all[varidx].reshape(-1,1)
	return x, y

def loadData(ctype, track_names, varidx, n_samples, xscaler=None, yscaler=None):

	track = track_names[0]
	data_dyn = np.load('../data/DYN-{}-{}.npz'.format(ctype, track))
	data_kin = np.load('../data/KIN-{}-{}.npz'.format(ctype, track))
	x, y = loadSingleTrack(data_dyn, data_kin, varidx, n_samples)

	for idt in range(1,len(track_names)):
		track = track_names[idt]
		data_dyn = np.load('../data/DYN-{}-{}.npz'.format(ctype, track))
		data_kin = np.load('../data/KIN-{}-{}.npz'.format(ctype, track))
		x_, y_ = loadSingleTrack(data_dyn, data_kin, varidx, n_samples)

		x = np.concatenate([x, x_])
		y = np.concatenate([y, y_])

	if xscaler is None or yscaler is None:
		xscaler = StandardScaler()
		yscaler = StandardScaler()
		xscaler.fit(x)
		yscaler.fit(y)
		return xscaler.transform(x), yscaler.transform(y), xscaler, yscaler
	else:
		return xscaler.transform(x), yscaler.transform(y)