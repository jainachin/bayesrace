""" Compute racing line using Bayesian Optimization (BayesOpt).
    This script compares EI, noisyEI and random strategies for sampling.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time
import numpy as np

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model

from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

from bayes_race.tracks import ETHZMobil
from bayes_race.params import ORCA
from bayes_race.raceline import randomTrajectory
from bayes_race.raceline import calcMinimumTime

from matplotlib import pyplot as plt

#####################################################################
# set device in torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.float

#####################################################################
# simulation settings

SEED = np.random.randint(1000)
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 1              # useful for parallelization, DON'T change
N_TRIALS = 3                # number of times bayesopt is run
N_BATCH = 100               # new observations after initialization
MC_SAMPLES = 64             # monte carlo samples
N_INITIAL_SAMPLES = 10      # samples to initialize GP
PLOT_RESULTS = False        # whether to plot results
SAVE_RESULTS = True         # whether to save results
N_WAYPOINTS = 100           # resampled waypoints
SCALE = 0.95                # shrinking factor for track width
LASTIDX = 0                 # fixed node at the end DO NOT CHANGE

# define indices for the nodes
NODES = [7, 21, 37, 52, 66, 81, 97, 111, 136, 160, 175, 191, 205, 220, 236, 250, 275, 299, 337, 376]

#####################################################################
# track specific data

params = ORCA()
track = ETHZMobil()

track_width = track.track_width*SCALE
theta = track.theta_track[NODES]
N_DIMS = len(NODES)
n_waypoints = N_DIMS

rand_traj = randomTrajectory(track=track, n_waypoints=n_waypoints)

bounds = torch.tensor([[-track_width/2] * N_DIMS, [track_width/2] * N_DIMS], device=device, dtype=dtype)

def evaluate_y(x_eval, mean_y=None, std_y=None):
    """ evaluate true output for given x (distance of nodes from center line)
        TODO: parallelize evaluations
    """
    if type(x_eval) is torch.Tensor:
        is_tensor = True
        x_eval = x_eval.cpu().numpy()
    else:
        is_tensor = False

    if len(x_eval.shape)==1:
        x_eval = x_eval.reshape(1,-1)
    n_eval = x_eval.shape[0]

    y_eval = np.zeros(n_eval)
    for ids in range(n_eval):
        wx, wy = rand_traj.calculate_xy(
            width=x_eval[ids],
            last_index=NODES[LASTIDX],
            theta=theta,
            )
        x, y = rand_traj.fit_cubic_splines(
            wx=wx, 
            wy=wy, 
            n_samples=N_WAYPOINTS,
            )
        y_eval[ids] = -calcMinimumTime(x, y, **params)       # we want to max negative lap times

    if mean_y and std_y:
        y_eval = normalize(y_eval, mean_y, std_y)

    if is_tensor:
        return torch.tensor(y_eval, device=device, dtype=dtype).unsqueeze(-1)
    else:
        return y_eval.ravel()

def generate_initial_data(n_samples=10):
    """ generate training data
    """
    train_x = np.zeros([n_samples, n_waypoints])
    train_y_ = np.zeros([n_samples, 1])

    for ids in range(n_samples):
        width_random = rand_traj.sample_nodes(scale=SCALE)
        t_random = evaluate_y(width_random)
        train_x[ids,:] = width_random
        train_y_[ids,:] = t_random

    mean_y, std_y = train_y_.mean(), train_y_.std()
    train_y = normalize(train_y_, mean_y, std_y)
    train_x = torch.tensor(train_x, device=device, dtype=dtype)
    train_y = torch.tensor(train_y, device=device, dtype=dtype)
    best_y = train_y.max().item()
    return train_x, train_y, best_y, mean_y, std_y

def normalize(y_eval, mean_y, std_y):
    """ normalize outputs for GP
    """
    return (y_eval - mean_y) / std_y

#####################################################################
# modeling and optimization functions called in closed-loop

def initialize_model(train_x, train_y, state_dict=None):
    """initialize GP model with/without initial states
    """
    model = SingleTaskGP(train_x, train_y).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def optimize_acqf_and_get_observation(acq_func, mean_y=None, std_y=None):
    """optimize acquisition function and evaluate new candidates
    """
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
    )

    # observe new values 
    new_x = candidates.detach()
    new_y = evaluate_y(new_x, mean_y=mean_y, std_y=std_y)
    return new_x, new_y

def sample_random_observations(mean_y, std_y):
    """sample a random trajectory
    """
    rand_x = torch.tensor(rand_traj.sample_nodes(scale=SCALE).reshape(1,-1), device=device, dtype=dtype)
    rand_y = evaluate_y(rand_x, mean_y=mean_y, std_y=std_y)  
    return rand_x, rand_y

#####################################################################
# main simulation loop

# define the qEI and qNEI acquisition modules using a QMC sampler
qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

def optimize():

    verbose = True
    
    best_observed_all_ei, best_observed_all_nei, best_random_all = [], [], []
    train_x_all_ei, train_x_all_nei, train_x_all_random = [], [], []
    train_y_all_ei, train_y_all_nei, train_y_all_random = [], [], []

    # statistics over multiple trials
    for trial in range(1, N_TRIALS + 1):

        print('\nTrial {} of {}'.format(trial, N_TRIALS))
        best_observed_ei, best_observed_nei = [], []
        best_random = []
        
        # generate initial training data and initialize model
        print('\nGenerating {} random samples'.format(N_INITIAL_SAMPLES))
        train_x_ei, train_y_ei, best_y_ei, mean_y, std_y = generate_initial_data(n_samples=N_INITIAL_SAMPLES)
        denormalize = lambda x: -(x*std_y + mean_y)
        mll_ei, model_ei = initialize_model(train_x_ei, train_y_ei)
        
        train_x_nei, train_y_nei, best_y_nei = train_x_ei, train_y_ei, best_y_ei
        mll_nei, model_nei = initialize_model(train_x_nei, train_y_nei)

        train_x_random, train_y_random, best_y_random = train_x_ei, train_y_ei, best_y_ei

        best_observed_ei.append(denormalize(best_y_ei))
        best_observed_nei.append(denormalize(best_y_nei))
        best_random.append(denormalize(best_y_random))

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):    
            
            print('\nBatch {} of {}\n'.format(iteration, N_BATCH))
            t0 = time.time()
            
            # fit the models
            fit_gpytorch_model(mll_ei)
            fit_gpytorch_model(mll_nei)
            
            # update acquisition functions
            qEI = qExpectedImprovement(
                model=model_ei, 
                best_f=train_y_ei.max(),
                sampler=qmc_sampler,
            )
            
            qNEI = qNoisyExpectedImprovement(
                model=model_nei, 
                X_baseline=train_x_nei,
                sampler=qmc_sampler,
            )
            
            # optimize acquisition function and evaluate new sample
            new_x_ei, new_y_ei = optimize_acqf_and_get_observation(qEI, mean_y=mean_y, std_y=std_y)
            print('EI: time to traverse is {:.4f}s'.format(-(new_y_ei.numpy().ravel()[0]*std_y+mean_y)))
            new_x_nei, new_y_nei = optimize_acqf_and_get_observation(qNEI, mean_y=mean_y, std_y=std_y)
            print('NEI: time to traverse is {:.4f}s'.format(-(new_y_nei.numpy().ravel()[0]*std_y+mean_y)))
            new_x_random, new_y_random = sample_random_observations(mean_y=mean_y, std_y=std_y)
            print('Random: time to traverse is {:.4f}s'.format(-(new_y_random.numpy().ravel()[0]*std_y+mean_y)))

            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_y_ei = torch.cat([train_y_ei, new_y_ei])

            train_x_nei = torch.cat([train_x_nei, new_x_nei])
            train_y_nei = torch.cat([train_y_nei, new_y_nei])

            train_x_random = torch.cat([train_x_random, new_x_random])
            train_y_random = torch.cat([train_y_random, new_y_random])

            # update progress
            best_value_ei = denormalize(train_y_ei.max().item())
            best_value_nei = denormalize(train_y_nei.max().item())
            best_value_random = denormalize(train_y_random.max().item())
            
            best_observed_ei.append(best_value_ei)
            best_observed_nei.append(best_value_nei)
            best_random.append(best_value_random)

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll_ei, model_ei = initialize_model(
                train_x_ei,
                train_y_ei,
                model_ei.state_dict(),
            )
            mll_nei, model_nei = initialize_model(
                train_x_nei,
                train_y_nei, 
                model_nei.state_dict(),
            )
            t1 = time.time()

            if verbose:
                print(
                    'best lap time (random, qEI, qNEI) = {:.2f}, {:.2f}, {:.2f}, time to compute = {:.2f}s'.format(
                        best_value_random, 
                        best_value_ei, 
                        best_value_nei,
                        t1-t0
                        )
                )
            else:
                print(".")
       
        best_observed_all_ei.append(best_observed_ei)
        best_observed_all_nei.append(best_observed_nei)
        best_random_all.append(best_random)

        train_x_all_ei.append(train_x_ei.cpu().numpy())
        train_x_all_nei.append(train_x_nei.cpu().numpy())
        train_x_all_random.append(train_x_random.cpu().numpy())

        train_y_all_ei.append(denormalize(train_y_ei.cpu().numpy()))
        train_y_all_nei.append(denormalize(train_y_nei.cpu().numpy()))
        train_y_all_random.append(denormalize(train_y_random.cpu().numpy()))

    iters = np.arange(N_BATCH + 1) * BATCH_SIZE
    y_ei = np.asarray(best_observed_all_ei)
    y_nei = np.asarray(best_observed_all_nei)
    y_rnd = np.asarray(best_random_all)
    savestr = time.strftime('%Y%m%d%H%M%S')

    #####################################################################
    # save results

    if SAVE_RESULTS:
        
        np.savez(
            'results/{}_raceline_data-{}.npz'.format('ETHZMobil', savestr),
            y_ei=y_ei,
            y_nei=y_nei,
            y_rnd=y_rnd,
            iters=iters,
            train_x_all_ei=np.asarray(train_x_all_ei),
            train_x_all_nei=np.asarray(train_x_all_nei),
            train_x_all_random=np.asarray(train_x_all_random),
            train_y_all_ei=np.asarray(train_y_all_ei),
            train_y_all_nei=np.asarray(train_y_all_nei),
            train_y_all_random=np.asarray(train_y_all_random),
            SEED=SEED,
            )

    #####################################################################
    # plot results

    if PLOT_RESULTS:

        def ci(y):
            return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

        plt.figure()
        plt.gca().set_prop_cycle(None)
        plt.plot(iters, y_rnd.mean(axis=0), linewidth=1.5)
        plt.plot(iters, y_ei.mean(axis=0), linewidth=1.5)
        plt.plot(iters, y_nei.mean(axis=0), linewidth=1.5)
        plt.gca().set_prop_cycle(None)
        plt.fill_between(iters, y_rnd.mean(axis=0)-ci(y_rnd), y_rnd.mean(axis=0)+ci(y_rnd), label='random', alpha=0.2)
        plt.fill_between(iters, y_ei.mean(axis=0)-ci(y_ei), y_ei.mean(axis=0)+ci(y_ei), label='qEI', alpha=0.2)
        plt.fill_between(iters, y_nei.mean(axis=0)-ci(y_nei), y_nei.mean(axis=0)+ci(y_nei), label='qNEI', alpha=0.2)
        plt.xlabel('number of observations (beyond initial points)')
        plt.ylabel('best lap times')
        plt.grid(True)
        plt.legend(loc=0)
        plt.savefig('results/{}_laptimes-{}.png'.format('ETHZMobil', savestr), dpi=600)
        plt.show()


if __name__ == '__main__':
    
    optimize()

