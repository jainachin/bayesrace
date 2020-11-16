# BayesRace

BayesRace is a model-based planning and control framework for autonomous racing. It focuses on two problems:

1. Computing the racing line using Bayesian optimization [PDF](https://arxiv.org/pdf/2002.04794.pdf)
```
@InProceedings{JainRaceOpt2020,
  author    = {Jain, Achin and Morari, Manfred},
  title     = {{Computing the racing line using Bayesian optimization}},  
  booktitle = {Proceedings of the 59th IEEE Conference on Decision and Control (CDC)},
  year      = {2020},
}
```

2. Designing a self-learning controller that reduces the effort required for system identification by learning from prior experience [PDF](https://arxiv.org/pdf/2005.04755.pdf)
```
@InProceedings{JainBayesRace2020,
  author    = {Jain, Achin and O'Kelly, Matthew and Chaudhari, Pratik and Morari, Manfred},
  title     = {{BayesRace: Learning to race autonomously using prior experience}},  
  booktitle = {Proceedings of the 4th Conference on Robot Learning (CoRL)},
  year      = {2020},
}
```

## Installation
We recommend creating a new [conda](https://docs.conda.io/en/latest/) environment:

```
conda create --name bayesrace python=3.6
conda activate bayesrace
```
Then install BayesRace:

```
git clone https://github.com/jainachin/bayesrace.git
cd bayesrace/
pip install -e .
```

## How to run
The following steps are explained for the [1:43 scale autonomous racing platform](https://arxiv.org/abs/1711.07300) at ETH Zurich. We also provide code for the 1:10 scale [F1TENTH](http://f1tenth.org/) racing platform at University of Pennsylvania.

1. Compute the racing line for the track we want to race on.
	```
	cd bayes_race/raceline
	python generate_raceline_ethz.py
	```
	<p align="center">
	<img src="https://github.com/jainachin/bayesrace/blob/master/bayes_race/raceline/results/ETHZ_bestlap.png" width="400" />
	</p>

2. Run a pure pursuit controller on a different track to log sensor measurements and state estimates. This data resemble true system dynamics.
	```
	cd bayes_race/pp
	python run_pp_orca.py
	```
	<p align="center">
	<img src="https://github.com/jainachin/bayesrace/blob/master/bayes_race/pp/track_training.png" width="400" />
	</p>

3. Given a trajectory of states from Step 2, generate an equivalent trajectory using a simpler and easy-to-tune e-kinematic model.
	```
	cd bayes_race/data
	python simulate_kinematic_orca.py
	```

4. Train Gaussian process models to predict mismatch between true system dynamics from Step 2 and e-kinemtic model from Step 3.
	```
	cd bayes_race/gp
	python train_model_orca.py
	```

5. Validate the trained models on the track we want to race.
	```
	cd bayes_race/gp
	python plot_uncertainty_orca.py
	```
	<p align="center">
	<img src="https://github.com/jainachin/bayesrace/blob/master/bayes_race/gp/track_validation.png" width="400" />
	</p>

6. Run MPC with and without GP correction. By default boundary constraints are turned off for faster execution.
	```
	cd bayes_race/mpc
	python run_nmpc_orca_gp.py
	```
	MPC without GP correction is shown on the left, and with GP correction on the right.
	<p align="center">
	<img src="https://github.com/jainachin/bayesrace/blob/master/bayes_race/mpc/error_kin_mpc.png" width="400" />
	<img src="https://github.com/jainachin/bayesrace/blob/master/bayes_race/mpc/error_gp_mpc.png" width="400" />
	</p>

7. Benchmark the performance in Step 6 against MPC with true model dynamics.
	```
	cd bayes_race/mpc
	python run_nmpc_orca_true.py
	```
	<p align="center">
	<img src="https://github.com/jainachin/bayesrace/blob/master/bayes_race/mpc/track_mpc.png" width="400" />
	</p>

8. Finally, update the GP models using data collected in Step 6 that is specific to the racing track and re-run MPC with the updated GP models.
	```
	cd bayes_race/gp
	python update_model_orca.py
	```

	```
	cd bayes_race/mpc
	python run_nmpc_orca_gp_updated.py
	```

	<p align="center">
	<img src="https://github.com/jainachin/bayesrace/blob/master/bayes_race/mpc/track_mpc_lap1.png" width="400" />
	</p>
