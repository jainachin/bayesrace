# BayesRace
Learning-based control for autonomous racing

## About
BayesRace focuses on two problems in autonomous racing:
1. Computing the racing lines using Bayesian optimization (BayesOpt). Check out paper on [arXiv](https://arxiv.org/pdf/2002.04794.pdf) and see [examples](https://github.com/jainachin/bayesrace#examples).
2. Self-learning controller that minimizes the effort required for system identification. Paper and code will be released soon.


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

## Examples
1. Compute the racing line for the track at [ETH Zurich](https://arxiv.org/abs/1711.07300)

	```
	python raceline_ethz.py
	```
	<p align="center">
	<img src="https://github.com/jainachin/bayesrace/blob/master/examples/results/bestlap-ETHZ.png" width="500" />
	</p>

2. Compute the racing line for the track at [UC Berkeley](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8896988)

	```
	python raceline_ucb.py
	```
	<p align="center">
	<img src="https://github.com/jainachin/bayesrace/blob/master/examples/results/bestlap-UCB.png" width="500" />
	</p>
