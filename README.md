# BayesRace
Learning-based control for autonomous racing

## About
BayesRace focuses on two problems in autonomous racing:

1. Computing the racing lines using Bayesian optimization (BayesOpt). Code [examples](https://github.com/jainachin/bayesrace#examples) below. [PDF](https://arxiv.org/pdf/2002.04794.pdf)
	<details>
	<summary>Cite this work</summary>
	<pre>
	@Article{JainRaceOpt2020,
	author   = {Jain, Achin and Morari, Manfred},
	journal  = {arXiv preprint arXiv:2002.04794},
	title    = {{Computing the racing line using Bayesian optimization}},
	year     = {2020},
	}</pre>
	</details>

2. Self-learning controller that reduces the effort required for system identification. Code will be released soon. [PDF](https://arxiv.org/pdf/2005.04755.pdf)
	<details>
	<summary>Cite this work</summary>
	<pre>
	@Article{JainBayesRace2020,
	author   = {Jain, Achin and Chaudhari, Pratik and Morari, Manfred},
	journal  = {arXiv preprint arXiv:2005.04755},
	title    = {{BayesRace: Learning to race autonomously using prior experience}},
	year     = {2020},
	}</pre>
	</details>

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
