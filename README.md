# BRAT: Boulevard Regularized Additive Regression Trees

This repository contains the code, experiments, and documentation for the paper [Statistical Inference for Gradient Boosting Regression](https://openreview.net/attachment?id=gLU0UV85Kv&name=pdf) submitted to the 2025 NeurIPS conference. All experiments are GPU optional. Simulations are done on python 3.9.6.

## Initialization

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from BRAT.algorithms import BRATD

# toy regression data
X, y = make_regression(
    n_samples=2000,
    n_features=10,
    noise=5.0,
    random_state=0,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# fit BRAT-D and track test MSE across boosting steps
model = BRATD(n_estimators=100, learning_rate=0.5, max_depth=4, dropout_rate=0.5)
mse_path = model.fit(X_train, y_train, X_test, y_test)

print(f"Final test MSE: {mse_path[-1]:.3f}")
print("First 3 predictions:", model.predict(X_test[:3]))

# built-in variance estimates and 95% prediction interval at a test point
sigma_hat2, r_norm, tau_hat2 = model.est_tau_hat2(
    in_bag=False, Nystrom_subsample=0.3, x=X_test[0]
)
print(f"Estimated noise var: {sigma_hat2:.3f}, ||r_n||: {r_norm.item():.3f}, tau^2: {tau_hat2.item():.3f}")

from BRAT.inferences import PI
pi, y_pred, _, _, _ = PI(model, in_bag=False, x=X_test[0], Nystrom_subsample=0.3)
print("95% prediction interval:", pi)
```

## Reproduce Results

You can find all the algorithms in `./src/BRAT/algorithms.py` defined. `./src/BRAT/variance_estimation.py` provides the methods of computing the reweighting vector $r_n$ and gives the built-in variance estimation. 

To reproduce the results presented in the paper, you can find 4 notebooks in `./experiments/`:

1. `1d_intervals.ipynb`: Visualizations of built-in interval estimations given by BRATD.

2. `optuna_mse.ipynb`: Fetch and clean the data from [UCI Machine Learning Repository](https://archive.ics.uci.edu/). Tune the models using *Optuna* and visualize the optimized models' mse trajectories with error bars.

3. `coverage_rates.ipynb`: Evaluate the performance of our built-in intervals in terms of coverage. You can reproduce the rainclouds plot of coverage rates in this notebook. 

4. `variable_importance.ipynb`: An example and a trajectory study of the type I and type II error made by the model as sample size increases is provided in this notebook.

## Repo Layout

```bash
src/BRAT/        # core library code (BRAT-D/BRAT-P, variance, utils)
experiments/     # reproduce the experiments here!
notebooks/       
plotting/        # plotting helpers
plots/           
reports/         
artifacts/       # cached models/results
jobs/            
submission/      # conference submission assets
requirements.txt # dependencies
setup.py         # package install entry
README.md
```
