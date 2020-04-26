"""
Provides routines for estimating multinomial distributions, ie:
    x[i·]|a ~ Multinomial(pi)
    where pi is the probability vector.

The prior distribution over pi is dirichlet:
    pi|(a) ~ Dirichlet(a)

The posterior parameter distribution over a is dirichlet:
    pi|(x, a) ~ Dirichlet(aN)
"""

from typing import Tuple

import numpy as np
from scipy.special import gammaln

from emus.tools.densities import eval_multinomial as eval_loglik


Data = Tuple[np.ndarray]
Param = Tuple[np.ndarray]
Hyper = Tuple[np.ndarray]


def update(x: np.ndarray, a: np.ndarray, w: np.ndarray=None) -> Hyper:
    """Compute the hyperparameters of the posterior parameter distribution.

    :param x: (Integer)[nobs, nvar]
    :param a: (>0)[nvar]
    :param w: (>0)[nobs] observation weights
    :returns: posterior hyperparameters
    """

    if w is not None:
        x = np.diag(w) @ x
    aN = np.sum(x, 0) + a

    return aN,


def sample_param(ndraws: int, a: np.ndarray) -> Param:
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param a: (>0)[nvar]
    :returns: samples from the parameter distribution
    """

    pi = np.random.dirichlet(a, ndraws)

    return pi,


def sample_data(ndraws: int, a: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param a: (>0)[nvar]
    :returns: samples from the data distribution
    """

    pi = sample_param(ndraws, a)[0]
    x = np.array([np.random.multinomial(1, pi_i, 1).flatten() for pi_i in pi])

    return np.sum(x, 0)


def eval_logmargin(x: np.ndarray, a: np.ndarray, w: np.ndarray = None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters.
    You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param x: (Integer)[nobs, nvar]
    :param a: (>0)[nvar]
    :param w: (>0)[nobs] observation weights
    :returns: log marginal likelihood
    """

    def nc(a0):
        return gammaln(np.sum(a0)) - np.sum(gammaln(a0))

    return nc(a) + gammaln(np.sum(x) + 1) - gammaln(np.sum(x) + np.sum(a)) + np.sum(gammaln(np.sum(x, 0) + a) - gammaln(np.sum(x, 0) + 1))


def get_ev(a: np.ndarray) -> Param:
    """Evaluate the expectation of parameters given hyperparameters.

    :param a: (>0)[nvar]
    :returns: parameter expectations
    """

    return a / np.sum(a),


def get_mode(a: np.ndarray) -> Param:
    """Evaluate the mode of parameters given hyperparameters.

    :param a: (>0)[nvar]
    :returns: parameter modes
    """

    return a / np.sum(a),


def _generate_fixture(nobs: int = 2, nvar: int = 3, seed: int = 666) -> (Data, Param, Hyper):
    """Generate a set of input data.

    :param nobs: (>0)
    :param nvar: (>0)
    :param seed: random number generator seed
    :returns: generated data, ground truth, hyperparameters
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set ground truth
    pi = np.ones(nvar) / nvar

    # set input
    x = np.random.multinomial(1, pi, nobs)

    # set hyperparameters
    a = np.ones(nvar) / nvar

    return (x,), (pi,), (a,)
