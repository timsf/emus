from typing import Iterator, Tuple
import unittest

import numpy as np
from scipy.stats import kstest, norm

from emus.emus import est_constant
from emus.models import multinomial
from emus.tools.densities import eval_dirichlet


def plot_estimate(w: np.ndarray, z: np.ndarray, z_est: np.ndarray, z_sd: np.ndarray):

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    plt.scatter(w, z, color='#fb8072')
    plt.errorbar(w, z_est, yerr=norm(0, 1).ppf(.975) * z_sd, fmt='_', color='gray', elinewidth=1)
    plt.xlabel('$\lambda$')
    plt.ylabel('$p(x | \lambda)$')
    plt.show()

    # plt.plot(w, z, alpha=.5, color='black')
    # plt.scatter(w, z_est, color='#fb8072')
    # plt.xlabel('$\lambda$')
    # plt.ylabel('$z(\lambda)$')
    # plt.show()


def plot_cov(w: np.ndarray, z_var: np.ndarray):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    plt.style.use('ggplot')

    sns.heatmap(pd.DataFrame(z_var, index=np.round(w, 2), columns=np.round(w, 2)), center=0)
    plt.xlabel('$\\lambda$')
    plt.ylabel('$\\lambda$')
    plt.show()


def plot_importance(w: np.ndarray, mu: np.ndarray, center: float = None):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    plt.style.use('ggplot')

    sns.heatmap(pd.DataFrame(mu, index=np.round(w, 2), columns=np.round(w, 2)), center=center)
    plt.xlabel('$\\lambda$')
    plt.ylabel('$\\lambda$')
    plt.show()


def line_plot(w: np.ndarray, a: np.ndarray, ylab: str):

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    plt.scatter(w, a, color='gray')
    plt.plot(w, a, color='gray')
    plt.xlabel('$\\lambda$')
    plt.ylabel(ylab)
    plt.show()


def generate_fixture(nobs: int = 2, nvar: int = 3, seed: int = 666) -> (multinomial.Data, multinomial.Param, Tuple[float]):

    np.random.seed(seed)
    a = 1
    b = a * np.ones(nvar)
    pi = np.random.dirichlet(b)
    x = np.random.multinomial(nobs, pi, 1)

    return (x,), (pi,), (a,)


def eval_logjoint(x: np.ndarray, pi: np.ndarray, a: float) -> float:

    b = a * np.ones(len(pi))

    return np.sum(multinomial.eval_loglik(x, pi)) + np.sum(eval_dirichlet(pi[np.newaxis], b))


def eval_logmargin(x: np.ndarray, a: float) -> float:

    b = a * np.ones(x.shape[1])

    return multinomial.eval_logmargin(x, b)


def param_sampler(x: np.ndarray, a: float) -> Iterator[multinomial.Param]:

    b = a * np.ones(x.shape[1])
    bN, = multinomial.update(x, b)

    while True:
        yield tuple([the[0] for the in multinomial.sample_param(1, bN)])


class EmusTest(unittest.TestCase):

    def setUp(self, n_obs=100, n_vars=10, n_windows=10, scale_windows=1, seed=666):

        self.data, self.param, self.hyper = generate_fixture(n_obs, n_vars, seed)
        self.windows = -np.log(1 - scale_windows * np.linspace(0, 1, n_windows + 2)[1:-1])

    def test_emus(self, alpha=.05, max_sd=.01):

        log_p = lambda w, the: eval_logjoint(*self.data, *the, w)
        sampler = lambda w: param_sampler(*self.data, w)

        z_est, z_sd, uv_mu, mv_mu, log_q = est_constant(log_p, sampler, self.windows, 100, max_sd, True)
        z = np.exp([eval_logmargin(*self.data, w) for w in self.windows])
        z = z / np.sum(z)

        self.assertLess(alpha, kstest((z - z_est) / z_sd, norm(0, 1).cdf)[1])

    def test_emus_log(self, alpha=.05, max_sd=.01):

        log_p = lambda w, the: eval_logjoint(*self.data, *the, w)
        sampler = lambda w: param_sampler(*self.data, w)

        z_est, z_var, chi, mu, log_q = est_constant(log_p, sampler, self.windows, 100, max_sd, True, True)
        z_sd = np.sqrt(np.diag(z_var))
        z = np.exp([eval_logmargin(*self.data, w, *self.hyper[1:]) for w in self.windows])
        z = np.log(z / np.sum(z))

        self.assertLess(alpha, kstest((z - z_est) / z_sd, norm(0, 1).cdf)[1])
