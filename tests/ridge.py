from typing import Iterator, Tuple
import unittest

import numpy as np
from scipy.stats import invgamma, kstest, norm

from emus.emus import est_constant
from emus.models import lm


def plot_estimate(w: np.ndarray, z: np.ndarray, z_est: np.ndarray, z_sd: np.ndarray):

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    plt.scatter(w, z, color='#fb8072')
    plt.errorbar(w, z_est, yerr=norm(0, 1).ppf(.975) * z_sd, fmt='_', color='gray', elinewidth=1)
    plt.xlabel('$\\lambda$')
    plt.ylabel('$p(y | X, \\lambda)$')
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


def generate_fixture(nres: int = 3, nobs: int = 2, nvar: int = 1, rho: int = 0, seed: int = 666) -> (lm.Data, lm.Param, Tuple[float, np.ndarray, np.ndarray]):

    np.random.seed(seed)
    sig = 1 / np.random.gamma(.5, .5, nres)
    bet = np.random.standard_normal((nres, nvar)) * np.sqrt(sig)[:, np.newaxis]
    tau = rho * np.ones((nvar, nvar)) + (1 - rho) * np.identity(nvar)
    x = np.random.multivariate_normal(np.zeros(nvar), tau, nobs).T
    y = bet @ x + np.random.standard_normal((nres, nobs)) * np.sqrt(sig)[:, np.newaxis]
    lam = 1
    v = np.ones(nres)
    s = np.ones(nres) * v

    return (y, x), (bet, sig), (lam, v, s)


def eval_logmargin(y: np.ndarray, x: np.ndarray, lam: float, v: np.ndarray, s: np.ndarray) -> float:

    m = np.zeros((y.shape[0], x.shape[0]))
    l = lam / np.var(x, 1)

    return lm.eval_logmargin(y, x, m, np.diag(l), v, s)


def eval_logjoint(y: np.ndarray, x: np.ndarray, bet: np.ndarray, sig: np.ndarray, lam: float, v: np.ndarray, s: np.ndarray) -> float:

    m = np.zeros((y.shape[0], x.shape[0]))
    l = lam / np.var(x, 1)

    return np.sum(lm.eval_loglik(y, x, bet, sig)) \
           + np.sum(lm.eval_norm(bet, m, sig / l)) \
           + np.sum(invgamma(a=v / 2, scale=s / 2).logpdf(sig))


def param_sampler(y: np.ndarray, x: np.ndarray, lam: float, v: np.ndarray, s: np.ndarray) -> Iterator[lm.Param]:

    m = np.zeros((y.shape[0], x.shape[0]))
    l = np.diag(lam / np.var(x, 1))
    mN, lN, vN, sN = lm.update(y, x, m, l, v, s)

    while True:
        yield tuple([the[0] for the in lm.sample_param(1, mN, lN, vN, sN)])


class EmusTest(unittest.TestCase):

    def setUp(self, n_obs=10, n_vars=2, cor=0, n_windows=10, scale_windows=1, seed=1):

        self.data, self.param, self.hyper = generate_fixture(1, n_obs, n_vars, cor, seed)
        self.windows = -np.log(1 - scale_windows * np.linspace(0, 1, n_windows + 2)[1:-1])

    def test_emus(self, alpha=.05, max_sd=.01):

        log_p = lambda w, the: eval_logjoint(*self.data, *the, w, np.ones(1), np.ones(1))
        sampler = lambda w: param_sampler(*self.data, w, np.ones(1), np.ones(1))

        z_est, z_var, chi, mu, log_q = est_constant(log_p, sampler, self.windows, 100, max_sd, False, True)
        z_sd = np.sqrt(np.diag(z_var))
        z = np.exp([eval_logmargin(*self.data, w, *self.hyper[1:]) for w in self.windows])
        z = z / np.sum(z)

        self.assertLess(alpha, kstest((z - z_est) / z_sd, norm(0, 1).cdf)[1])

    def test_emus_log(self, alpha=.05, max_sd=.01):

        log_p = lambda w, the: eval_logjoint(*self.data, *the, w, np.ones(1), np.ones(1))
        sampler = lambda w: param_sampler(*self.data, w, np.ones(1), np.ones(1))

        z_est, z_var, chi, mu, log_q = est_constant(log_p, sampler, self.windows, 100, max_sd, True, True)
        z_sd = np.sqrt(np.diag(z_var))
        z = np.exp([eval_logmargin(*self.data, w, *self.hyper[1:]) for w in self.windows])
        z = np.log(z / np.sum(z))

        self.assertLess(alpha, kstest((z - z_est) / z_sd, norm(0, 1).cdf)[1])
