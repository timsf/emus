from typing import Iterator, List

import numpy as np
from scipy.stats import norm, truncnorm


def sample_posterior_censored(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> Iterator[np.ndarray]:

    cmean, ccov = compress_joint(y, mean, cov)
    return sample_posterior_probit(np.repeat(False, np.sum(y == 0)), cmean, ccov)


def sample_posterior_probit(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> Iterator[np.ndarray]:

    ccoefs, ccov = zip(*[regress(np.arange(len(mean)) != i, cov) for i in range(len(mean))])
    z = np.where(y, 1.0, -1.0)
    while True:
        z = sample_signed(z, mean, ccoefs, ccov)
        yield z


def sample_signed(z: np.ndarray, mean: np.ndarray, coefs: List[np.ndarray], cov: List[np.ndarray]) -> np.ndarray:

    z = np.copy(z)
    for i, (coefs_, ccov_) in enumerate(zip(coefs, cov)):
        cmean_ = mean[i] + coefs_ @ (np.delete(z, i) - np.delete(mean, i))
        if z[i] < 0:
            z[i] = sample_trunc_norm(cmean_[0], np.sqrt(ccov_[0, 0]), -np.inf, 0)
        else:
            z[i] = sample_trunc_norm(cmean_[0], np.sqrt(ccov_[0, 0]), 0, np.inf)
    return z


def sample_trunc_norm(mu: float, sig: float, lb: float, ub: float) -> float:

    qlo = norm.cdf((lb - mu) / sig)
    qhi = norm.cdf((ub - mu) / sig)
    if qlo != qhi:
        return norm.ppf(qlo + np.random.uniform() * (qhi - qlo)) * sig + mu
    else:
        return truncnorm.rvs((lb - mu) / sig, (ub - mu) / sig, mu, sig)


def regress(where: np.ndarray, cov: np.ndarray) -> (np.ndarray, np.ndarray):

    ccoefs = np.linalg.solve(cov[where][:, where], cov[where][:, ~where]).T
    ccov = cov[~where][:, ~where] - ccoefs @ cov[where][:, ~where]
    return ccoefs, ccov


def compress_joint(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> (np.ndarray, np.ndarray):

    return update_joint(y[y != 0], y != 0, mean, cov)


def update_joint(y: np.ndarray, where: np.ndarray, mean: np.ndarray, cov: np.ndarray
                 ) -> (np.ndarray, np.ndarray, np.ndarray):

    ccoefs, ccov = regress(where, cov)
    cmean = mean[~where] + ccoefs @ (y - mean[where])
    return cmean, ccov


def sample_posterior_naive(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> Iterator[np.ndarray]:

    cmean, ccov = compress_joint(y, mean, cov)
    cchol = np.linalg.cholesky(ccov)

    while True:
        z = cmean + cchol @ np.random.standard_normal(len(cmean))
        if np.all(z < 0):
            yield z
