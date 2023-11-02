from typing import Iterator, Tuple

import numpy as np


def sample_posterior(y: np.ndarray, mean: np.ndarray, cov: np.ndarray, phi: float, ome: np.random.Generator
                     ) -> Iterator[np.ndarray]:

    cmean, ccov = eval_post_moments(y, mean, cov, phi)
    cf_ccov = np.linalg.cholesky(ccov)
    while True:
        yield cmean + cf_ccov @ ome.standard_normal(len(y))


def eval_post_moments(y: np.ndarray, mean: np.ndarray, cov: np.ndarray, phi: float) -> Tuple[np.ndarray, np.ndarray]:

    ext_mean = np.hstack([mean, mean])
    ext_cov = np.vstack([np.hstack([cov + np.diag(np.repeat(phi ** 2, len(y))), cov]),
                         np.hstack([cov, cov])])
    cmean, ccov = update_joint(y, np.repeat([True, False], len(y)), ext_mean, ext_cov)
    return cmean, ccov


def regress(where: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    ccoefs = np.linalg.solve(cov[where][:, where], cov[where][:, ~where]).T
    ccov = cov[~where][:, ~where] - ccoefs @ cov[where][:, ~where]
    return ccoefs, ccov


def update_joint(y: np.ndarray, where: np.ndarray, mean: np.ndarray, cov: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:

    ccoefs, ccov = regress(where, cov)
    cmean = mean[~where] + ccoefs @ (y - mean[where])
    return cmean, ccov
