import numpy as np


def est_lugsail_cov(x: np.ndarray, b: int, c: float = 0.5, r: float = 3) -> np.ndarray:
    """Use the "lugsail" estimator to estimate the monte carlo covariance.

    :param x: time series array
    :param b: batch size of underlying batch means estimator
    :param c: bias-variance trade-off parameter
    :param r: bias-variance trade-off parameter
    :returns: "lugsail" monte carlo covariance estimate

    >>> np.random.seed(666)
    >>> est_lugsail_cov(np.random.standard_normal((1000, 2)), 9)
    array([[ 0.84855114, -0.29886087],
           [-0.29886087,  0.72087742]])
    """

    # enforce equally sized batches
    b -= b % r
    if -int(x.shape[0] % b) != 0:
        x = x[:-int(x.shape[0] % b)]

    return (est_batch_cov(x, b) - c * est_batch_cov(x, np.floor(b / r))) / (1 - c)


def est_batch_cov(x: np.ndarray, batch_size: int) -> np.ndarray:
    """Use the "batch means" estimator to estimate the monte carlo covariance.

    :param x: time series array
    :param batch_size:
    :returns: "batch means" monte carlo covariance estimate

    >>> np.random.seed(666)
    >>> est_batch_cov(np.random.standard_normal((1000, 2)), 10)
    array([[ 0.91857231, -0.12105535],
           [-0.12105535,  0.93242508]])
    """

    # enforce equally sized batches
    if -int(x.shape[0] % batch_size) != 0:
        x = x[:-int(x.shape[0] % batch_size)]

    return batch_size * np.cov(est_batch_means(x, batch_size), rowvar=False)


def est_batch_means(x: np.ndarray, batch_size: int) -> [np.ndarray]:
    """Split an array into batches and compute batch means.

    :param x: time series array
    :param batch_size:
    :returns: mean for each batch
    """

    n_batches = x.shape[0] / batch_size
    batches = np.split(x, n_batches)

    return [np.mean(batch, 0) for batch in batches]
