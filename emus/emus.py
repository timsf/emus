from typing import Callable, Iterator, List, TypeVar

import numpy as np
from scipy.linalg import eig, eigh
from scipy.special import logsumexp


ConfigSpace = TypeVar('ConfigSpace')
StateSpace = TypeVar('StateSpace')


def est_constant(log_p: Callable[[ConfigSpace, StateSpace], float],
                 gen_x_sampler: Callable[[ConfigSpace], Iterator[StateSpace]],
                 w: List[ConfigSpace],
                 batch_size: int = 1000,
                 max_sd: float = .1,
                 log: bool = False,
                 iid: bool = False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    :param log_p:
    :param gen_x_sampler:
    :param w:
    :param batch_size:
    :param max_sd:
    :param log:
    :param iid:
    :return:
    """

    log_p_samplers = [(np.array([log_p(w__, x_) for w__ in w]) for x_ in gen_x_sampler(w_)) for w_ in w]

    log_q = len(w) * [np.empty((0, len(w)))]
    z_est = np.ones(len(w))
    z_var = np.diag(np.repeat(np.inf, len(w)))
    chi = np.ones((len(w), len(w)))

    while np.sqrt(np.mean(np.diag(z_var))) > max_sd:
        log_q = [np.append(log_q_, log_q_ext, 0)
                 for log_q_, log_q_ext
                 in zip(log_q, resample(log_p_samplers, np.int32(np.ceil(len(w) * batch_size * reweight(chi)))))]
        z_est, f_est = eval_emus_estimator(log_q)
        z_var, chi = eval_emus_err(log_q, z_est, f_est, log, np.zeros(len(w)) if iid else None)

    return z_est if not log else np.log(z_est), z_var, chi, reweight(chi), log_q


def reweight(chi: np.ndarray) -> np.ndarray:
    """
    :param chi:
    :return:
    """

    return np.sqrt(np.sum(chi ** 2, 1)) / np.sum(np.sqrt(np.sum(chi ** 2, 1)))


def resample(log_p_samplers: List[Iterator[np.ndarray]], batch_size: np.ndarray) -> List[np.ndarray]:
    """
    :param log_p_samplers:
    :param batch_size:
    :return:
    """

    log_q = [np.array([next(s) for _ in range(n)]) for n, s in zip(batch_size, log_p_samplers)]
    log_q = [log_q_ - logsumexp(log_q_, 1)[:, np.newaxis] for log_q_ in log_q]

    return log_q


def eval_emus_estimator(log_q: List[np.ndarray]) -> (np.ndarray, np.ndarray):
    """
    :param log_q:
    :return:
    """

    log_f_est = np.array([logsumexp(log_q_, 0) - np.log(log_q_.shape[0]) for log_q_ in log_q])
    f_est = np.exp(log_f_est)
    z_est = eig(f_est, left=True, right=False)[1][:, 0]

    if z_est.dtype == np.dtype('complex128'):
        if np.all(z_est.imag == 0):
            z_est = z_est.real
        else:
            raise Exception('Stochastic matrix has complex eigenvalues.')

    return z_est / np.sum(z_est), f_est


def eval_emus_err(log_q: List[np.ndarray],
                  z_est: np.ndarray,
                  f_est: np.ndarray,
                  log: bool,
                  n_lags: np.ndarray = None) -> (np.ndarray, np.ndarray):
    """
    :param log_q:
    :param z_est:
    :param f_est:
    :param log:
    :param n_lags:
    :return:
    """

    n_windows = len(log_q)
    n_samples = np.array([log_q_.shape[0] for log_q_ in log_q])
    if n_lags is None:
        n_lags = np.int32(np.floor([log_q_.shape[0] ** (1 / 3) for log_q_ in log_q]))
    f_cov = [(np.cov(np.exp(log_q_.T)) if n == 0 else est_lugsail_cov(np.exp(log_q_), n))
             for n, log_q_ in zip(n_lags, log_q)]

    f_inv = grp_invert(f_est)

    z_jac = [(z_est[i] if not log else 1) * f_inv for i in range(n_windows)]
    chi_sq = [z_jac[i].T @ f_cov[i] @ z_jac[i] for i in range(n_windows)]
    chi = np.array([np.sqrt(np.diag(chi_sq[i])) for i in range(n_windows)])
    z_var = sum([chi_sq[i] / n_samples[i] for i in range(n_windows)])

    return z_var, chi


def grp_invert(p: np.ndarray) -> np.ndarray:
    """
    :param p:
    :return:
    """

    a = np.identity(p.shape[0]) - p
    q, r = np.linalg.qr(a)
    u = r[:-1, :-1]
    pi = q[:, -1] / np.sum(q[:, -1])

    u2 = np.zeros(np.array(u.shape) + 1)
    u2[:-1, :-1] = np.linalg.inv(u)
    a2 = np.identity(len(pi)) - np.outer(np.ones(len(pi)), pi)

    return a2 @ u2 @ q.T @ a2


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


def est_batch_means(x: np.ndarray, batch_size: int) -> List[np.ndarray]:
    """Split an array into batches and compute batch means.

    :param x: time series array
    :param batch_size:
    :returns: mean for each batch
    """

    n_batches = x.shape[0] / batch_size
    batches = np.split(x, n_batches)

    return [np.mean(batch, 0) for batch in batches]


def correct_pd(s: np.ndarray, n_samples: int, e: float = None, r: float = None) -> np.ndarray:
    """Ensure positive definiteness of a covariance matrix estimate by inflating its negative eigenvalues

    :param s: symmetric covariance matrix estimate
    :param n_samples: size of estimation sample
    :param e: eigenvalue threshold scale
    :param r: eigenvalue threshold power
    :returns: corrected covariance matrix estimate
    """

    # set defaults for threshold scaling
    if e is None:
        e = np.sqrt(np.log(n_samples) / s.shape[0])
    if r is None:
        r = 0.5

    sd = np.sqrt(np.diag(s))
    eigval, eigvec = eigh((s / sd).T / sd)
    threshold = e * n_samples ** (-r)
    pd_eigval = np.where(eigval > threshold, eigval, threshold)

    return (eigvec.T @ np.diag(pd_eigval) @ eigvec * sd).T * sd
