from typing import List

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from umbrella import emus


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]
ConfigSpace = FloatArr  # TypeVar('ConfigSpace')


def eval_smoothed_estimator(
    log_psi: List[FloatArr],
    log_k: FloatArr,
    n_iter: int,
    init_z: FloatArr = None,
) -> FloatArr:

    if init_z is None:
        init_z = np.ones(len(log_psi)) / len(log_psi)
    f_est = est_overlap(log_psi, log_k, init_z)
    z_emus = u = init_z
    for _ in range(n_iter):
        g = log_k + np.log(u)
        g = np.exp(g.T - logsumexp(g, 1)).T
        z_emus = emus.solve_emus_system(g @ f_est, np.ones(len(init_z)) / len(init_z), 10)[0]
        u = z_emus
    return z_emus

    # lb = np.min(lam) - np.diff(lam)[0] / 2
    # ub = np.max(lam) + np.diff(lam)[-1] / 2
    # comp_smoother = logsumexp([
    #     norm.logpdf(0, loc=lam[:, np.newaxis] - lam[np.newaxis], scale=np.sqrt(sq_bandwidth)),
    #     norm.logpdf(0, loc=lam[:, np.newaxis] + lam[np.newaxis] - 2 * lb, scale=np.sqrt(sq_bandwidth)),
    #     norm.logpdf(0, loc=lam[:, np.newaxis] + lam[np.newaxis] - 2 * ub, scale=np.sqrt(sq_bandwidth)),
    # ], 0)
    # log_tpsi = [logdotexp(log_psi_, comp_smoother) for log_psi_ in log_psi]
    # z_emus, f_inv_est = emus.eval_vardi_estimator(log_tpsi)
    # return z_emus
    # g = smoother + np.log(z)
    # g = np.exp(g.T - logsumexp(g, 1)).T
    # sol = np.linalg.eig(f_est.T @ g.T)
    # z_emus = sol[1][:, np.argmax(sol[0])].real


def est_overlap(
    log_psi: List[FloatArr],
    log_k: FloatArr,
    z_guess: FloatArr,
) -> FloatArr:

    log_bpsi = [logdotexp(log_psi_, log_k) - np.log(z_guess[np.newaxis]) for log_psi_ in log_psi]
    f_est = np.exp([logsumexp(log_r_ - logsumexp(log_r_, 1)[:, np.newaxis], 0) - np.log(log_r_.shape[0])
                    for log_r_ in log_bpsi])
    return f_est


def logdotexp(A: FloatArr, B: FloatArr) -> FloatArr:

    max_A = np.max(A)
    max_B = np.max(B)
    C = np.dot(np.exp(A - max_A), np.exp(B - max_B))
    np.log(C, out=C)
    C += max_A + max_B
    return C
