from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from umbrella import emus


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]
ConfigSpace = FloatArr  # TypeVar('ConfigSpace')


def eval_vardi_estimator(
    log_psi: List[FloatArr],
    log_k: FloatArr,
    max_iter: int = 1,
    n_polish: int = 8,
    init_z: FloatArr = None,
    weight: bool = False,
) -> Tuple[FloatArr, FloatArr]:

    def iter_vardi(old_z: FloatArr, n_iter: int) -> Tuple[FloatArr, FloatArr]:
        if weight:
            f_est = est_full_weighted_overlap(log_psi, log_k, old_z)
        else:
            f_est = est_overlap(log_psi, log_k, old_z)
        new_z, new_z_rel, f_inv = emus.solve_emus_system(f_est, old_z, n_polish)
        if n_iter == 1 or np.allclose(new_z_rel, np.ones(len(log_psi)) / len(init_z)):
            return new_z, f_inv
        return iter_vardi(new_z, n_iter - 1)

    if init_z is None:
        init_z = np.ones(len(log_psi)) / len(log_psi)
    return iter_vardi(init_z, max_iter)


def est_overlap(
    log_psi: List[FloatArr],
    log_k: FloatArr,
    z_guess: FloatArr,
) -> FloatArr:

    log_bpsi = [logdotexp(log_psi_, log_k) - np.log(z_guess[np.newaxis]) for log_psi_ in log_psi]
    f_est = np.exp([logsumexp(log_r_ - logsumexp(log_r_, 1)[:, np.newaxis], 0) - np.log(log_r_.shape[0])
                    for log_r_ in log_bpsi])
    return f_est


def est_weighted_overlap(
    log_psi: List[FloatArr],
    log_k: FloatArr,
    z_guess: FloatArr,
) -> FloatArr:

    log_bpsi = [logdotexp(log_psi_, log_k) - np.log(z_guess[np.newaxis]) for log_psi_ in log_psi]
    log_w = [log_bpsi_[:, i] - log_psi_[:, i] - logsumexp(log_bpsi_[:, i] - log_psi_[:, i]) for i, (log_psi_, log_bpsi_) in enumerate(zip(log_psi, log_bpsi))]
    f_est = np.exp([logsumexp(log_w_[:, np.newaxis] + log_psi_ - logsumexp(log_psi_, 1)[:, np.newaxis], 0)
                    for log_w_, log_psi_ in zip(log_w, log_bpsi)])
    return f_est


def est_full_weighted_overlap(
    log_psi: List[FloatArr],
    log_k: FloatArr,
    z_guess: FloatArr,
) -> FloatArr:

    flat_log_psi = np.vstack(log_psi)
    flat_log_bpsi = logdotexp(flat_log_psi, log_k) - np.log(z_guess[np.newaxis])
    flat_log_tpsi = flat_log_bpsi - logsumexp(flat_log_bpsi, 1)[:, np.newaxis]
    flat_log_v = flat_log_bpsi - logsumexp(flat_log_psi, 1)[:, np.newaxis]
    flat_log_w = flat_log_v - logsumexp(flat_log_v, 0)
    f_est = np.exp(logdotexp(flat_log_w.T, flat_log_tpsi))
    return f_est


def logdotexp(A: FloatArr, B: FloatArr) -> FloatArr:

    max_A = np.max(A)
    max_B = np.max(B)
    C = np.dot(np.exp(A - max_A), np.exp(B - max_B))
    np.log(C, out=C)
    C += max_A + max_B
    return C
