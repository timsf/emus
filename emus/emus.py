from typing import Iterator, List, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

import umbrella.tools.ledwolf
import umbrella.tools.linalg


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]
ConfigSpace = FloatArr  # TypeVar('ConfigSpace')


def eval_vardi_estimator(
    log_psi: List[FloatArr],
    max_iter: int = 1,
    n_polish: int = 100,
) -> Tuple[FloatArr, FloatArr]:

    def iter_vardi(old_z: FloatArr, n_iter: int) -> Tuple[FloatArr, FloatArr]:
        f_est = est_overlap(log_psi, old_z / n_samples)
        new_z, new_z_rel, f_inv = solve_emus_system(f_est, old_z, n_polish)
        if n_iter == 1 or np.allclose(new_z_rel, n_samples / np.sum(n_samples)):
            return new_z, f_inv
        return iter_vardi(new_z, n_iter - 1)

    n_samples = np.array([log_psi_.shape[0] for log_psi_ in log_psi])
    return iter_vardi(n_samples / np.sum(n_samples), max_iter)


def solve_emus_system(
    f_est: FloatArr,
    old_z: FloatArr,
    n_polish: int,
) -> Tuple[FloatArr, FloatArr, FloatArr]:

    new_z_rel, f_inv = umbrella.tools.linalg.eval_grpinv(f_est)
    new_z_rel = umbrella.tools.linalg.power_iterate_stat(f_est, np.where(new_z_rel > 0, new_z_rel, 0) / np.sum(new_z_rel[new_z_rel > 0]), n_polish)
    f_inv = umbrella.tools.linalg.power_iterate_grpinv(f_inv, f_est, new_z_rel, n_polish)
    new_z = new_z_rel * old_z / np.sum(new_z_rel * old_z)
    return new_z, new_z_rel, f_inv


def eval_vardi_estimator_alt(
    log_psi: List[FloatArr],
    max_iter: int = 100,
    eps: float = 1,
) -> Tuple[FloatArr, FloatArr]:

    def iter_vardi(old_z: FloatArr, n_iter: int) -> Tuple[FloatArr, FloatArr]:
        f_est = est_overlap(log_psi, old_z / n_samples)
        new_z_rel = eps * (np.sum(f_est, 0) - 1) + 1
        new_z_rel = np.where(new_z_rel > 0, new_z_rel, 0) / np.sum(new_z_rel[new_z_rel > 0])
        new_z = new_z_rel * old_z / np.sum(new_z_rel * old_z)
        if n_iter == 1 or np.allclose(new_z_rel, n_samples / np.sum(n_samples)):
            return new_z
        return iter_vardi(new_z, n_iter - 1)

    n_samples = np.array([log_psi_.shape[0] for log_psi_ in log_psi])
    return iter_vardi(n_samples / np.sum(n_samples), max_iter)


def resample_lam(
    n_samples: int,
    bounds_lam: Tuple[FloatArr, FloatArr],
    ome: np.random.Generator,
) -> List[ConfigSpace]:

    strata = [np.linspace(lo, hi, n_samples + 1) for lo, hi in zip(*bounds_lam)]
    sample_from = [ome.permutation(np.arange(n_samples)) for _ in range(len(bounds_lam))]
    return list(np.array([ome.uniform(strata_[sample_from_], strata_[sample_from_ + 1])
                for strata_, sample_from_ in zip(strata, sample_from)]).T)


def resample_the(log_psi_samplers: List[Iterator[FloatArr]], batch_size: IntArr) -> List[FloatArr]:

    return [np.array([next(s) for _ in range(n)]) if n != 0 else np.empty((0, len(batch_size)))
            for n, s in zip(batch_size, log_psi_samplers)]


def est_overlap(log_psi: List[FloatArr], z_guess: FloatArr) -> FloatArr:

    log_r = [log_psi_ - np.log(z_guess[np.newaxis]) for log_psi_ in log_psi]
    log_f_est = np.array([logsumexp(log_r_ - logsumexp(log_r_, 1)[:, np.newaxis], 0) - np.log(log_r_.shape[0])
                          for log_r_ in log_r])
    return np.exp(log_f_est)


def est_overlap_var(log_psi: List[FloatArr], shrink: bool = False) -> List[FloatArr]:

    norm_log_psi = [log_psi_ - logsumexp(log_psi_, 1)[:, np.newaxis] for log_psi_ in log_psi]
    target = [np.ones(log_psi_.shape[1]) for log_psi_ in norm_log_psi]
    cond_target = [np.diag(target_) - np.outer(target_, target_) / np.sum(target_)
                   for target_ in target]
    # if n_lags is None:
    #     n_lags = np.int_(np.floor([log_psi_.shape[0] ** (1 / 3) for log_psi_ in log_psi]))
    # base = np.identity(log_psi[0].shape[1]) - np.ones((log_psi[0].shape[1], log_psi[0].shape[1])) / log_psi[0].shape[1]
    if shrink:
        f_cov = [umbrella.tools.ledwolf.eval_ledwolf(np.exp(log_psi_), target_) if log_psi_.shape[0] > 1 else target_
                 for log_psi_, target_ in zip(norm_log_psi, cond_target)]
    else:
        f_cov = [np.cov(np.exp(log_psi_.T)) if log_psi_.shape[0] > 1 else target
                 for log_psi_ in norm_log_psi]
    return f_cov


def eval_emus_coefs(
    log_psi: List[FloatArr],
    z_est: FloatArr,
    inv_est_if: FloatArr,
) -> List[FloatArr]:

    f_cov = est_overlap_var(log_psi)
    z_jac = [z_est[i] * inv_est_if for i in range(len(log_psi))]
    chi_sq = [z_jac[i].T @ f_cov[i] @ z_jac[i] for i in range(len(log_psi))]
    return chi_sq


def eval_rhs_err(log_psi: List[FloatArr], z_est: FloatArr) -> FloatArr:

    f_cov = est_overlap_var(log_psi)
    return sum([(z_est_ ** 2 / log_psi_.shape[0]) * f_cov_ for f_cov_, z_est_, log_psi_ in zip(f_cov, z_est, log_psi)])


def eval_emus_err(
    log_psi: List[FloatArr],
    z_est: FloatArr,
    inv_est_if: FloatArr,
) -> FloatArr:

    return sum([chi_sq_ / log_psi_.shape[0] for chi_sq_, log_psi_ in zip(eval_emus_coefs(log_psi, z_est, inv_est_if), log_psi)])


def eval_analytic_weights(chi_sq: List[FloatArr]) -> FloatArr:

    diag_chi_sq = np.array([np.diag(chi_sq[i]) for i in range(len(chi_sq))])
    score = np.sqrt(np.sum(diag_chi_sq, 1))
    return score / np.sum(score)


def extrapolate(
    log_psi: List[FloatArr], 
    log_tpsi: List[FloatArr], 
    zt_est: FloatArr
) -> FloatArr:

    log_g_est = np.array([logsumexp(log_psi_ - logsumexp(log_tpsi_, 1)[:, np.newaxis], 0) - np.log(log_psi_.shape[0])
                          for log_psi_, log_tpsi_ in zip(log_psi, log_tpsi)])
    g_est = np.exp(log_g_est)
    z_est = g_est.T @ zt_est
    return z_est / np.sum(z_est)