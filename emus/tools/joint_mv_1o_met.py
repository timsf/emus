from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt


FloatArr = npt.NDArray[np.float_]


def sample_joint(x: FloatArr, thi: FloatArr, u_tau_thi: FloatArr, l_tau_thi: FloatArr, gam: float, delt: float, eps: FloatArr,
                 f_log_f: Callable[[FloatArr], Tuple[float, FloatArr]], f_eval_moments: Callable[[FloatArr], FloatArr],
                 ome: np.random.Generator) -> Tuple[FloatArr, FloatArr, FloatArr, FloatArr, float]:

    sig_thi = f_eval_moments(thi)
    l_sig_thi, u_tau_thi = np.linalg.eigh(sig_thi)
    l_tau_thi = 1 / l_sig_thi
    l_b_thi = l_sig_thi + (gam * delt)
    l_bi_thi = 1 / l_b_thi

    psi = ome.normal(thi, np.sqrt(delt * eps))
    sig_psi = f_eval_moments(psi)
    l_sig_psi, u_tau_psi = np.linalg.eigh(sig_psi)
    l_tau_psi = 1 / l_sig_psi
    l_b_psi = l_sig_psi + gam * delt
    l_bi_psi = 1 / l_b_psi
    l_ai_psi = l_tau_psi + 1 / (gam * delt)
    l_a_psi = 1 / l_ai_psi

    x_log_f, dx_log_f = f_log_f(x)
    mean_x = x + gam * dx_log_f
    z = ome.normal(mean_x, np.sqrt(gam * delt))
    mean_z = (((z / gam) @ u_tau_psi) * l_a_psi) @ u_tau_psi.T
    y = sample_norm_cov(mean_z, u_tau_psi, l_a_psi[np.newaxis], ome)[0]
    y_log_f, dy_log_f = f_log_f(y)
    mean_y = y + gam * dy_log_f

    log_aux_prob = y_log_f - x_log_f
    log_aux_prop = (z - x - (gam / 2) * dx_log_f) @ dx_log_f - (z - y - (gam / 2) * dy_log_f) @ dy_log_f
    log_aux_odds = log_aux_prob - log_aux_prop
    log_hyp_odds = eval_norm_prec(z[np.newaxis], np.zeros_like(z), u_tau_psi, l_bi_psi[np.newaxis])[0] - eval_norm_prec(z[np.newaxis], np.zeros_like(z), u_tau_thi, l_bi_thi[np.newaxis])[0] + np.sum(thi ** 2 - psi ** 2) / 2

    log_acc_odds = log_aux_odds + log_hyp_odds
    acc_prob = np.exp(min(0, log_acc_odds)) if not np.isnan(log_acc_odds) else 0
    if ome.uniform() < acc_prob:
        return y, psi, u_tau_psi, l_tau_psi, acc_prob
    return x, thi, u_tau_thi, l_tau_thi, acc_prob


def eval_norm_prec(x: FloatArr, mu: FloatArr, u: FloatArr, l_tau: FloatArr) -> FloatArr:

    mah = np.sum(np.square(((x - mu) @ u) * np.sqrt(l_tau)), 1)
    return (np.sum(np.log(l_tau), 1) - mah - x.shape[1] * np.log(2 * np.pi)) / 2


def sample_norm_cov(mu: FloatArr, u: FloatArr, l_sig: FloatArr, ome: np.random.Generator) -> FloatArr:

    z = ome.standard_normal(mu.shape)
    return mu + (z * np.sqrt(l_sig)) @ u.T


class LatentGaussSampler(object):

    def __init__(self, n: int, f_eval_moments: Callable[[FloatArr], FloatArr], opt_prob: float = .25):

        self.emp_prob = [1]
        self.step = [0]
        self.opt_prob = opt_prob
        self.f_eval_moments = f_eval_moments
        self.mean = np.zeros(n)
        self.var = np.ones(n)

    def sample(self, x_nil: FloatArr, thi_nil: FloatArr, u_tau_nil: FloatArr, l_tau_nil: FloatArr,
               gam: float, f_log_f: Callable[[FloatArr], Tuple[float, FloatArr]], 
               ome: np.random.Generator) -> Tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
        
        x_prime, thi_prime, u_tau_prime, l_tau_prime, emp_prob = \
            sample_joint(x_nil, thi_nil, u_tau_nil, l_tau_nil, gam, np.exp(self.step[-1]), self.var, f_log_f, self.f_eval_moments, ome)
        self.emp_prob.append(emp_prob)
        self.step.append(self.step[-1] + (emp_prob - self.opt_prob) / np.sqrt(len(self.emp_prob)))
        self.mean = self.mean + (thi_prime - self.mean) / np.sqrt(len(self.step))
        self.var = self.var + np.square(thi_prime - self.mean) / np.sqrt(len(self.step))
        return x_prime, thi_prime, u_tau_prime, l_tau_prime
