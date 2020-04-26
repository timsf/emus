from typing import Tuple

import numpy as np
from scipy.linalg import cho_factor, solve_triangular, solve


def eval_quad(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate the quadratic form x[i].T @ inv(s) @ x[i] for each row i in x.

    :param x:
    :param s: inverse scaling matrix. if 1-dimensional, assume diagonal matrix
    :returns: evaluated quadratic forms

    >>> np.random.seed(666)
    >>> x = np.random.standard_normal((3, 2))
    >>> s = np.diag(np.random.standard_normal(2) ** 2)
    >>> eval_quad(x, s)
    array([1876.35129871, 3804.08373042,  902.76990678])
    """

    l, _ = cho_factor(s, lower=True)
    root = solve_triangular(l, x.T, lower=True).T

    return np.sum(root ** 2, 1)


def eval_matquad(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate the quadratic form x @ inv(s) @ x.T.

    :param x:
    :param s: inverse scaling matrix. if 1-dimensional, assume diagonal matrix
    :returns: evaluated quadratic form
    :returns: evaluated quadratic forms

    >>> np.random.seed(666)
    >>> x = np.random.standard_normal((3, 2))
    >>> s = np.diag(np.random.standard_normal(2) ** 2)
    >>> eval_matquad(x, s)
    array([[ 1876.35129871,  2671.64559208, -1301.46391505],
           [ 2671.64559208,  3804.08373042, -1853.03472648],
           [-1301.46391505, -1853.03472648,   902.76990678]])
    """

    l, _ = cho_factor(s, lower=True)
    root = solve_triangular(l, x.T, lower=True).T

    return root @ root.T


def logdet_pd(s: np.ndarray) -> float:
    """Evaluate log determinant of the PD matrix s

    :param s: PD matrix
    :returns: log determinant of s

    >>> np.random.seed(666)
    >>> s = np.diag(np.random.standard_normal(2) ** 2)
    >>> logdet_pd(s)
    -1.854793046254502
    """

    r, _ = cho_factor(s)

    return float(np.sum(np.log(np.diag(r))) * 2)


def precond_solve_pd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the linear system a @ x = b in a more stable way by preconditioning a.

    :param a:
    :param b:
    :returns: solution of a @ x = b

    >>> np.random.seed(666)
    >>> a = np.diag(np.random.standard_normal(2) ** 2)
    >>> b = np.random.standard_normal(2) ** 2
    >>> x = precond_solve_pd(a, b)
    >>> a @ x, b
    (array([1.37702718, 0.82636839]), array([1.37702718, 0.82636839]))
    """

    precond = 1 / np.sqrt(np.diag(a))

    return (solve(((a * precond).T * precond).T, (b.T * precond).T, sym_pos=True).T * precond).T


def eval_detquad(x: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, float]:
    """

    :param x:
    :param s:
    :returns:

    >>> np.random.seed(666)
    >>> x = np.random.standard_normal((3, 2))
    >>> s = np.diag(np.random.standard_normal(2) ** 2)
    >>> eval_detquad(x, s)
    (array([[ 43.31388528,   0.50856729],
           [ 61.66973278,   0.96321845],
           [-30.04590564,  -0.11602223]]), -8.039424065916027)
    """

    if len(s.shape) == 2:
        l, _ = cho_factor(s, lower=True)
        root = solve_triangular(l, x.T, lower=True).T
        logdet_s = np.sum(np.log(np.diag(l))) * 2
    else:
        root = x / np.sqrt(s)
        logdet_s = np.sum(np.log(s))

    return root, float(logdet_s)


def eval_double_detquad(x: np.ndarray, s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """

    :param x:
    :param s:
    :param t:
    :returns:

    >>> np.random.seed(666)
    >>> x = np.random.standard_normal((3, 2))
    >>> s = np.diag(np.random.standard_normal(3) ** 2)
    >>> t = np.diag(np.random.standard_normal(2) ** 2)
    >>> eval_double_detquad(x, s, t)
    (array([[55.07567087, 41.42730829],
           [ 1.58103634,  1.5819772 ],
           [-1.13487599, -0.28074367]]), -8.930207969151354, -1.4727706473071034)
    """

    root, logdet_s = eval_detquad(x.T, s)
    root, logdet_t = eval_detquad(root.T, t)

    return root, logdet_s, logdet_t
