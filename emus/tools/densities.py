import numpy as np
import numpy.typing as npt
from scipy.special import gammaln, multigammaln

from umbrella.tools.linalg import eval_detquad, eval_double_detquad


FloatArr = npt.NDArray[np.float_]


def eval_norm(x: FloatArr, mu: FloatArr, sig: FloatArr) -> FloatArr:
    """Evaluate the log density given parameters

    :param x: data vector
    :param mu: mean vector
    :param sig: variance vector
    :returns: log density for each independent element of x

    >>> np.random.seed(666)
    >>> mu = np.random.standard_normal(2)
    >>> sig = np.random.standard_normal(2) ** 2
    >>> x = np.random.standard_normal((3, 2))
    >>> eval_norm(x, mu, sig)
    array([[-1.78642742, -1.03381855],
           [-1.314294  , -2.05003107],
           [-1.09114376, -1.79396696]])
    """

    d = (x - mu) ** 2 / sig
    kern = -d / 2
    cons = -(np.log(sig) + np.log(2 * np.pi)) / 2

    return cons + kern


def eval_mvnorm(x: FloatArr, mu: FloatArr, sig: FloatArr) -> FloatArr:
    """Evaluate the log density given parameters

    :param x: data matrix, rows corresponding to observations and columns to variables
    :param mu: mean vector
    :param sig: covariance matrix. if 1-dimensional, assume diagonal matrix
    :returns: log density for each row of x

    >>> np.random.seed(666)
    >>> mu = np.random.standard_normal(2)
    >>> sig = np.diag(np.random.standard_normal(2) ** 2)
    >>> x = np.random.standard_normal((3, 2))
    >>> eval_mvnorm(x, mu, sig)
    array([-2.82024596, -3.36432507, -2.88511072])
    """

    nvar = x.shape[1]
    d, logdet_sig = eval_detquad(x - mu, sig)
    kern = -np.sum(d ** 2, 1) / 2
    cons = -(logdet_sig + nvar * np.log(2 * np.pi)) / 2

    return cons + kern


def eval_mvnorm_inv(x: FloatArr, mu: FloatArr, ome: FloatArr) -> FloatArr:
    """Evaluate the log density given parameters

    :param x: data matrix, rows corresponding to observations and columns to variables
    :param mu: mean vector
    :param ome: precision matrix. if 1-dimensional, assume diagonal matrix
    :returns: log density for each row of x
    """

    nvar = x.shape[1]
    d = ((x - mu) @ ome @ (x - mu).T)[0]
    logdet_ome = np.linalg.slogdet(ome)[1]
    kern = -d / 2
    cons = (logdet_ome - nvar * np.log(2 * np.pi)) / 2

    return cons + kern


def eval_matnorm(x: FloatArr, mu: FloatArr, sig: FloatArr, tau: FloatArr) -> float:
    """Evaluate the log density given parameters

    :param x: data matrix
    :param mu: mean matrix
    :param sig: row covariance matrix. if 1-dimensional, assume diagonal matrix
    :param tau: column covariance matrix. if 1-dimensional, assume diagonal matrix
    :returns: log density at x

    >>> np.random.seed(666)
    >>> mu = np.random.standard_normal((3, 2))
    >>> tau = np.diag(np.random.standard_normal(3) ** 2)
    >>> sig = np.diag(np.random.standard_normal(2) ** 2)
    >>> x = np.random.standard_normal((3, 2))
    >>> eval_matnorm(x, mu, tau, sig)
    -7806.272568943354
    """

    rows, cols = x.shape
    d, logdet_sig, logdet_tau = eval_double_detquad(x - mu, sig, tau)
    kern = -float(np.sum(d ** 2) / 2)
    cons = -(rows * cols * np.log(2 * np.pi) + cols * logdet_sig + rows * logdet_tau) / 2

    return cons + kern


def eval_t(x: FloatArr, mu: FloatArr, sig: FloatArr, nu: FloatArr) -> FloatArr:
    """Evaluate the log density given parameters

    :param x: data vector
    :param mu: location vector
    :param sig: scale vector
    :param nu: degrees of freedom vector
    :returns: log density for each independent element of x

    >>> np.random.seed(666)
    >>> mu = np.random.standard_normal(2)
    >>> sig = np.random.standard_normal(2) ** 2
    >>> nu = np.random.standard_normal(2) ** 2
    >>> x = np.random.standard_normal((3, 2))
    >>> eval_t(x, mu, sig, nu)
    array([[-2.19506649, -5.51275428],
           [-1.65112507, -5.39490678],
           [-1.66824503, -5.50369595]])
    """

    d = (x - mu) ** 2 / sig
    kern = -(nu + 1) / 2 * np.log(d / nu + 1)
    cons = gammaln((nu + 1) / 2) - (gammaln(nu / 2) + (np.log(nu) + np.log(np.pi) + np.log(sig)) / 2)

    return kern + cons


def eval_mvt(x: FloatArr, mu: FloatArr, sig: FloatArr, nu: float) -> FloatArr:
    """Evaluate the log density given parameters

    :param x: data matrix, rows corresponding to observations and columns to variables
    :param mu: mean vector
    :param sig: covariance matrix. if 1-dimensional, assume diagonal matrix
    :param nu: degrees of freedom
    :returns: log density for each row of x

    >>> np.random.seed(666)
    >>> mu = np.random.standard_normal(2)
    >>> sig = np.diag(np.random.standard_normal(2) ** 2)
    >>> nu = np.random.standard_normal(1) ** 2
    >>> x = np.random.standard_normal((3, 2))
    >>> eval_mvt(x, mu, sig, nu)
    array([-3.43197508, -4.32754577, -4.13695343])
    """

    nvar = x.shape[1]
    d, logdet_sig = eval_detquad(x - mu, sig)
    kern = -(nu + nvar) / 2 * np.log(np.sum(d ** 2, 1) / nu + 1)
    cons = gammaln((nu + nvar) / 2) - gammaln(nu / 2) - (nvar * (np.log(nu) + np.log(np.pi)) + logdet_sig) / 2

    return cons + kern


def eval_matt(x: FloatArr, mu: FloatArr, sig: FloatArr, tau: FloatArr, nu: float) -> float:
    """Evaluate the log density given parameters

    :param x: data matrix
    :param mu: mean matrix
    :param sig: row covariance matrix. if 1-dimensional, assume diagonal matrix
    :param tau: column covariance matrix. if 1-dimensional, assume diagonal matrix
    :param nu: degrees of freedom
    :returns: log density at x

    >>> np.random.seed(666)
    >>> mu = np.random.standard_normal((3, 2))
    >>> tau = np.diag(np.random.standard_normal(3) ** 2)
    >>> sig = np.diag(np.random.standard_normal(2) ** 2)
    >>> nu = np.random.standard_normal(1) ** 2
    >>> x = np.random.standard_normal((3, 2))
    >>> eval_matt(x, mu, tau, sig, nu)
    array([-20.52753181])
    """

    rows, cols = x.shape
    d, logdet_sig, logdet_tau = eval_double_detquad(x - mu, sig, tau)
    if d.shape[0] > d.shape[1]:
        d = d.T @ d
    else:
        d = d @ d.T
    kern = -(nu + rows + cols - 1) / 2 * np.prod(np.linalg.slogdet(np.identity(d.shape[0]) + d / nu))
    cons = multigammaln((nu + rows + cols - 1) / 2, rows + cols) \
        - multigammaln((nu + cols - 1) / 2, cols) \
        - multigammaln((nu + rows - 1) / 2, rows) \
        - (rows * cols * (np.log(nu) + 2 * np.log(np.pi)) + cols * logdet_sig + rows * logdet_tau) / 2

    return cons + kern


def eval_poisson(x: FloatArr, lam: FloatArr) -> FloatArr:
    """Evaluate the log density given parameters

    :param x: data matrix
    :param lam: rate vector
    :returns: log density at x

    >>> np.random.seed(666)
    >>> lam = np.random.standard_normal(2) ** 2
    >>> x = np.random.poisson(lam, (3, 2))
    >>> eval_poisson(x, lam)
    array([[-1.06599903, -1.69844737],
           [-0.679286  , -0.23036736],
           [-0.679286  , -0.23036736]])
    """

    return x * np.log(lam) - lam - gammaln(x + 1)


def eval_multinomial(x: FloatArr, pi: FloatArr) -> FloatArr:
    """Evaluate the log density given parameters

    :param x: data matrix
    :param pi: probability vector
    :returns: log density at x

    >>> np.random.seed(666)
    >>> pi = np.random.dirichlet(np.ones(3))
    >>> x = np.random.multinomial(3, pi, 2)
    >>> eval_multinomial(x, pi)
    array([-2.71638384])
    """

    return x @ np.log(pi) - np.sum(gammaln(x + 1), 1) + gammaln(np.sum(x, 1) + 1)


def eval_dirichlet(x: FloatArr, a: FloatArr) -> FloatArr:
    """Evaluate the log density given parameters

    :param x: data matrix
    :param pi: probability vector
    :returns: log density at x
    """

    return (gammaln(np.sum(a)) - np.sum(gammaln(a))) + np.log(x) @ (a - 1)
