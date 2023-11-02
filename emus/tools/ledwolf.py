import numpy as np
import numpy.typing as npt


FloatArr = npt.NDArray[np.float_]


def eval_ledwolf(x: FloatArr, target: FloatArr) -> FloatArr:

    y = x - np.mean(x, 0)
    s = y.T @ y / x.shape[0]
    m = eval_iprod(s, target)
    d_sq = eval_norm(s - m * target) ** 2
    b_sq = min(sum([eval_norm(np.outer(y_, y_) - s) ** 2 for y_ in y]) / y.shape[0] ** 2, d_sq)
    return b_sq / d_sq * m * target + (d_sq - b_sq) / d_sq * s


def eval_norm(a: FloatArr) -> float:

    return np.sqrt(eval_iprod(a, a))


def eval_iprod(a: FloatArr, b: FloatArr) -> float:

    return np.trace(a @ b.T) / a.shape[0]
