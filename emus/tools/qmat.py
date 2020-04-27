import numpy as np


def eval_qmat(F):
    """
    :param F:
    :return:
    """

    L = np.shape(F)[1]
    Q = np.zeros((L, L))
    A = np.zeros((L, L))
    Aj = np.zeros((L, L))
    A = np.identity(L) - F
    for j in range(0, L):
        Aj = np.copy(A)
        ej = np.zeros(L)
        ej[j] = 1
        Aj[j, :] = ej
        AjInv = np.linalg.inv(Aj)
        indj = list(np.arange(0, L))
        indj.pop(j)
        for i in indj:
            Q[i, j] = AjInv[i, j] / AjInv[i, i]
    Q[Q == np.diag(Q)] = np.nan
    return Q
