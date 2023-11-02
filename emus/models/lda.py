from typing import Iterator

import numpy as np


def sample_posterior(v: np.ndarray, d: np.ndarray, n_topics: int = 2, prior_n_pi: float = 1, prior_n_kap: float = 1,
                     ) -> Iterator[np.ndarray]:

    t = np.random.choice(np.arange(n_topics), len(v))
    by_type, by_doc, total = initiate_counts(v, d, t)

    while True:
        t, by_type, by_doc, total = update_assignments(v, d, t, by_type, by_doc, total, prior_n_pi, prior_n_kap)
        yield t


def initiate_counts(w: np.ndarray, d: np.ndarray, t: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):

    by_type = np.vstack([np.bincount(w[t == i_], minlength=np.max(w) + 1) for i_ in range(np.max(t) + 1)]).T
    by_doc = np.vstack([np.bincount(d[t == i_], minlength=np.max(d) + 1) for i_ in range(np.max(t) + 1)]).T
    total = np.sum(by_type, 0)

    return by_type, by_doc, total


def update_assignments(v: np.ndarray, d: np.ndarray, t: np.ndarray,
                       by_type: np.ndarray, by_doc: np.ndarray, total: np.ndarray,
                       prior_n_pi: float = 1, prior_n_kap: float = 1) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    t = t.copy()
    by_type = by_type.copy()
    by_doc = by_doc.copy()
    total = total.copy()

    for i, (v_, d_) in enumerate(zip(v, d)):
        t_nil = t[i]
        t_prime = sample_assignment(v_, d_, t_nil, by_type, by_doc, total, prior_n_pi, prior_n_kap)
        t[i] = t_prime
        if t_nil != t_prime:
            dt_oh = to_onehot(t_prime, len(total)) - to_onehot(t_nil, len(total))
            by_type[v_] += dt_oh
            by_doc[d_] += dt_oh
            total += dt_oh

    return t, by_type, by_doc, total


def sample_assignment(v_: int, d_: int, t_: int,
                      by_type: np.ndarray, by_doc: np.ndarray, total: np.ndarray,
                      prior_n_pi: float = 1, prior_n_kap: float = 1) -> int:

    t_nil_oh = to_onehot(t_, len(total))
    p_inc = (by_doc[d_] - t_nil_oh + prior_n_pi) * (by_type[v_] - t_nil_oh + prior_n_kap) \
            / (total - t_nil_oh + by_type.shape[0] * prior_n_kap)
    return np.argmax(np.random.multinomial(1, p_inc / np.sum(p_inc))).item()


def to_onehot(t_: int, dim: int) -> np.ndarray:

    t_oh = np.zeros(dim, dtype=np.int32)
    t_oh[t_] = 1
    return t_oh
