import numpy as np


def get_holder_conjugates(p_norm):
    """Return the Holder conjugate norm orders.

    :param p_norm: p-norm either as a number or string, one
            of [1, 2, "inf"]

    >>> get_holder_conjugates("1")
    (1, inf)
    >>> get_holder_conjugates("2")
    (2, 2)
    >>> get_holder_conjugates("inf")
    (inf, 1)

    """

    if p_norm == "1" or p_norm == 1:
        p_norm = 1
        q_norm = np.inf
    elif p_norm == "2" or p_norm == 2:
        p_norm = 2
        q_norm = 2
    elif p_norm == "inf" or p_norm == np.inf:
        p_norm = np.inf
        q_norm = 1

    return p_norm, q_norm
