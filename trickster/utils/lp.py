"""Utility for easily computing Holder conjugates of $$L_p$$ spaces norms.
"""

import numpy as np


class LpSpace:
    """$$L_p$$ space representation.

    :param p: Norm of the space

    >>> LpSpace(1).q == np.inf
    True
    >>> LpSpace(2).q == 2
    True
    >>> LpSpace("inf").q == 1
    True

    """

    def __init__(self, p):
        if isinstance(p, LpSpace):
            other_space = p
            self.p = other_space.p
            self._q = other_space.q

        elif str(p) == "1":
            self.p = 1
            self._q = np.inf
        elif str(p) == "2":
            self.p = 2
            self._q = 2
        elif p == "inf" or p == np.inf:
            self.p = np.inf
            self._q = 1
        else:
            raise ValueError("Unsupported norm: %s" % str(p))

    @property
    def q(self):
        """Holder conjugate norm."""
        return self._q

    def __repr__(self):
        return "LpSpace(p={})".format(self.p)
