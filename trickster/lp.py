r"""
Settings for an optimal (or $\varepsilon$-suboptimal) A* adversarial example search instance.
"""

import typing

import attr
import numpy as np


class LpSpace:
    """$L_p$ space representation.

    :param p: Norm of the space
    :param q: Holder conjugate norm

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
        return self._q

    def __repr__(self):
        return "LpSpace(p={})".format(self.p)


@attr.s(auto_attribs=True)
class LpProblemContext:
    """Context for a search problem in Lp space.

    :param clf: Target classifier.
    :param target_class: Target class.
    :param target_confidence: Target class confidence.
    :param LpSpace $L_p$ space.
    :param epsilon: Epsilon for sub-optimal search.

    >>> problem_ctx = LpProblemContext( \
            clf="stub", target_class=1, target_confidence=0.5, \
            lp_space="inf", epsilon=10)

    """
    clf: typing.Any
    target_class: float
    target_confidence: float = 0.5
    epsilon: float = 1.0
    lp_space: LpSpace = attr.ib(default=1, converter=LpSpace)
