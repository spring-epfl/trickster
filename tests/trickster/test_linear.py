import pytest

import numpy as np
import scipy as sp

from sklearn.utils import check_array

from trickster.optim import CategoricalLpProblemContext
from trickster.linear import LinearHeuristic


class FakeModel:
    """
    Linear model with discrimiant f(x, y) = 2x - y + 3

    >>> f = FakeModel()
    >>> X = [[1, 0], [0, 0], [0, 1]]
    >>> f.decision_function(X)
    array([5, 3, 2])
    >>> f.predict_proba(X).shape
    (3, 2)
    """

    weights = np.array([2, -1])
    bias = 3

    def decision_function(self, x):
        x = check_array(x)
        return np.dot(self.weights, x.T) + self.bias

    def predict_proba(self, x):
        x = check_array(x)
        p = np.expand_dims(sp.special.expit(self.decision_function(x)), -1)
        return np.hstack([1 - p, p])

    def grad(self, x, target_class=None):
        return target_class * np.array([2, -1])


@pytest.fixture(scope="function")
def problem_ctx():
    ctx = CategoricalLpProblemContext(
        clf=FakeModel(), target_class=1, target_confidence=0.5, lp_space=1
    )
    return ctx


def test_heuristic_target_side(problem_ctx):
    h = LinearHeuristic(problem_ctx)
    assert h([1, 0]) == 0


def test_heuristic_source_side(problem_ctx):
    h = LinearHeuristic(problem_ctx)

    # f([0, 6]) = -3, ||grad(f)|| = ||[2, -1]|| = 2  (inf-norm)
    # h = |-3| / 2 = 1.5
    assert h([0, 6]) == 1.5


def test_heuristic_custom_target(problem_ctx):
    problem_ctx.target_confidence = 0.95

    h = LinearHeuristic(problem_ctx)

    # Further away from the boundary than with 0.5 threshold
    assert h([0, 6]) > 2.9
