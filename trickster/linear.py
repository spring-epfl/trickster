"""
Tools for working with linear and linearized models.
"""

import attr
import numpy as np
import scipy as sp
from profiled import profiled

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression


def _get_forward_grad_lr(clf, x, target_class=None):
    return clf.coef_[0]


def _get_forward_grad_default(clf, x, target_class=None):
    return clf.grad(x, target_class)


def get_forward_grad(clf, x, target_class=None):
    """Get the forward gradient of a classifier.

    :param clf: Classifier.
    :param x: Input.
    :param target_class: Not supported.
    """

    if isinstance(clf, LogisticRegressionCV) or isinstance(clf, LogisticRegression):
        return _get_forward_grad_lr(clf, x, target_class)

    else:
        return _get_forward_grad_default(clf, x, target_class)


def create_reduced_linear_classifier(clf, x, transformable_feature_idxs):
    r"""Construct a reduced-dimension classifier based on the original one for a given example.

    The reduced-dimension classifier should behave the same way as the original one, but operate in
    a smaller feature space. This is done by fixing the score of the classifier on a static part of
    ``x``, and integrating it into the bias parameter of the reduced classifier.

    For example, let $$x = [1, 2, 3]$$, weights of the classifier $$w = [1, 1, 1]$$ and bias term $$b =
    0$$, and the only transformable feature index is 0. Then the reduced classifier has weights $$w' =
    [1]$$, and the bias term incorporates the non-transformable part of $$x$$: $$b' = -1 \cdot 2 + 1
    \cdot 3$$.

    :param clf: Original logistic regression classifier
    :param x: An example
    :param transformable_feature_idxs: List of features that can be changed in the given example.
    """

    # Establish non-transformable feature indexes.
    feature_idxs = np.arange(x.size)
    non_transformable_feature_idxs = np.setdiff1d(
        feature_idxs, transformable_feature_idxs
    )

    # Create the reduced classifier.
    clf_reduced = LogisticRegressionCV()
    clf_reduced.coef_ = clf.coef_[:, transformable_feature_idxs]
    clf_reduced.intercept_ = np.dot(
        clf.coef_[0, non_transformable_feature_idxs], x[non_transformable_feature_idxs]
    )
    clf_reduced.intercept_ += clf.intercept_

    assert np.allclose(
        clf.predict_proba([x]),
        clf_reduced.predict_proba([x[transformable_feature_idxs]]),
    )
    return clf_reduced


@attr.s
class LinearHeuristic:
    r"""$$L_p$$ distance to the decision boundary of a linear or linearized classifier.

    :param trickster.lp.LpProblemCtx problem_ctx: Problem context.
    """

    problem_ctx = attr.ib()

    @profiled
    def __call__(self, x):
        ctx = self.problem_ctx
        confidence = ctx.clf.predict_proba([x])[0, ctx.target_class]
        if confidence >= ctx.target_confidence:
            return 0.0

        score = ctx.clf.decision_function([x])[0]
        if ctx.target_confidence != 0.5:
            score -= sp.special.logit(ctx.target_confidence)

        fgrad = get_forward_grad(ctx.clf, x, target_class=ctx.target_class)
        h = np.abs(score) / np.linalg.norm(fgrad, ord=ctx.lp_space.q)
        return h * ctx.epsilon
