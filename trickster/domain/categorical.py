"""
Transformations and stats for quantized, categorical, and collection features.
"""

import attr
import typing
import numpy as np

from trickster.utils.counter import ExpansionCounter, CounterLimitExceededError


def expand_quantized_increment(sample, feat_idxs):
    '''
    Get the neighbouring value (shift '1' right) in a quantized one-hot feature vector."""

    :param sample: Initial node.
    :type sample: numpy array.
    :param feat_idxs: Indexes pointing to transformable features in the sample array.
    :type feat_idxs: numpy array or list of ints.
    :returns: list of numpy arrays.
    '''
    if len(feat_idxs) == 0:
        return []

    sub_sample = sample[feat_idxs]
    idx = np.argmax(sub_sample)
    children = []

    if idx != len(sub_sample) - 1:
        child = np.array(sample)
        child[feat_idxs] = np.roll(sub_sample, 1)
        children.append(child)

    return children


def expand_quantized_decrement(sample, feat_idxs):
    """
    Get the neighbouring value (shift '1' left) in a quantized one-hot feature vector.

    :param sample: Initial node.
    :type sample: numpy array.
    :param feat_idxs: Indexes pointing to transformable features in the sample array.
    :type feat_idxs: numpy array or list of ints.
    :returns: list of numpy arrays.
    """
    if len(feat_idxs) == 0:
        return []

    sub_sample = sample[feat_idxs]
    idx = np.argmax(sub_sample)
    children = []

    if idx != 0:
        child = np.array(sample)
        child[feat_idxs] = np.roll(sub_sample, -1)
        children.append(child)

    return children


def expand_quantized(sample, feat_idxs):
    """
    Get the neighbouring value (shift '1' right and left) in a quantized one-hot feature vector.

    :param sample: Initial node.
    :type sample: numpy array.
    :param feat_idxs: Indexes pointing to transformable features in the sample array.
    :type feat_idxs: numpy array or list of ints.
    :returns: list of numpy arrays.
    """
    children = []

    children.extend(expand_quantized_increment(sample, feat_idxs))
    children.extend(expand_quantized_decrement(sample, feat_idxs))

    return children


def expand_categorical(sample, feat_idxs):
    """
    Expand all values of a single categorical feature.

    :param sample: Initial node.
    :type sample: numpy array.
    :param feat_idxs: Indexes pointing to transformable features in the sample array.
    :type feat_idxs: numpy array or list of ints.
    :returns: list of numpy arrays.
    """
    sub_sample = sample[feat_idxs]
    idx = np.argmax(sub_sample)
    children = []

    for i in range(1, len(feat_idxs)):
        child = np.array(sample)
        child[feat_idxs] = np.roll(sub_sample, i)
        children.append(child)

    return children


def expand_collection_set(sample, feat_idxs):
    """
    Expand all values of a collection of categorical features (set [0,1] to [1,0]).

    :param sample: Initial node.
    :type sample: numpy array.
    :param feat_idxs: Indexes pointing to transformable features in the sample array.
    :type feat_idxs: numpy array or list of ints.
    :returns: list of numpy arrays.
    """
    children = []

    for i, idx in enumerate(feat_idxs):

        # Skip 'absence' features.
        if i % 2 != 0:
            continue

        if sample[idx] == 0:
            child = np.array(sample)
            child[idx] = 1
            child[idx + 1] = 0
            children.append(child)

    return children


def expand_collection_unset(sample, feat_idxs):
    """
    Expand all values of a collection of categorical features (unset [1,0] to [0,1]).

    :param sample: Initial node.
    :type sample: numpy array.
    :param feat_idxs: Indexes pointing to transformable features in the sample array.
    :type feat_idxs: numpy array or list of ints.
    :returns: list of numpy arrays.
    """
    children = []

    for i, idx in enumerate(feat_idxs):

        # Skip 'presence' features.
        if i % 2 == 0:
            continue

        if sample[idx] == 0:
            child = np.array(sample)
            child[idx] = 1
            child[idx - 1] = 0
            children.append(child)

    return children


def expand_collection(sample, feat_idxs):
    """
    Expand all values of a collection of categorical features (set and unset).

    :param sample: Initial node.
    :type sample: Numpy array.
    :param feat_idxs: Indexes pointing to transformable features in the sample array.
    :type feat_idxs: Numpy array or list of ints.
    :returns: List of numpy arrays.
    """
    children = []

    children.extend(expand_collection_set(sample, feat_idxs))
    children.extend(expand_collection_unset(sample, feat_idxs))

    return children


@attr.s(auto_attribs=True)
class FeatureExpansionSpec:
    """Categorical feature expansion specification.

    :param idxs: Indexes in a feature vector that correspond to this feature.
    :param expand_fn: Expansion funciton.
    :param feature_name: Feature name.
    """

    idxs: typing.List[int]
    expand_fn: typing.Callable
    feature_name: str = None


def expand(sample, expansion_specs):
    """
    Perform multiple expansions.

    :param sample: Initial node.
    :type sample: Numpy array.
    :param expansion_specs: List of :py:class:`FeatureExpansionSpec` object.
    :returns: List of numpy arrays.
    """
    children = []
    for spec in expansion_specs:
        children.extend(spec.expand_fn(sample, spec.idxs))

    return children


@attr.s(auto_attribs=True)
class Node:
    """Node in a transformation graph.

    :param x: `Raw` example.
    :param feature_extract_fn: Feature extraction funcion.
    :param depth: Number of hops from the original example.
    """

    src: typing.List
    depth: int = 0
    feature_extract_fn: typing.Callable = None

    @property
    def features(self):
        """Return the feature vector."""
        if hasattr(self, "_features"):
            return self._features

        if self.feature_extract_fn is None:
            return self.src
        else:
            self._features = self.feature_extract_fn(self.src)
            return self._features

    def expand(self, expansion_specs):
        """Return the expanded neighbour nodes."""
        children = []

        # Increment the counter of expanded nodes.
        counter = ExpansionCounter.get_default()
        counter.increment()

        for child in expand(self.src, expansion_specs):
            children.append(
                self.__class__(
                    src=child, depth=self.depth + 1, feature_extract_fn=self.feature_extract_fn
                )
            )

        return children

    def __eq__(self, other):
        return self.src == other.src


def get_feature_coef_importance(X, clf, transformable_feature_idxs):
    """
    Get the most important features from the transformable feature set
    based on classifier parameters.
    """
    if issparse(X):
        X = X[:, transformable_feature_idxs].toarray()
    else:
        X = X[:, transformable_feature_idxs]

    importance = np.std(X, 0) * np.abs(clf.coef_[0][transformable_feature_idxs])
    imps_sum = np.sum(importance)
    importance_coef = [
        (idx, imp / imps_sum)
        for idx, imp in zip(transformable_feature_idxs, importance)
    ]
    return sorted(importance_coef, key=lambda x: x[1], reverse=True)


def get_feature_diff_importance(difference, transformable_feature_idxs):
    """
    Get the most important features from the transformable feature set
    based on the feature difference between the initial and adversarial example.
    """
    difference = CollectionsCounter(
        [item for sublist in difference for item in sublist]
    )
    cnts_sum = np.sum([v for k, v in difference.items()])
    importance_diff = [(idx, cnt / cnts_sum) for idx, cnt in difference.items()]
    importance_diff += [
        (idx, 0) for idx in transformable_feature_idxs if idx not in difference.keys()
    ]
    return sorted(importance_diff, key=lambda x: x[1], reverse=True)
