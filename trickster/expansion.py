import numpy as np


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
    :type sample: numpy array.
    :param feat_idxs: Indexes pointing to transformable features in the sample array.
    :type feat_idxs: numpy array or list of ints.
    :returns: list of numpy arrays.
    """
    children = []

    children.extend(expand_collection_set(sample, feat_idxs))
    children.extend(expand_collection_unset(sample, feat_idxs))

    return children


def expand(sample, expansions):
    """
    Convenience function to perform above expansions.

    :param sample: Initial node.
    :type sample: numpy array.
    :param expansions: List of expansion procedures. Each expansion procedure
                is a tuple consiting of indexes pointing to transformable features
                and an expansion function (pre-defined or custom).
    :returns: list of numpy arrays.
    """
    children = []

    for feat_idxs, expansion_fn in expansions:
        children.extend(expansion_fn(sample, feat_idxs))

    return children
