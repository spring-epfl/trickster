import pytest
import numpy as np

from trickster.expansions import *

# Feature vector for quantized or categorical features.
A = np.array([0, 0, 0, 1, 0, 0, 0])

# Feature indices.
FEAT_IDXS = [[3], [2, 3, 4], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6]]

# Feature vector for a collection set. Three attributes: set, unset, set.
B = np.array([1, 0, 0, 1, 1, 0])


@pytest.mark.parametrize(
    "a, feat_idxs, expected",
    [
        (A, FEAT_IDXS[0], []),
        (A, FEAT_IDXS[1], [[0, 0, 0, 0, 1, 0, 0]]),
        (A, FEAT_IDXS[2], [[0, 0, 0, 0, 1, 0, 0]]),
        (A, FEAT_IDXS[3], [[0, 0, 0, 0, 1, 0, 0]]),
    ],
)
def test_expand_quantized_increment(a, feat_idxs, expected):
    children = expand_quantized_increment(a, feat_idxs)
    assert np.array_equal(np.array(children), np.array(expected))


@pytest.mark.parametrize(
    "a, feat_idxs, expected",
    [
        (A, FEAT_IDXS[0], []),
        (A, FEAT_IDXS[1], [[0, 0, 1, 0, 0, 0, 0]]),
        (A, FEAT_IDXS[2], [[0, 0, 1, 0, 0, 0, 0]]),
        (A, FEAT_IDXS[3], [[0, 0, 1, 0, 0, 0, 0]]),
    ],
)
def test_expand_quantized_decrement(a, feat_idxs, expected):
    children = expand_quantized_decrement(a, feat_idxs)
    assert np.array_equal(np.array(children), np.array(expected))


@pytest.mark.parametrize(
    "a, feat_idxs, expected",
    [
        (A, FEAT_IDXS[0], []),
        (A, FEAT_IDXS[1], [[0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0]]),
        (A, FEAT_IDXS[2], [[0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0]]),
        (A, FEAT_IDXS[3], [[0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0]]),
    ],
)
def test_expand_quantized(a, feat_idxs, expected):
    children = expand_quantized(a, feat_idxs)
    assert np.array_equal(np.array(children), np.array(expected))


@pytest.mark.parametrize(
    "a, feat_idxs, expected",
    [
        (A, FEAT_IDXS[0], []),
        (A, FEAT_IDXS[1], [[0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0]]),
        (
            A,
            FEAT_IDXS[2],
            [
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
            ],
        ),
        (
            A,
            FEAT_IDXS[3],
            [
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
            ],
        ),
    ],
)
def test_expand_categorical(a, feat_idxs, expected):
    children = expand_categorical(a, feat_idxs)
    assert np.array_equal(np.array(children), np.array(expected))


@pytest.mark.parametrize(
    "a, feat_idxs, expected",
    [
        (B, [0], []),
        (B, [0, 1], []),
        (B, [0, 1, 2], [[1, 0, 1, 0, 1, 0]]),
        (B, [0, 1, 2, 3, 4, 5], [[1, 0, 1, 0, 1, 0]]),
    ],
)
def test_expand_collection_set(a, feat_idxs, expected):
    children = expand_collection_set(a, feat_idxs)
    assert np.array_equal(np.array(children), np.array(expected))


@pytest.mark.parametrize(
    "a, feat_idxs, expected",
    [
        (B, [0], []),
        (B, [0, 1], [[0, 1, 0, 1, 1, 0]]),
        (B, [0, 1, 2, 3, 4], [[0, 1, 0, 1, 1, 0]]),
        (B, [0, 1, 2, 3, 4, 5], [[0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1]]),
    ],
)
def test_expand_collection_unset(a, feat_idxs, expected):
    children = expand_collection_unset(a, feat_idxs)
    assert np.array_equal(np.array(children), np.array(expected))


@pytest.mark.parametrize(
    "a, feat_idxs, expected",
    [
        (B, [0], []),
        (B, [0, 1], [[0, 1, 0, 1, 1, 0]]),
        (
            B,
            [0, 1, 2, 3, 4, 5],
            [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1]],
        ),
    ],
)
def test_expand_collection(a, feat_idxs, expected):
    children = expand_collection(a, feat_idxs)
    assert np.array_equal(np.array(children), np.array(expected))
