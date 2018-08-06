import pytest
import numpy as np

from trickster.expansion import *

A = np.array([0, 0, 0, 1, 0, 0, 0])

FEAT_IDXS = [[3], [2, 3, 4], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6]]


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
        (A, FEAT_IDXS[0], []),
        (A, FEAT_IDXS[1], [[0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0]]),
        (
            A,
            FEAT_IDXS[2],
            [
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
            ],
        ),
        (
            A,
            FEAT_IDXS[3],
            [
                [1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1],
            ],
        ),
    ],
)
def test_expand_collection_set(a, feat_idxs, expected):
    children = expand_collection_set(a, feat_idxs)
    assert np.array_equal(np.array(children), np.array(expected))


@pytest.mark.parametrize(
    "a, feat_idxs, expected",
    [
        (A, FEAT_IDXS[0], [[0, 0, 0, 0, 0, 0, 0]]),
        (A, FEAT_IDXS[1], [[0, 0, 0, 0, 0, 0, 0]]),
        (A, FEAT_IDXS[2], [[0, 0, 0, 0, 0, 0, 0]]),
        (A, FEAT_IDXS[3], [[0, 0, 0, 0, 0, 0, 0]]),
    ],
)
def test_expand_collection_reset(a, feat_idxs, expected):
    children = expand_collection_reset(a, feat_idxs)
    assert np.array_equal(np.array(children), np.array(expected))


@pytest.mark.parametrize(
    "a, feat_idxs, expected",
    [
        (A, FEAT_IDXS[0], [[0, 0, 0, 0, 0, 0, 0]]),
        (
            A,
            FEAT_IDXS[1],
            [[0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
        ),
        (
            A,
            FEAT_IDXS[2],
            [
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ),
        (
            A,
            FEAT_IDXS[3],
            [
                [1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ),
    ],
)
def test_expand_collection(a, feat_idxs, expected):
    children = expand_collection(a, feat_idxs)
    assert np.array_equal(np.array(children), np.array(expected))


@pytest.mark.parametrize(
    "a, feat_idxs, expected",
    [
        (A, FEAT_IDXS[0], [[0, 0, 0, 0, 0, 0, 0]]),
        (
            A,
            FEAT_IDXS[1],
            [
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ),
        (
            A,
            FEAT_IDXS[2],
            [
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
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
                [1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ),
    ],
)
def test_expand(a, feat_idxs, expected):
    expansions = [(feat_idxs, expand_categorical), (feat_idxs, expand_collection)]

    children = expand(a, expansions)
    assert np.array_equal(np.array(children), np.array(expected))
