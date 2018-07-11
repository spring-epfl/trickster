import pytest

from trickster.expansion import *
from trickster.utils.expansion import *

@pytest.mark.parametrize("a, feat_idxs, expected", TEST_DATA_EXPAND_QUANTIZED_INCREMENT)
def test_expand_quantized_increment(a, feat_idxs, expected):

    children = expand_quantized_increment(a, feat_idxs)
    assert children == expected

@pytest.mark.parametrize("a, feat_idxs, expected", TEST_DATA_EXPAND_QUANTIZED_DECREMENT)
def test_expand_quantized_decrement(a, feat_idxs, expected):

    children = expand_quantized_decrement(a, feat_idxs)
    assert children == expected

@pytest.mark.parametrize("a, feat_idxs, expected", TEST_DATA_EXPAND_QUANTIZED)
def test_expand_quantized(a, feat_idxs, expected):

    children = expand_quantized(a, feat_idxs)
    assert children == expected

@pytest.mark.parametrize("a, feat_idxs, expected", TEST_DATA_EXPAND_CATEGORICAL)
def test_expand_categorical(a, feat_idxs, expected):

    children = expand_categorical(a, feat_idxs)
    assert children == expected

@pytest.mark.parametrize("a, feat_idxs, expected", TEST_DATA_EXPAND_COLLECTION_SET)
def test_expand_collection_set(a, feat_idxs, expected):

    children = expand_collection_set(a, feat_idxs)
    assert children == expected

@pytest.mark.parametrize("a, feat_idxs, expected", TEST_DATA_EXPAND_COLLECTION_RESET)
def test_expand_collection_reset(a, feat_idxs, expected):

    children = expand_collection_reset(a, feat_idxs)
    assert children == expected

@pytest.mark.parametrize("a, feat_idxs, expected", TEST_DATA_EXPAND_COLLECTION)
def test_expand_collection(a, feat_idxs, expected):

    children = expand_collection(a, feat_idxs)
    assert children == expected

@pytest.mark.parametrize("a, feat_idxs, expected", TEST_DATA_EXPAND)
def test_expand(a, feat_idxs, expected):
    
    expansions = [
        (feat_idxs, expand_categorical),
        (feat_idxs, expand_collection)
    ]
    
    children = expand(a, expansions)
    assert children == expected
