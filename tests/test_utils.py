import pytest
import numpy as np

from trickster.utils import fast_hash


def test_fast_hash_deterministic():
    assert fast_hash('whatever') == fast_hash('whatever')


def test_fast_hash_works_with_numpy():
    array = np.arange(10)
    assert fast_hash(array) is not None

