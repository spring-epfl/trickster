import pytest

from trickster.core import Trickster


def test_trickster_no_setup(model_data):
    X, Y, model = model_data
    x = X[42, ]
    adv = Trickster(model)
    with pytest.raises(NotImplementedError):
        adv.perturb(x, target_class=2)
