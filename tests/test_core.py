import os
import pytest

from keras.models import load_model
from trickster.core import SaliencyOracle

from fixtures.train_iris import get_data as get_iris_data
from fixtures.train_news import get_data as get_news_data


FIXTURE_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'fixtures')


def get_model_fixture(name='iris'):
    if name == 'iris':
        data = get_iris_data()
    if name == 'news':
        data = get_news_data()

    model = load_model(os.path.join(FIXTURE_PATH,
        '{name}_model.keras'.format(name=name)))
    return data, model


@pytest.mark.parametrize('data,model', [
        get_model_fixture('iris'),
        get_model_fixture('news')
    ])
def test_saliency_oracle(data, model):
    X, y = data
    oracle = SaliencyOracle(model, target_class=0)
    examples = X[:5, ]
    val = oracle.eval(examples)

    # examples x classes x features
    assert val.shape == (examples.shape[0],) + X.shape[1:]
