import os
import pytest

from keras.models import load_model

from .fixtures.train_iris import get_data as get_iris_data
from .fixtures.train_news import get_data as get_news_data


FIXTURE_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'fixtures')


@pytest.fixture(params=['iris', 'news'])
def model_data(request):
    name = request.param
    if name == 'iris':
        X, Y = get_iris_data()
    if name == 'news':
        X, Y = get_news_data()

    model = load_model(os.path.join(FIXTURE_PATH,
        '{name}_model.keras'.format(name=name)))
    return X, Y, model

