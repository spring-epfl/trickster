import random
import pytest
import numpy as np

from trickster.oracle import SaliencyOracle

random.seed(42)


TEST_SAMPLES = 20


def test_featurewise_oracle_shape(model_data):
    X, Y, model = model_data
    examples = X[:TEST_SAMPLES, ]
    saliency = SaliencyOracle(model, target_class=1).eval(examples)

    # examples x features
    assert saliency.shape == (examples.shape[0],) + X.shape[1:]


def test_aggregated_oracle_shape(model_data):
    X, Y, model = model_data
    examples = X[:TEST_SAMPLES, ]
    saliency = SaliencyOracle(model, target_class=1) \
            .eval(examples, featurewise=False)

    assert saliency.shape == (examples.shape[0],)


def test_saliency_complementarity(model_data):
    X, Y, model = model_data
    m = Y.shape[-1]
    # choose some examples
    indices = [random.randrange(X.shape[0]) for _ in range(TEST_SAMPLES)]
    for i in indices:
        x, c = X[i], np.argmax(Y[i])
        # choose a target t != c
        t = random.choice([j for j in range(m) if j != c])
        check_single_example_complementarity(model, x, c, t)


def check_single_example_complementarity(model, x, c, t):
    oracle1 = SaliencyOracle(model, target_class=c)
    saliency1, = oracle1.eval([x])

    oracle2 = SaliencyOracle(model, target_class=t)
    saliency2, = oracle2.eval([x])

    intersect = (saliency1 > 0) & (saliency2 > 0)
    assert np.mean(intersect) < \
            max(np.mean(saliency1 > 0), np.mean(saliency2 > 0))

