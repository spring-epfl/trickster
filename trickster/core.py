from functools import partial

import numpy as np

from .oracle import SaliencyOracle


class AlgoSetup(object):
    @staticmethod
    def gen_mutations(trickster, example):
        raise NotImplementedError

    @staticmethod
    def score(trickster, example):
        raise NotImplementedError


class Trickster(object):
    '''
    Generate adversarial examples for a given model
    '''

    def __init__(self, model, method='evo', algo_setup=None):
        '''
        :param model: Keras model
        :param method: One of ['evo', 'beam']
        '''
        self._model = model
        self.method = method
        if algo_setup is None:
            algo_setup = AlgoSetup()
        self.algo_setup = algo_setup

    def perturb(self, example, target_class):
        X = np.expand_dims(example, axis=0)
        true_class = np.squeeze(self._model.predict_classes(X, verbose=False))
        if target_class == true_class:
            return example

        oracle = SaliencyOracle(self._model, target_class)
        if self.method == 'evo':
            self._perturb_evolutionary(oracle, example, target_class)

    def _perturb_evolutionary(self, oracle, example, target_class):
        candidates = self.algo_setup.gen_mutations(self, example)
        candidates = sort(candidates,
                key=partial(self.algo_setup.score, self))

