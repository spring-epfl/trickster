from functools import partial

import numpy as np

from .oracle import SaliencyOracle
from .utils import generalized_graph_search


class DomainSpecs(object):
    @staticmethod
    def candidate_generator(example, saliency_oracle=None):
        raise NotImplementedError


def default_score_fn(saliency_oracle, original, candidate,
                     gscore=None, rank=0):
    original_saliency, = saliency_oracle.eval([original])
    candidate_saliency, = saliency_oracle.eval([candidate])
    return (original_saliency - candidate_saliency) / (rank + 1)


class Trickster(object):
    '''
    Generate adversarial examples for a given model
    '''
    def __init__(self, model, domain_specs=None, score_fn=None):
        '''
        :param model: Keras model.
        :param domain_specs: DomainSpecs object.
        :param score_fn: f-score function. See ``utils`` for more details.
                The smaller the value, the "better" the example (smaller
                perturbation yet large influence towards target class).

        '''
        self._model = model
        if domain_specs is None:
            domain_specs = DomainSpecs()
        self.domain_specs = domain_specs
        self.score_fn = score_fn or default_score_fn

    def perturb(self, example, target_class, opt_args=None):
        '''
        Perturb example until adversarial goal is met.

        :param example: Example to perturb
        :param target_class: Target class
        '''
        default_opt_args = {
            'iter_lim': 1000000,
            'beam_size': 500
        }
        default_opt_args.update(opt_args or {})
        opt_args = default_opt_args

        def _goal_predicate(x):
            dummy_batch = np.array([x])
            yhat = self._model.predict(dummy_batch)[0]
            return np.argmax(yhat) == target_class

        saliency_oracle = SaliencyOracle(self._model, target_class=target_class)
        result = generalized_graph_search(
            example,
            generate_neighbours_fn=self.domain_specs.candidate_generator,
            fscore_fn=partial(self.score_fn, saliency_oracle),
            goal_predicate=_goal_predicate,
            **opt_args,
        )
        if result is None:
            raise ValueError('Unable to find an adversarial example.')

        fscore, adv_example = result
        return adv_example
