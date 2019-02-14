"""
Everything needed to run the experiments in the optimal setting.
"""

# Ignore warnings.
import warnings

warnings.filterwarnings("ignore")

import logging

import attr
import numpy as np
import pandas as pd
import typing

from scipy.sparse import issparse

from tqdm import tqdm
from collections import Counter as CollectionsCounter
from profiled import Profiler, profiled

from trickster import linear
from trickster.search import a_star_search
from trickster.base import ProblemContext, GraphSearchProblem, WithProblemContext
from trickster.domain.categorical import FeatureExpansionSpec
from trickster.domain.categorical import Node
from trickster.utils.lp import LpSpace
from trickster.utils.log import LOGGER_NAME
from trickster.utils.log import setup_custom_logger
from trickster.utils.counter import ExpansionCounter
from trickster.utils.counter import CounterLimitExceededError


@attr.s(auto_attribs=True)
class CategoricalLpProblemContext(ProblemContext):
    """Context for an optimal search instance in Lp space with categorical transformations.

    :param clf: Target classifier.
    :param target_class: Target class.
    :param target_confidence: Target class confidence.
    :param LpSpace lp_space: $$L_p$$ space.
    :param epsilon: Runtime-optimality trade-off parameter. 1 means optimal.
            More is sub-optimal but faster.
    :param expansion_specs: List of categorical expansion specs
            (:py:class:`trickster.domain.categorical.FeatureExpansionSpec`)

    >>> problem_ctx = CategoricalLpProblemContext(
    ...     clf="stub", target_class=1, target_confidence=0.5,
    ...     lp_space="inf", epsilon=10)

    """

    clf: typing.Any
    target_class: float
    target_confidence: float = 0.5
    epsilon: float = 1.0
    lp_space: LpSpace = attr.ib(default=1, converter=LpSpace)
    expansion_specs: typing.List[FeatureExpansionSpec] = attr.Factory(list)

    def get_graph_search_problem(self):
        problem_ctx = self
        expand_fn = SpecExpandFunc(problem_ctx=problem_ctx)
        goal_fn = GoalFunc(problem_ctx=problem_ctx)

        raw_heuristic = linear.LinearHeuristic(problem_ctx=problem_ctx)
        heuristic_fn = lambda x: raw_heuristic(x.features)

        bench_cost_fn = BenchCost(problem_ctx=problem_ctx)
        hash_fn = _default_hash_fn

        return GraphSearchProblem(
            search_fn=a_star_search,
            expand_fn=expand_fn,
            heuristic_fn=heuristic_fn,
            goal_fn=goal_fn,
            hash_fn=hash_fn,
            bench_cost_fn=bench_cost_fn,
        )


class SpecExpandFunc(WithProblemContext):
    """
    Expand candidates using given specs with each transformation having $L_p$ cost.
    """

    @profiled
    def __call__(self, x):
        children = x.expand(self.problem_ctx.expansion_specs)
        costs = [
            np.linalg.norm(x.features - c.features, ord=self.problem_ctx.lp_space.p)
            for c in children
        ]

        return list(zip(children, costs))


class GoalFunc(WithProblemContext):
    """Tells whether an example flips the decision of the classifier."""

    @profiled
    def __call__(self, x):
        return (
            self.problem_ctx.clf.predict_proba([x.features])[
                0, self.problem_ctx.target_class
            ]
            >= self.problem_ctx.target_confidence
        )


class BenchCost(WithProblemContext):
    """Alternative cost function used for analyses and stats."""

    problem_ctx = attr.ib()

    @profiled
    def __call__(self, x, another):
        return np.linalg.norm(
            x.features - another.features, ord=self.problem_ctx.lp_space.p
        )


@profiled
def _default_hash_fn(x):
    """Hash function for examples."""
    return hash(x.src.tostring())


@profiled
def _find_adversarial_example(initial_example_node, graph_search_problem, **kwargs):
    """Run the graph search procedure for a single example

    :param initial_example_node: Graph node corresponding to the initial examplean
    :param GraphSearchProblem graph_search_problem: Graph search instance specification.
    """
    return graph_search_problem.search_fn(
        initial_example_node,
        expand_fn=graph_search_problem.expand_fn,
        goal_fn=graph_search_problem.goal_fn,
        heuristic_fn=graph_search_problem.heuristic_fn,
        hash_fn=graph_search_problem.hash_fn,
        **kwargs,
    )


@profiled
def _dataset_find_adversarial_examples(
    data,
    idxs,
    problem_ctx,
    transformable_feature_idxs=None,
    reduce_classifier=True,
    counter_kwargs=None,
    graph_search_kwargs=None,
):
    """Find adversarial examples for specified indexes, and record statistics for reporting.

    :param data: Iterable of examples.
    :param idxs: Indices of examples for which the search will be run.
    :param problem_ctx: Problem context.
    :param reduce_classifier: Whether to use a reduced linear classifier.
    :param transformable_feature_idxs: Indices of features that can be transformed.
    :param counter_kwargs: Parameters passed to the :py:class:`trickster.utils.ExpansionCounter`.
    :param graph_search_kwargs: Parameters passed to the search function call.

    """
    logger = logging.getLogger(LOGGER_NAME)
    counter_kwargs = counter_kwargs or {}
    graph_search_kwargs = graph_search_kwargs or {}
    get_node_fn = lambda x: Node(src=x)

    # Dataframe for storing the results.
    results = pd.DataFrame(
        columns=[
            "dataset_index",
            "found",
            "expansion_specs",
            "x_features",
            "init_confidence",
            "x_adv_features",
            "adv_confidence",
            "bench_cost",
            "path_cost",
            "path",
            "nodes_expanded",
            "runtime",
        ]
    )

    for i, idx in enumerate(tqdm(idxs, ascii=True)):
        logger.debug(
            "Searching for adversarial example {}/{} using initial example at index: {}.".format(
                i, len(idxs), idx
            )
        )

        orig_example = data[idx]
        if issparse(orig_example):
            orig_example = orig_example.toarray()

        # If transformable_feature_idxs is specified, transform x and the classifier to
        # operate in the reduced feature space. See create_reduced_linear_classifier
        # for details.
        orig_problem_ctx = problem_ctx
        orig_clf = orig_problem_ctx.clf

        if transformable_feature_idxs is not None and reduce_classifier:
            clf = linear.create_reduced_linear_classifier(
                orig_clf, orig_example, transformable_feature_idxs
            )
            example = orig_example[transformable_feature_idxs]
            problem_ctx = attr.evolve(orig_problem_ctx, clf=clf)

        else:
            example = orig_example
            problem_ctx = orig_problem_ctx
            clf = orig_clf

        # Instantiate a counter for expanded nodes, and a profiler.
        expanded_counter = ExpansionCounter(**counter_kwargs)
        ExpansionCounter.set_global_default(expanded_counter)
        per_example_profiler = Profiler()
        Profiler.set_global_default(per_example_profiler)

        x_adv = None
        x_adv_found = None
        x_adv_reduced = None
        adv_confidence = None
        bench_cost = None
        path_cost = None
        runtime = None
        path = None

        # Run the search.
        graph_search_problem = problem_ctx.get_graph_search_problem()
        try:
            wrapped_x_adv, path_costs, path = _find_adversarial_example(
                initial_example_node=get_node_fn(example),
                graph_search_problem=graph_search_problem,
                return_path=True,
                **graph_search_kwargs,
            )
            if wrapped_x_adv is None:
                x_adv_found = False
            else:
                x_adv_found = True
                path_cost = path_costs[graph_search_problem.hash_fn(wrapped_x_adv)]

        except CounterLimitExceededError as e:
            logger.debug("For example at index {}: {}".format(idx, e))

        # Record some basic statistics and info.
        # - Number of node expansions.
        nodes_expanded = expanded_counter.count

        # - Initial confidence.
        init_confidence = clf.predict_proba([example])[0, problem_ctx.target_class]

        # - Expansion functions used.
        expansion_specs_repr = [
            (s.feature_name, s.idxs, s.expand_fn.__name__)
            for s in problem_ctx.expansion_specs
        ]

        # - Runtime statistics.
        runtime_stats = per_example_profiler.compute_stats()
        if "_find_adversarial_example" in runtime_stats:
            # Total time spent in the `_find_adversarial_example` function.
            runtime = runtime_stats["_find_adversarial_example"]["tot"]

        if x_adv_found:
            logger.debug(
                "Adversarial example {}/{} found from the initial index: {}!".format(
                    i, len(idxs), idx
                )
            )
            # Reconstruct the actual adversarial example.
            # TODO: Figure out how to handle wrapped_x_adv in a generic way.
            if transformable_feature_idxs is not None and reduce_classifier:
                temp_x_adv = get_node_fn(np.array(orig_example))
                temp_x_adv.src[transformable_feature_idxs] = wrapped_x_adv.src
                wrapped_x_adv = temp_x_adv

            # Confidence on an adversarial examples.
            adv_confidence = orig_problem_ctx.clf.predict_proba(
                [wrapped_x_adv.features]
            )[0, orig_problem_ctx.target_class]
            # Benchmark cost.
            bench_cost = graph_search_problem.bench_cost_fn(
                get_node_fn(orig_example), wrapped_x_adv
            )

            results.loc[i] = {
                "dataset_index": idx,
                "found": x_adv_found,
                "expansion_specs": expansion_specs_repr,
                "x_features": orig_example,
                "init_confidence": init_confidence,
                "x_adv_features": wrapped_x_adv.features,
                "adv_confidence": adv_confidence,
                "bench_cost": bench_cost,
                "path_cost": path_cost,
                "path": path,
                "nodes_expanded": nodes_expanded,
                "runtime": runtime,
            }

        else:
            results.loc[i] = {
                "dataset_index": idx,
                "found": x_adv_found,
                "expansion_specs": expansion_specs_repr,
                "x_features": orig_example,
                "init_confidence": init_confidence,
                "x_adv_features": None,
                "adv_confidence": None,
                "bench_cost": None,
                "path_cost": None,
                "path": None,
                "nodes_expanded": nodes_expanded,
                "runtime": runtime,
            }

        problem_ctx = orig_problem_ctx

    return results


def run_experiment(
    problem_ctx,
    data,
    confidence_margin=1.0,
    reduce_classifier=True,
    transformable_feature_idxs=None,
    make_graph_search_problem=None,
    graph_search_kwargs=None,
    logger=None,
):
    """
    Experiment runner for optimal setting in categorical feature space.

    An experiment finds adversarial examples for a given model and a given dataset.

    :param trickster.base.ProblemContext problem_ctx: Search problem context.
    :param data: Data tuple (X, y)
    :param float confidence_margin: Pick initial examples that are at most this far away from the
            target confidence. Use this to get to relax the problem.
    :param bool reduce_classifier: Whether to use :py:func:`trickster.linear.create_reduced_linear_classifier` if possible.
    :param list transformable_feature_idxs:
    :param dict graph_search_kwargs: Parameters passed to the search function call.
    :param logger: Logger instance.
    """
    logger = logger or setup_custom_logger()
    graph_search_kwargs = graph_search_kwargs or {}

    X, y = data

    # Estimate contribution of each (transformable) feature to the classifier performance.
    # logger.debug(
    #     "Computing importance of each feature based on the classifier parameters."
    # )
    # importance_coef = get_feature_coef_importance(X, clf, transformable_feature_idxs)
    importance_coef = None

    # Indices of examples in the original class that are within a margin.
    preds = problem_ctx.clf.predict_proba(X)[:, problem_ctx.target_class]
    idxs, = np.where(
        (preds < problem_ctx.target_confidence)
        & (preds >= (problem_ctx.target_confidence - confidence_margin))
    )

    # Perform adversarial example graph search.
    logger.debug(
        "Searching for adversarial examples for {} examples using graph search...".format(
            len(idxs)
        )
    )

    search_results = _dataset_find_adversarial_examples(
        data=X,
        idxs=idxs,
        problem_ctx=problem_ctx,
        reduce_classifier=reduce_classifier,
        graph_search_kwargs=graph_search_kwargs,
        transformable_feature_idxs=transformable_feature_idxs,
    )

    # Compute feature importance based on the count of feature transformations.
    # logger.debug(
    #     'Computing importance of each feature based on difference between "x" and adversarial "x".'
    # )
    # importance_diff = get_feature_diff_importance(search_results['difference'], transformable_feature_idxs)
    importance_diff = None

    # Compute score.
    score = problem_ctx.clf.score(X, y)

    # Output result.
    result = {
        "feature_count": X.shape[1],
        "transformable_feature_idxs": transformable_feature_idxs,
        "classifier": problem_ctx.clf,
        "clf_score": score,
        "coef_importance": importance_coef,
        "diff_importance": importance_diff,
        "target_confidence": problem_ctx.target_confidence,
        "confidence_margin": confidence_margin,
        "success_rates": search_results["found"].mean(),
        "avg_init_confidences": search_results["init_confidence"].mean(),
        "avg_adv_confidences": search_results["adv_confidence"].mean(),
        "avg_path_costs": search_results["path_cost"].mean(),
        "avg_bench_cost": search_results["bench_cost"].mean(),
        "avg_counter": search_results["nodes_expanded"].mean(),
        "avg_runtime": search_results["runtime"].mean(),
        "search_results": search_results,
    }

    return result
