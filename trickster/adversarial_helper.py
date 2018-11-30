"""
Generic utils.

TODO(bogdan): Rename this and possibly split across multiple modules.
"""

# Ignore warnings.
import warnings
warnings.filterwarnings("ignore")

import logging

import attr
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from scipy import stats
from scipy.sparse import issparse

from collections import Counter as CollectionsCounter
from defaultcontext import with_default_context
from profiled import Profiler, profiled
from tqdm import tqdm

from trickster.search import a_star_search
from trickster.expansions import expand
from trickster.utils.log import LOGGER_NAME, setup_custom_logger


def get_feature_coef_importance(X, clf, transformable_feature_idxs):
    """
    Get the most important features from the transformable feature set
    based on classifier parameters.
    """
    if issparse(X):
        X = X[:, transformable_feature_idxs].toarray()
    else:
        X = X[:, transformable_feature_idxs]

    importance = np.std(X, 0) * np.abs(clf.coef_[0][transformable_feature_idxs])
    imps_sum = np.sum(importance)
    importance_coef = [
        (idx, imp / imps_sum)
        for idx, imp in zip(transformable_feature_idxs, importance)
    ]
    return sorted(importance_coef, key=lambda x: x[1], reverse=True)


def get_feature_diff_importance(difference, transformable_feature_idxs):
    """
    Get the most important features from the transformable feature set
    based on the feature difference between the initial and adversarial example.
    """
    difference = CollectionsCounter(
        [item for sublist in difference for item in sublist]
    )
    cnts_sum = np.sum([v for k, v in difference.items()])
    importance_diff = [(idx, cnt / cnts_sum) for idx, cnt in difference.items()]
    importance_diff += [
        (idx, 0) for idx in transformable_feature_idxs if idx not in difference.keys()
    ]
    return sorted(importance_diff, key=lambda x: x[1], reverse=True)


def find_substring_occurences(xs, item):
    """Can be used to get the indexes of the required substring within a list of strings.

    >>> find_substring_occurences(['ab', 'bcd', 'de'], 'd')
    [1, 2]

    """

    idxs = [i for (i, x) in enumerate(xs) if item in x]
    return idxs


def create_reduced_linear_classifier(clf, x, transformable_feature_idxs):
    r"""Construct a reduced-dimension classifier based on the original one for a given example.

    The reduced-dimension classifier should behave the same way as the original one, but operate in
    a smaller feature space. This is done by fixing the score of the classifier on a static part of
    ``x``, and integrating it into the bias parameter of the reduced classifier.

    For example, let $x = [1, 2, 3]$, weights of the classifier $w = [1, 1, 1]$ and bias term $b =
    0$, and the only transformable feature index is 0. Then the reduced classifier has weights $w' =
    [1]$, and the bias term incorporates the non-transformable part of $x$: $b' = -1 \cdot 2 + 1
    \cdot 3$.

    :param clf: Original logistic regression classifier
    :param x: An example
    :param transformable_feature_idxs: List of features that can be changed in the given example.
    """

    # Establish non-transformable feature indexes.
    feature_idxs = np.arange(x.size)
    non_transformable_feature_idxs = np.setdiff1d(
        feature_idxs, transformable_feature_idxs
    )

    # Create the reduced classifier.
    clf_reduced = LogisticRegressionCV()
    clf_reduced.coef_ = clf.coef_[:, transformable_feature_idxs]
    clf_reduced.intercept_ = np.dot(
        clf.coef_[0, non_transformable_feature_idxs], x[non_transformable_feature_idxs]
    )
    clf_reduced.intercept_ += clf.intercept_

    assert np.allclose(
        clf.predict_proba([x]),
        clf_reduced.predict_proba([x[transformable_feature_idxs]]),
    )
    return clf_reduced


class CounterLimitExceededError(Exception):
    pass


@with_default_context(use_empty_init=True)
class ExpansionCounter:
    """Context-local counter of expanded nodes."""

    def __init__(self, counter_lim=None, debug_freq=None):
        self.count = 0
        self.counter_lim = counter_lim
        self.debug_freq = debug_freq

    def increment(self):
        logger = logging.getLogger(LOGGER_NAME)
        if self.counter_lim is not None and self.count > self.counter_lim:
            raise CounterLimitExceededError(
                "Expansion counter limit {} reached.".format(self.counter_lim)
            )
        if self.debug_freq is not None and self.count % self.debug_freq == 0:
            logger.debug("Counter is: {}.".format(self.count))
        self.count += 1


class Node(object):
    """Single node in a transformation graph.

    :param src: `Raw` example
    :param feature_extract_fn: Feature extraction funcion
    :param depth: Number of hops from the original example
    """

    def __init__(self, x, depth=0, feature_extract_fn=None):
        self.src = x
        self.depth = depth
        self.feature_extract_fn = feature_extract_fn

    @property
    def features(self):
        """Return the feature vector."""
        if hasattr(self, "_features"):
            return self._features

        if self.feature_extract_fn is None:
            return self.src
        else:
            self._features = self.feature_extract_fn(self.src)
            return self._features

    def expand(self, expansions):
        """Return the expanded neighbour nodes."""
        children = []

        # Increment the counter of expanded nodes.
        counter = ExpansionCounter.get_default()
        counter.increment()

        for child in expand(self.src, expansions):
            children.append(
                self.__class__(
                    child, self.depth + 1, feature_extract_fn=self.feature_extract_fn
                )
            )

        return children

    def __repr__(self):
        return "{}(x={}, depth={})".format(
            self.__class__.__name__, self.src, self.depth
        )

    def __eq__(self, other):
        return self.src == other.src


@with_default_context
@attr.s
class AdvProblemContext:
    """Parameters for a single adversarial example search instance."""

    clf = attr.ib()
    expansions = attr.ib(default=attr.Factory(list))
    target_class = attr.ib(default=1)
    target_confidence = attr.ib(default=0.5)
    p_norm = attr.ib(default=1)
    q_norm = attr.ib(default=np.inf)
    epsilon = attr.ib(default=1)


@profiled
def default_expand_fn(x):
    """Expand x and compute the costs.
    Returns a list of tuples (child, cost)
    """
    problem_ctx = AdvProblemContext.get_default()
    children = x.expand(problem_ctx.expansions)
    costs = [np.linalg.norm(x.src - c.src, ord=problem_ctx.p_norm) for c in children]

    return list(zip(children, costs))


@profiled
def default_goal_fn(x):
    """Tell whether the example has reached the goal."""
    problem_ctx = AdvProblemContext.get_default()
    return (
        problem_ctx.clf.predict_proba([x.src])[0, problem_ctx.target_class]
        >= problem_ctx.target_confidence
    )


@profiled
def default_heuristic_fn(x):
    """Distance to the decision boundary of a logistic regression classifier.
    By default the distance is w.r.t. L1 norm. This means that the denominator
    has to be in terms of the Holder dual norm (`q_norm`), so L-inf. I know,
    this interface is horrible.
    NOTE: The value has to be zero if the example is already on the target side
    of the boundary.
    """
    problem_ctx = AdvProblemContext.get_default()
    confidence = problem_ctx.clf.predict_proba([x.src])[0, problem_ctx.target_class]
    if confidence >= problem_ctx.target_confidence:
        return 0.0
    score = problem_ctx.clf.decision_function([x.src])[0]
    h = np.abs(score) / np.linalg.norm(
        problem_ctx.clf.coef_[0], ord=problem_ctx.q_norm
    )
    return h * problem_ctx.epsilon


@profiled
def default_hash_fn(x):
    """Hash function for examples."""
    return hash(x.src.tostring())


@profiled
def default_example_wrapper_fn(x):
    """Initial example data to a search graph node."""
    return Node(x)


@profiled
def default_real_cost_fn(x, another):
    """Real cost for transforming example into another one."""
    problem_ctx = AdvProblemContext.get_default()
    return np.linalg.norm(x.src - another.src, ord=problem_ctx.p_norm)


@attr.s
class GraphSearchFuncs:
    """
    Functions that define the graph search instance.

    :param example_wrapper_fn: Is called on each node after expansion.
            If input examples are `raw`, e.g., numpy arrays, this function can be used to wrap the examples
            in a :py:class:`Node` class.
    :param expand_fn: Returns the expanded neighbour nodes for a given node.
    :param goal_fn: Predicate that tells whether a given node is a target node.
    :param heuristic_fn: Returns an estimate of how far the given example.
    :param hash_fn: Hash function for nodes.
    :param real_cost_fn: An extra cost function between an example and adv. example used for analysis and reporting.
    """

    example_wrapper_fn = attr.ib(
        default=attr.Factory(lambda: default_example_wrapper_fn)
    )
    expand_fn = attr.ib(default=attr.Factory(lambda: default_expand_fn))
    goal_fn = attr.ib(default=attr.Factory(lambda: default_goal_fn))
    heuristic_fn = attr.ib(default=attr.Factory(lambda: default_heuristic_fn))
    hash_fn = attr.ib(default=attr.Factory(lambda: default_hash_fn))
    real_cost_fn = attr.ib(default=attr.Factory(lambda: default_real_cost_fn))


@profiled
def find_adversarial_example(
    example, search_fn, graph_search_funcs, graph_search_kwargs, return_path=False
):
    """Transform an example until it is classified as target."""
    wrapped_example = graph_search_funcs.example_wrapper_fn(example)
    return search_fn(
        start_node=wrapped_example,
        expand_fn=graph_search_funcs.expand_fn,
        goal_fn=graph_search_funcs.goal_fn,
        heuristic_fn=graph_search_funcs.heuristic_fn,
        hash_fn=graph_search_funcs.hash_fn,
        return_path=return_path,
        **graph_search_kwargs
    )


@profiled
def dataset_find_adversarial_examples(
    dataset,
    idxs,
    search_fn,
    graph_search_funcs,
    reduce_classifier=True,
    transformable_feature_idxs=None,
    counter_kwargs=None,
    graph_search_kwargs=None,
):
    """Find adversarial examples for specified indexes, and record statistics for reports.

    :param dataset: Numpy datasets (X, y).
    :param idxs: Example indices.
    :param search_fn: Graph search function.
    :param reduce_classifier: Whether to use a reduced linear classifier as a substitute target.
    :param SearchFuncs graph_search_funcs: Search functions.
    :param transformable_feature_idxs: Indexes of feature that can be transformed.
    :param counter_kwargs: Parameters passed to the :py:class:`ExpansionCounter`.
    :param graph_search_kwargs: Parameters passed to the search function call.

    .. note::

        `transformable_feature_idxs` only works if the target model is logistic regression.

    """
    logger = logging.getLogger(LOGGER_NAME)
    counter_kwargs = counter_kwargs or {}
    graph_search_kwargs = graph_search_kwargs or {}

    # Dataframe for storing the results.
    results = pd.DataFrame(
        columns=[
            "dataset_index",
            "found",
            "expansions",
            "x_features",
            "init_confidence",
            "x_adv_features",
            "adv_confidence",
            "real_cost",
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

        orig_example = dataset[idx]
        if issparse(orig_example):
            orig_example = orig_example.toarray()

        # If transformable_feature_idxs is specified, transform x and the classifier to operate in the reduced feature space.
        orig_problem_ctx = AdvProblemContext.get_default()
        if transformable_feature_idxs is not None and reduce_classifier:
            clf_reduced = create_reduced_linear_classifier(
                orig_problem_ctx.clf, orig_example, transformable_feature_idxs
            )
            example = orig_example[transformable_feature_idxs]
            problem_ctx = attr.evolve(orig_problem_ctx, clf=clf_reduced)
        else:
            example = orig_example
            problem_ctx = orig_problem_ctx

        # Instantiate a counter for expanded nodes, and a profiler.
        expanded_counter = ExpansionCounter(**counter_kwargs)
        ExpansionCounter.set_global_default(expanded_counter)
        per_example_profiler = Profiler()
        Profiler.set_global_default(per_example_profiler)

        AdvProblemContext.set_global_default(problem_ctx)

        x_adv = None
        x_adv_reduced = None
        x_adv_found = None
        adv_confidence = None
        real_cost = None
        path_cost = None
        runtime = None
        path = None

        # Run the search.
        try:
            x_adv_reduced, path_costs, path = find_adversarial_example(
                example=example,
                search_fn=search_fn,
                return_path=True,
                graph_search_funcs=graph_search_funcs,
                graph_search_kwargs=graph_search_kwargs
            )
            if x_adv_reduced is None:
                x_adv_found = False
            else:
                x_adv_found = True
                path_cost = path_costs[graph_search_funcs.hash_fn(x_adv_reduced)]

        except CounterLimitExceededError as e:
            logger.debug("For example at index {}: {}".format(idx, e))

        # Record some basic statistics.
        # - Number of node expansions.
        nodes_expanded = expanded_counter.count

        # Initial confidence.
        init_confidence = clf_reduced.predict_proba([example])[
            0, problem_ctx.target_class
        ]

        # Expansion functions used.
        expansion_funcs = [(idxs, fn.__name__) for (idxs, fn) in problem_ctx.expansions]

        # Runtime statistics.
        runtime_stats = per_example_profiler.compute_stats()
        if "find_adversarial_example" in runtime_stats:
            # Total time spent in the `find_adversarial_example` function.
            runtime = runtime_stats["find_adversarial_example"]["tot"]

        if x_adv_found:
            logger.debug(
                "Adversarial example {}/{} found from the initial index: {}!".format(
                    i, len(idxs), idx
                )
            )
            # Construct the actual adversarial example.
            if transformable_feature_idxs is not None and reduce_classifier:
                x_adv = graph_search_funcs.example_wrapper_fn(np.array(orig_example))
                x_adv.src[transformable_feature_idxs] = x_adv_reduced.src
            else:
                x_adv = x_adv_reduced

            # Compute further statistics.
            adv_confidence = orig_problem_ctx.clf.predict_proba([x_adv.src])[
                0, orig_problem_ctx.target_class
            ]
            real_cost = graph_search_funcs.real_cost_fn(
                graph_search_funcs.example_wrapper_fn(orig_example), x_adv
            )

            results.loc[i] = {
                "dataset_index": idx,
                "found": x_adv_found,
                "expansions": expansion_funcs,
                "x_features": orig_example,
                "init_confidence": init_confidence,
                "x_adv_features": x_adv.src,
                "adv_confidence": adv_confidence,
                "real_cost": real_cost,
                "path_cost": path_cost,
                "path": path,
                "nodes_expanded": nodes_expanded,
                "runtime": runtime,
            }

        else:
            results.loc[i] = {
                "dataset_index": idx,
                "found": x_adv_found,
                "expansions": expansion_funcs,
                "x_features": orig_example,
                "init_confidence": init_confidence,
                "x_adv_features": None,
                "adv_confidence": None,
                "real_cost": None,
                "path_cost": None,
                "path": None,
                "nodes_expanded": nodes_expanded,
                "runtime": runtime,
            }

    return results


def experiment_wrapper(
    load_transform_data_fn,
    clf_fit_fn,
    target_class=1.0,
    target_confidence=0.5,
    confidence_margin=1.0,
    reduce_classifier=True,
    problem_params=None,
    graph_search_funcs=None,
    graph_search_kwargs=None,
    baseline_dataset_find_examples_fn=None,
    load_kwargs=None,
    get_expansions_fn=None,
    get_expansions_kwargs=None,
    clf_fit_kwargs=None,
    logger=None,
    random_state=None,
    test_size=0.1,
):
    """
    Experiment runner design to be more or less domain-independent.

    An experiment finds adversarial examples for a given model and a given dataset.

    :param load_transform_data_fn: A function that returns ``(X, y, features)``.
    :param clf_fit_fn: A function that returns a trained target model.
    :param target_class: Target class for adversarial examples.
    :param target_confidence: Target confidence level for adversarial examples.
    :param bool reduce_classifier: Whether to use :py:func:`create_reduce_linear_classifier` if possible.
    :param confidence_margin: Allows to only pick initial examples that are this far away from the target confidence.
    :param problem_params: Extra parameters for :py:class:`AdvProblemContext`.
    :param GraphSearchFuncs graph_search_funcs: Search functions.
    :param graph_search_kwargs: Extra parameters passed directly to the graph search function.
    """
    load_kwargs = load_kwargs or {}
    get_expansions_kwargs = get_expansions_kwargs or {}
    clf_fit_kwargs = clf_fit_kwargs or {}
    logger = logger or setup_custom_logger()
    graph_search_funcs = graph_search_funcs or GraphSearchFuncs()
    graph_search_kwargs = graph_search_kwargs or {}

    random_state = 1 if random_state is None else random_state
    np.random.seed(random_state)

    # Load and prepare data for learning.
    X, y, features = load_transform_data_fn(**load_kwargs)
    logger.debug("Shape of X: {}. Shape of y: {}.".format(X.shape, y.shape))

    # Get required expansions and sorted indexes of transformable features.
    if get_expansions_fn is None:
        expansions = None
        transformable_feature_idxs = None
    else:
        expansions, transformable_feature_idxs = get_expansions_fn(
            features, **get_expansions_kwargs
        )

    # Split into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.debug(
        "Number of training points: {}. Number of test points: {}.".format(
            X_train.shape[0], X_test.shape[0]
        )
    )

    # Fit and evaluate the classifier.
    clf = clf_fit_fn(X_train, y_train, **clf_fit_kwargs)
    train_score, test_score = (
        clf.score(X_train, y_train) * 100,
        clf.score(X_test, y_test) * 100,
    )
    logger.debug(
        "Resulting training accuracy is: {:.2f}%. Test accuracy is: {:.2f}%.\n".format(
            train_score, test_score
        )
    )

    # Estimate contribution of each (transformable) feature to the classifier performance.
    # logger.debug(
    #     "Computing importance of each feature based on the classifier parameters."
    # )
    # importance_coef = get_feature_coef_importance(X, clf, transformable_feature_idxs)
    importance_coef = None

    # Indices of examples in the original class that are within a margin.
    preds = clf.predict_proba(X_test)[:, target_class]
    idxs, = np.where(
        (preds < target_confidence) & (preds >= (target_confidence - confidence_margin))
    )

    # Perform adversarial example search using A* search.
    logger.debug(
        "Searching for adversarial examples for {} examples using A*...".format(
            len(idxs)
        )
    )
    problem_ctx = AdvProblemContext(
        clf=clf,
        expansions=expansions,
        target_class=target_class,
        target_confidence=target_confidence,
        **problem_params
    )
    with problem_ctx.as_default():
        search_results = dataset_find_adversarial_examples(
            dataset=X_test,
            idxs=idxs,
            transformable_feature_idxs=transformable_feature_idxs,
            search_fn=a_star_search,
            graph_search_funcs=graph_search_funcs,
            graph_search_kwargs=graph_search_kwargs
        )

    # Compute feature importance based on the count of feature transformations.
    # logger.debug(
    #     'Computing importance of each feature based on difference between "x" and adversarial "x".'
    # )
    # importance_diff = get_feature_diff_importance(search_results['difference'], transformable_feature_idxs)
    importance_diff = None

    # Perform adversarial example search using a baseline search.
    baseline_results = None
    if baseline_dataset_find_examples_fn is not None:
        logger.debug(
            "Searching for adversarial examples for {} examples using a baseline search...".format(
                len(idxs)
            )
        )
        with problem_ctx.as_default():
            baseline_results = baseline_dataset_find_examples_fn(
                dataset=X_test,
                idxs=idxs,
                transformable_feature_idxs=transformable_feature_idxs,
                search_fn=a_star_search,
                graph_search_funcs=graph_search_funcs,
            )

    # Compute average robustness of the correctly classified examples.
    correctly_classified = X_test[clf.predict(X_test) == y_test]
    scores = clf.decision_function(correctly_classified)
    grad = clf.coef_[0]
    grad_norm = np.linalg.norm(grad, ord=problem_ctx.q_norm)
    robustness = np.abs(scores) / grad_norm

    # Output result.
    result = {
        "feature_count": X.shape[1],
        "features": features,
        "transformable_feature_idxs": transformable_feature_idxs,
        "classifier": clf,
        "clf_test_score": test_score,
        "coef_importance": importance_coef,
        "diff_importance": importance_diff,
        "target_confidence": target_confidence,
        "confidence_margin": confidence_margin,
        "success_rates": search_results["found"].mean(),
        "avg_init_confidences": search_results["init_confidence"].mean(),
        "avg_adv_confidences": search_results["adv_confidence"].mean(),
        "avg_path_costs": search_results["path_cost"].mean(),
        "avg_real_cost": search_results["real_cost"].mean(),
        "avg_counter": search_results["nodes_expanded"].mean(),
        "avg_runtime": search_results["runtime"].mean(),
        "avg_robustness": robustness.mean(),
        "baseline_results": baseline_results,
        "search_results": search_results,
    }

    return result
