#!/usr/bin/env python3
"""
Experiment with generating adversarial examples for a website fingerprinting classifier
implemented as a logistic regression on top of CUMUL features.
"""

import sys

sys.path.append("..")

import os
import math
import pickle
import argparse
import logging
import pprint
import random
import functools

import attr
import click
import numpy as np
import pandas as pd

from trickster.search import generalized_a_star_search
from trickster.utils.counter import ExpansionCounter
from trickster.optim import GraphSearchProblem, _find_adversarial_example
from trickster.optim import GoalFunc
from trickster.optim import LpSpace
from trickster.optim import CategoricalLpProblemContext
from trickster.linear import LinearGridHeuristic, LinearHeuristic
from trickster.base import ProblemContext, WithProblemContext
from trickster.domain.wfp import extract, pad_and_onehot, load_data
from trickster.domain.wfp import insert_dummy_packets
from trickster.utils.log import LOGGER_NAME, setup_custom_logger

from tqdm import tqdm
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV

from defaultcontext import with_default_context
from profiled import Profiler, profiled


DEBUG_PLOT_FREQ = 50


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)


def add_options(options):
    """Combine several click options in a single decorator.

    https://github.com/pallets/click/issues/108#issuecomment-255547347
    """

    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@attr.s
class Datasets:
    X_train_cell = attr.ib()
    X_test_cell = attr.ib()
    X_train_features = attr.ib()
    X_test_features = attr.ib()
    y_train = attr.ib()
    y_test = attr.ib()
    idxs_train = attr.ib()
    idxs_test = attr.ib()


def prepare_data(
    data_path,
    features="cumul",
    shuffle=False,
    max_traces=None,
    max_trace_len=None,
    filter_by_len=True,
    normalize=True,
    verbose=True,
):
    """Load a dataset and extract features."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Loading the data...")

    idxs, X, y = load_data(
        path=data_path,
        shuffle=shuffle,
        max_traces=max_traces,
        max_trace_len=max_trace_len,
        filter_by_len=filter_by_len,
        verbose=verbose,
        return_idxs=True,
    )

    logger.info("Loaded.")

    # Split into training and test sets
    idxs_train, idxs_test, X_train, X_test, y_train, y_test = train_test_split(
        idxs, X, y, test_size=0.1, random_state=1
    )

    logger.info(
        "Number of train samples: {}, Number of test samples: {}".format(
            X_train.shape[0], X_test.shape[0]
        )
    )

    logger.info("Extracting features...")

    # Extract features
    if features == "cumul":
        X_train_features = np.array([extract(trace) for trace in X_train])
        X_test_features = np.array([extract(trace) for trace in X_test])

        if normalize:
            X_train_features[:, :4] = preprocessing.scale(X_train_features[:, :4])
            X_train_features[:, 4:] = preprocessing.scale(X_train_features[:, 4:])
            X_test_features[:, :4] = preprocessing.scale(X_test_features[:, :4])
            X_test_features[:, 4:] = preprocessing.scale(X_test_features[:, 4:])

    elif features == "raw":
        pad_len, X_train_features = pad_and_onehot(X_train, pad_len=max_trace_len)
        _, X_test_features = pad_and_onehot(X_test, pad_len=pad_len)

    elif features == "total":
        X_train_features = np.array(
            [extract(trace, interpolated_features=False) for trace in X_train]
        )
        X_test_features = np.array(
            [extract(trace, interpolated_features=False) for trace in X_test]
        )

        if normalize:
            X_train_features = preprocessing.scale(X_train_features)
            X_test_features = preprocessing.scale(X_test_features)

    logger.info("Train features shape: {}".format(X_train_features.shape))
    logger.info("Test features shape: {}".format(X_test_features.shape))

    return Datasets(
        X_train_cell=X_train,
        y_train=y_train,
        X_test_cell=X_test,
        y_test=y_test,
        X_train_features=X_train_features,
        X_test_features=X_test_features,
        idxs_train=idxs_train,
        idxs_test=idxs_test,
    )


def fit_lr(datasets):
    clf = LogisticRegressionCV(Cs=21, cv=5, n_jobs=-1, penalty="l2")
    clf.fit(datasets.X_train_features, datasets.y_train)

    logger = logging.getLogger(LOGGER_NAME)
    logger.info(
        "Test score is: {:.2f}%.".format(
            clf.score(datasets.X_test_features, datasets.y_test) * 100
        )
    )

    return clf


def fit_svmrbf(datasets):
    params = {"kernel": ["rbf"], "gamma": [1e-5], "C": [100]}

    clf = GridSearchCV(SVC(probability=True), params, cv=5, scoring="accuracy")
    clf.fit(datasets.X_train_features, datasets.y_train)

    logger = logging.getLogger(LOGGER_NAME)
    logger.info(
        "Test score is: {:.3f}%.".format(
            clf.score(datasets.X_test_features, datasets.y_test) * 100
        )
    )

    return clf


class TraceNode:
    """A node in the tranformation graph of traces.

    :param trace: A trace
    :param depth: Depth in the graph
    :param features: One of ["cumul", "raw", "total"]
    :param max_len: Max trace length (to pad to)
    :param dummies_per_insertion: Number of dummies to insert for each neighbouring node.
    """

    def __init__(
        self, trace, depth=0, features="cumul", max_len=None, dummies_per_insertion=1
    ):
        self._features_type = features

        self.trace = list(trace)
        self.depth = depth
        self.max_len = max_len
        self.dummies_per_insertion = dummies_per_insertion

    @property
    @profiled
    def features(self):
        """Extract features."""
        if hasattr(self, "_features"):
            return self._features

        if self._features_type == "cumul":
            encoded_trace = extract(self.trace)

        elif self._features_type == "raw":
            _, (encoded_trace,) = pad_and_onehot([self.trace], pad_len=self.max_len)

        elif self._features_type == "total":
            encoded_trace = extract(self.trace, interpolated_features=False)

        self._features = encoded_trace
        return encoded_trace

    def clone(self, new_trace=None, new_depth=None):
        """Clone the current node, but change some parameters."""

        cloned = self.__class__(
            trace=new_trace if new_trace is not None else self.trace,
            depth=new_depth if new_depth is not None else self.depth,
            features=self._features_type,
            max_len=self.max_len,
            dummies_per_insertion=self.dummies_per_insertion,
        )

        # If the new trace has not changed and features are extracted, copy the
        # extracted features over. This optimization gives a huge boost to the script's
        # performance.
        if (new_trace is None or new_trace == self.trace) and hasattr(
            self, "_features"
        ):
            cloned._features = self._features
        return cloned

    @profiled
    def expand(self):
        """Generate neighbours in the graph."""

        # Increment the counter of expanded nodes.
        counter = ExpansionCounter.get_default()
        counter.increment()

        children = []
        for i in range(len(self.trace)):
            trace = insert_dummy_packets(self.trace, i, self.dummies_per_insertion)
            if self.max_len is not None and (len(trace) > self.max_len):
                trace = trace[: self.max_len]
            node = self.clone(new_trace=trace, new_depth=self.depth + 1)
            children.append(node)

            trace = insert_dummy_packets(self.trace, i, self.dummies_per_insertion, direction=-1)
            if self.max_len is not None and (len(trace) > self.max_len):
                trace = trace[: self.max_len]
            node = self.clone(new_trace=trace, new_depth=self.depth + 1)
            children.append(node)
        return children

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.trace)


class LpExpandFunc(WithProblemContext):
    """Expand nodes with Lp costs."""

    @profiled
    def __call__(self, x):
        children = x.expand()
        costs = [
            np.linalg.norm(np.array(x.features - c.features), ord=self.problem_ctx.lp_space.p)
            for c in children
        ]

        # Poor man's logging.
        n = ExpansionCounter.get_default().count
        if n % DEBUG_PLOT_FREQ == 0:
            logger = logging.getLogger(LOGGER_NAME)
            logger.debug("Current depth     : %i" % x.depth)
            logger.debug("Branches          : %i" % len(children))
            logger.debug("Number of expands : %i" % n)
            logger.debug(
                "Cost stats        : %f / %f / %f"
                % (min(costs), float(sum(costs)) / len(children), max(costs))
            )
            logger.debug("---")

        return zip(children, costs)


class ZeroCostExpandFunc(WithProblemContext):
    """Expand nodes with zero cost."""

    @profiled
    def __call__(self, x):
        children = x.expand()

        # Poor man's logging.
        n = ExpansionCounter.get_default().count
        if n % DEBUG_PLOT_FREQ == 0:
            logger = logging.getLogger(LOGGER_NAME)
            logger.debug("Current depth     : %i" % x.depth)
            logger.debug("Branches          : %i" % len(children))
            logger.debug("Number of expands : %i" % n)
            logger.debug("---")

        for child in children:
            yield child, 0


@profiled
def hash_fn(x):
    """Hash function for examples."""
    x_str = str(x.trace)
    return hash(x_str)


@attr.s
class WfpProblemContext(CategoricalLpProblemContext):
    heuristic = attr.ib(default="confidence")
    cost = attr.ib(default="zero")

    # Only used if heuristic is random.
    heuristic_seed = attr.ib(default=0)

    def get_graph_search_problem(self):
        if self.cost == "zero":
            expand_fn = ZeroCostExpandFunc(problem_ctx=self)
        elif self.cost == "lp":
            expand_fn = LpExpandFunc(problem_ctx=self)

        if self.heuristic == "confidence":
            raw_heuristic = ConfidenceHeuristic(problem_ctx=self)
            heuristic_fn = lambda x: raw_heuristic(x.features)
        elif self.heuristic == "dist":
            raw_heuristic = LinearHeuristic(problem_ctx=self)
            heuristic_fn = lambda x: raw_heuristic(x.features)
        elif self.heuristic == "random":
            random.seed(self.heuristic_seed)
            heuristic_fn = lambda x: random.random()

        return GraphSearchProblem(
            goal_fn=GoalFunc(problem_ctx=self),
            search_fn=generalized_a_star_search,
            expand_fn=expand_fn,
            heuristic_fn=heuristic_fn,
            hash_fn=hash_fn,
        )


@attr.s
class ConfidenceHeuristic:
    """Target confidence maximization heuristic."""
    problem_ctx = attr.ib()

    def __call__(self, x):
        confidence = self.problem_ctx.clf.predict_proba([x])[
                0, self.problem_ctx.target_class]
        if confidence >= self.problem_ctx.target_confidence:
            return -np.inf

        score = self.problem_ctx.clf.decision_function([x])[0]
        sign = -1 if self.problem_ctx.target_class == 1 else +1
        return sign * self.problem_ctx.epsilon * score


common_options = [
    click.option(
        "--data_path",
        default="data/knndata",
        show_default=True,
        type=click.Path(exists=True, file_okay=False),
        help="Path to knndata traces.",
    ),
    click.option(
        "--log_file", default="log/wfp.log", type=click.Path(), help="Log file path."
    ),
    click.option("--seed", default=1, type=int, help="Random seed."),
    click.option(
        "--features",
        default="cumul",
        show_default=True,
        type=click.Choice(["raw", "cumul", "total"]),
        help="Feature extraction.",
    ),
    click.option(
        "--shuffle/--no_shuffle",
        default=False,
        show_default=True,
        help="Whether to shuffle the traces in the dataset.",
    ),
    click.option(
        "--normalize/--no_normalize",
        default=False,
        show_default=True,
        help="Whether to normalize the dataset.",
    ),
    click.option(
        "--num_traces",
        default=None,
        type=int,
        help="Number of traces from the dataset to consider.",
    ),
    click.option(
        "--max_trace_len",
        default=6746,
        show_default=True,
        type=int,
        help="Number of packets to cut traces to.",
    ),
    click.option(
        "--filter_by_len/--no_filter_by_len",
        default=True,
        show_default=True,
        help="Whether to filter out traces over max_trace_len.",
    ),
]


@click.group()
def cli():
    """Generate adversarial examples for website fingerprinting models."""
    pass


@cli.command()
@add_options(common_options)
@click.option(
    "--model", default="lr", type=click.Choice(["lr", "svmrbf"]), help="Model type."
)
@click.option(
    "--model_pickle", required=True, type=click.File("wb"), help="Model pickle path."
)
@click.pass_context
def train(
    ctx,
    data_path,
    log_file,
    seed,
    features,
    shuffle,
    normalize,
    num_traces,
    max_trace_len,
    filter_by_len,
    model,
    model_pickle,
):
    """Train a target model."""
    set_seed(seed)
    logger = setup_custom_logger(log_file)
    logger.info("Params: %s" % pprint.pformat(ctx.params))

    datasets = prepare_data(
        data_path,
        features=features,
        shuffle=shuffle,
        max_traces=num_traces,
        max_trace_len=max_trace_len,
        filter_by_len=filter_by_len,
        normalize=normalize,
    )

    logger.info("Fitting the model...")
    if model == "lr":
        clf = fit_lr(datasets)
    elif model == "svmrbf":
        clf = fit_svmrbf(datasets)

    logger.info("Saving the model...")
    pickle.dump(clf, model_pickle)
    logger.info("Done.")


@cli.command()
@add_options(common_options)
@click.option(
    "--model_pickle", type=click.File("rb"), help="Pickled model path.", required=True
)
@click.option(
    "--confidence_level",
    default=0.5,
    show_default=True,
    help="Target confidence level.",
)
@click.option(
    "--num_adv_examples", type=int, help="Number of adversarial examples to generate."
)
@click.option(
    "--iter_lim",
    default=10000,
    show_default=True,
    help="Max number of search iterations before giving up.",
)
@click.option(
    "--beam_size",
    default=None,
    show_default=True,
    type=int,
    help="A* fringe size.",
)
@click.option(
    "--sort_by_len/--no_sort_by_len",
    default=False,
    show_default=True,
    help="If true, the search will start from smaller traces.",
)
@click.option(
    "--dummies_per_insertion",
    default=1,
    show_default=True,
    help="Number of dummy packets to insert for each transformation.",
)
@click.option("--epsilon", default=1, show_default=True, help="The more the greedier.")
@click.option(
    "--cost",
    default="zero",
    type=click.Choice(["lp", "zero"]),
    help="Cost of transformations.",
)
@click.option(
    "--p_norm",
    default="inf",
    type=click.Choice(["1", "2", "inf"]),
    help="The p parameter of the Lp norm, only if cost is Lp.",
)
@click.option(
    "--heuristic",
    default="confidence",
    type=click.Choice(["confidence", "dist", "random"]),
    help="Heuristic: target confidence, or distance to the decision boundary."
)
@click.option(
    "--heuristic_seed",
    default="1",
    type=int,
    help="If using random heuristic, its seed.",
)
@click.option(
    "--output_pickle",
    type=click.Path(exists=False, dir_okay=False),
    help="Output results dataframe pickle.",
)
@click.pass_context
def generate(
    ctx,
    data_path,
    log_file,
    seed,
    features,
    shuffle,
    normalize,
    num_traces,
    max_trace_len,
    filter_by_len,
    model_pickle,
    confidence_level,
    num_adv_examples,
    iter_lim,
    beam_size,
    sort_by_len,
    dummies_per_insertion,
    epsilon,
    cost,
    p_norm,
    heuristic,
    heuristic_seed,
    output_pickle,
):
    """Generate adversarial examples."""
    if normalize:
        raise NotImplementedError(
            "Generation of normalized feature vectors not supported"
        )

    set_seed(seed)

    logger = setup_custom_logger(log_file)
    logger.info("Generating adversarial examples.")
    logger.info("Params: %s" % pprint.pformat(ctx.params))

    clf = pickle.load(model_pickle)
    datasets = prepare_data(
        data_path,
        features=features,
        shuffle=shuffle,
        max_trace_len=max_trace_len,
        filter_by_len=filter_by_len,
        normalize=normalize,
    )

    # Dataframe for storing the results.
    results = pd.DataFrame(
        columns=[
            "index",
            "filename",
            "found",
            "confidence",
            "original_confidence",
            "x",
            "adv_x",
            "real_cost",
            "path_cost",
            "nodes_expanded",
            "runtime",
            "conf_level",
        ]
    )

    # Set the global search parameters.
    problem_ctx = WfpProblemContext(
        clf=clf,
        target_class=1,
        target_confidence=confidence_level,
        lp_space=p_norm,
        epsilon=epsilon,
        heuristic=heuristic,
        heuristic_seed=heuristic_seed,
        cost=cost,
    )

    # Set the transformation graph parameters.
    node_params = dict(
        features=features,
        max_len=max_trace_len,
        dummies_per_insertion=dummies_per_insertion,
    )

    # Find indices of examples classified as negative.
    neg_indices, = np.where(
        clf.predict_proba(datasets.X_test_features)[:, 1] < confidence_level
    )
    neg_indices = list(neg_indices)
    if sort_by_len:
        neg_indices = sorted(neg_indices, key=lambda i: -len(datasets.X_test_cell[i]))

    logger.info("Searching for adversarial examples...")

    num_adv_examples_found = 0
    for i, original_index in enumerate(tqdm(neg_indices, ascii=True)):
        if (
            num_adv_examples is not None
            and num_adv_examples_found >= num_adv_examples
        ):
            break

        x = datasets.X_test_cell[original_index]
        initial_example_node = TraceNode(x, **node_params)

        # Instantiate counters for expanded nodes, and a profiler.
        expanded_counter = ExpansionCounter()
        ExpansionCounter.set_global_default(expanded_counter)
        per_example_profiler = Profiler()
        Profiler.set_global_default(per_example_profiler)

        x_adv, path_cost = _find_adversarial_example(
            initial_example_node=initial_example_node,
            graph_search_problem=problem_ctx.get_graph_search_problem(),
            iter_lim=iter_lim,
            beam_size=beam_size,
        )

        nodes_expanded = expanded_counter.count
        profiler_stats = per_example_profiler.compute_stats()
        runtime = profiler_stats["_find_adversarial_example"]["tot"]

        features = datasets.X_test_features[original_index]
        original_confidence = clf.predict_proba([features])[0, 1]

        if x_adv is None:
            # If an adversarial example was not found, only record index, runtime, and
            # the number of expanded nodes.
            results.loc[i] = [
                original_index,
                datasets.idxs_test[original_index],
                False,
                None,
                original_confidence,
                x,
                None,
                None,
                None,
                nodes_expanded,
                runtime,
                confidence_level,
            ]

        else:
            num_adv_examples_found += 1
            confidence = clf.predict_proba([x_adv.features])[0, 1]
            real_cost = x_adv.depth * x_adv.dummies_per_insertion

            results.loc[i] = [
                original_index,
                datasets.idxs_test[original_index],
                True,
                confidence,
                original_confidence,
                x,
                x_adv.trace,
                real_cost,
                path_cost,
                nodes_expanded,
                runtime,
                confidence_level,
            ]

        if output_pickle is not None:
            results.to_pickle(output_pickle)

        logger.debug(results.loc[i])

    return results


if __name__ == "__main__":
    cli()

