#!/usr/bin/env python3
"""
Experiment with generating adversarial examples for a fingerprinting classifier
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

import attr
import click
import numpy as np
import pandas as pd

from trickster.search import a_star_search
from trickster.adversarial_helper import ExpansionCounter
from trickster.adversarial_helper import SearchParams, SearchFuncs
from trickster.adversarial_helper import find_adversarial_example
from trickster.adversarial_helper import setup_custom_logger
from trickster.adversarial_helper import LOGGER_NAME
from trickster.wfp_helper import extract, pad_and_onehot, load_data
from trickster.wfp_helper import insert_dummy_packets
from trickster.utils.cli import add_options

from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn import preprocessing

from defaultcontext import with_default_context
from profiled import Profiler, profiled


DEBUG_PLOT_FREQ = 50


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)


@attr.s
class Datasets:
    X_train_cell = attr.ib()
    X_test_cell = attr.ib()
    X_train_features = attr.ib()
    X_test_features = attr.ib()

    y_train = attr.ib()
    y_test = attr.ib()


def prepare_data(
    data_path,
    features="cumul",
    shuffle=False,
    max_traces=None,
    max_trace_len=None,
    filter_by_len=True,
    normalize=True,
):
    """Load a dataset and extract features."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Loading the data...")

    X, y = load_data(
        path=data_path,
        shuffle=shuffle,
        max_traces=max_traces,
        max_trace_len=max_trace_len,
        filter_by_len=filter_by_len,
    )

    logger.info("Loaded.")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    logger.info(
        "Number of train samples: {}, Number test samples: {}".format(
            X_train.shape[0], X_test.shape[0]
        )
    )

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
    )


def fit_logistic_regression_model(datasets):
    """Train the target model --- logistic regression."""
    clf = LogisticRegressionCV(Cs=21, cv=5, n_jobs=-1, penalty="l2")
    clf.fit(datasets.X_train_features, datasets.y_train)

    logger = logging.getLogger(LOGGER_NAME)
    logger.info(
        "Test score is: {:.2f}%.".format(
            clf.score(datasets.X_test_features, datasets.y_test) * 100
        )
    )

    return clf


def fit_svm(datasets):
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
        return children

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.trace)


@profiled
def goal_fn(x):
    """Tell whether the example has reached the goal."""
    search_params = SearchParams.get_default()
    return (
        search_params.clf.predict_proba([x.features])[0, search_params.target_class]
        >= search_params.target_confidence
    )


@profiled
def heuristic_fn(x):
    """Distance to the decision boundary of a logistic regression classifier.

    NOTE: The value has to be zero if the example is already on the target side
    of the boundary.
    """
    search_params = SearchParams.get_default()
    score = search_params.clf.decision_function([x.features])[0]
    if score >= 0 and search_params.target_class == 1:
        return 0.0
    if score <= 0 and search_params.target_class == 0:
        return 0.0
    h = np.abs(score) / np.linalg.norm(
        search_params.clf.coef_[0], ord=search_params.q_norm
    )
    return search_params.epsilon * h


@profiled
def expand_fn(x):
    children = x.expand()
    search_params = SearchParams.get_default()
    costs = [
        np.linalg.norm(np.array(x.features - c.features), ord=search_params.p_norm)
        for c in children
    ]

    # Poor man's logging.
    n = ExpansionCounter().get_default().count
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

    return list(zip(children, costs))


@profiled
def hash_fn(x):
    """Hash function for examples."""
    x_str = str(x.trace)
    return hash(x_str)


def run_wfp_experiment(
    model_pickle,
    data_path="data/knndata",
    features="cumul",
    target_confidence=0.5,
    p_norm="1",
    epsilon=1.,
    shuffle=False,
    max_traces=None,
    max_trace_len=None,
    iter_lim=None,
    max_num_adv_examples=None,
    sort_by_len=False,
    dummies_per_insertion=1,
    output_pickle=None,
    log_file=None,
    filter_by_len=True,
    normalize=False,
):
    """Find adversarial examples for a dataset"""
    logger = logging.getLogger(LOGGER_NAME)

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

    # Pick appropriate values of p and q norms.
    if p_norm == "1":
        p_norm = 1
        q_norm = np.inf
    elif p_norm == "2":
        p_norm = 2
        q_norm = 2
    elif p_norm == "inf":
        p_norm = np.inf
        q_norm = 1

    # Set the global search parameters.
    search_params = SearchParams(
        clf=clf,
        target_class=1,
        target_confidence=target_confidence,
        p_norm=p_norm,
        q_norm=q_norm,
        epsilon=epsilon,
    )
    SearchParams.set_global_default(search_params)

    # Set the A* search functions.
    node_params = dict(
        features=features,
        max_len=max_trace_len,
        dummies_per_insertion=dummies_per_insertion,
    )
    search_funcs = SearchFuncs(
        example_wrapper_fn=lambda example: TraceNode(example, **node_params),
        expand_fn=expand_fn,
        goal_fn=goal_fn,
        heuristic_fn=heuristic_fn,
        hash_fn=hash_fn,
    )

    # Find indices of examples classified as negative.
    neg_indices, = np.where(
        clf.predict_proba(datasets.X_test_features)[:, 1] < target_confidence
    )
    neg_indices = list(neg_indices)
    if sort_by_len:
        neg_indices = sorted(neg_indices, key=lambda i: len(datasets.X_test_cell[i]))

    logger.info("Searching for adversarial examples...")

    num_adv_examples_found = 0
    for i, original_index in enumerate(tqdm(neg_indices)):
        if (
            max_num_adv_examples is not None
            and num_adv_examples_found >= max_num_adv_examples
        ):
            break

        x = datasets.X_test_cell[original_index]

        # Instantiate counters for expanded nodes, and a profiler.
        expanded_counter = ExpansionCounter()
        ExpansionCounter.set_global_default(expanded_counter)
        per_example_profiler = Profiler()
        Profiler.set_global_default(per_example_profiler)

        x_adv, path_cost = find_adversarial_example(
            x, a_star_search, search_funcs, iter_lim=iter_lim
        )

        nodes_expanded = expanded_counter.count
        profiler_stats = per_example_profiler.compute_stats()
        runtime = profiler_stats["find_adversarial_example"]["tot"]

        features = datasets.X_test_features[original_index]
        original_confidence = clf.predict_proba([features])[0, 1]

        if x_adv is None:
            # If an adversarial example was not found, only record index, runtime, and
            # the number of expanded nodes.
            results.loc[i] = [
                original_index,
                False,
                None,
                original_confidence,
                x,
                None,
                None,
                None,
                nodes_expanded,
                runtime,
                target_confidence,
            ]

        else:
            num_adv_examples_found += 1
            confidence = clf.predict_proba([x_adv.features])[0, 1]
            real_cost = x_adv.depth * x_adv.dummies_per_insertion

            results.loc[i] = [
                original_index,
                True,
                confidence,
                original_confidence,
                x,
                x_adv.trace,
                real_cost,
                path_cost,
                nodes_expanded,
                runtime,
                target_confidence,
            ]

        if output_pickle is not None:
            results.to_pickle(output_pickle)

        logger.debug(results.loc[i])

    return results


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
        clf = fit_logistic_regression_model(datasets)
    elif model == "svmrbf":
        clf = fit_svm(datasets)

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
    help="Max number of search iterations until before giving up.",
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
    "--p_norm",
    default="inf",
    type=click.Choice(["1", "2", "inf"]),
    help="The p parameter of the Lp norm for computing the cost.",
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
    sort_by_len,
    dummies_per_insertion,
    epsilon,
    p_norm,
    output_pickle,
):
    """Generate adversarial examples."""
    if normalize:
        raise NotImplementedError(
            "Generation of normalize feature vectors not supported"
        )

    set_seed(seed)

    logger = setup_custom_logger(log_file)
    logger.info("Generating adversarial examples.")
    logger.info("Params: %s" % pprint.pformat(ctx.params))

    run_wfp_experiment(
        model_pickle=model_pickle,
        data_path=data_path,
        target_confidence=confidence_level,
        features=features,
        p_norm=p_norm,
        epsilon=epsilon,
        iter_lim=iter_lim,
        max_trace_len=max_trace_len,
        max_num_adv_examples=num_adv_examples,
        sort_by_len=sort_by_len,
        dummies_per_insertion=dummies_per_insertion,
        output_pickle=output_pickle,
        filter_by_len=filter_by_len,
        shuffle=shuffle,
        max_traces=num_traces,
        normalize=normalize,
    )


if __name__ == "__main__":
    cli()
