#!/usr/bin/env python3

import sys
import copy

sys.path.append("..")

import os
import math
import pickle
import argparse

import attr
import numpy as np
import pandas as pd

from trickster.search import a_star_search
from trickster.adversarial_helper import ExpansionCounter
from trickster.adversarial_helper import SearchParams, SearchFuncs
from trickster.adversarial_helper import find_adversarial_example
from trickster.wfp_helper import extract, pad_and_onehot, load_data
from trickster.wfp_helper import insert_dummy_packets

from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC

from defaultcontext import with_default_context
from profiled import Profiler, profiled


SEED = 1
np.random.seed(seed=SEED)

DEBUG_PLOT_FREQ = 10


@attr.s
class Datasets:
    X_train_cell = attr.ib()
    X_test_cell = attr.ib()
    X_train_features = attr.ib()
    X_test_features = attr.ib()

    y_train = attr.ib()
    y_test = attr.ib()


def prepare_data(data_path, features="cumul", max_len=None):
    X, y = load_data(path=data_path, max_len=max_len)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED
    )
    print(
        "Number of train samples: {}, Number test samples: {}".format(
            X_train.shape[0], X_test.shape[0]
        )
    )

    # Extract features
    if features == "cumul":
        X_train_features = np.array([extract(trace) for trace in X_train])
        X_test_features = np.array([extract(trace) for trace in X_test])

    elif features == "raw":
        pad_len, X_train_features = pad_and_onehot(X_train, pad_len=max_len)
        _, X_test_features = pad_and_onehot(X_test, pad_len=pad_len)

    print("Train features shape:", X_train_features.shape)
    print("Test features shape:", X_test_features.shape)

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
    print("Fitting the model...")
    clf = LogisticRegressionCV(Cs=21, cv=5, n_jobs=-1, penalty="l2", random_state=SEED)
    clf.fit(datasets.X_train_features, datasets.y_train)

    print(
        "Test score is: {:.2f}%.".format(
            clf.score(datasets.X_test_features, datasets.y_test) * 100
        )
    )

    return clf


class TraceNode:
    """A node in the tranformation graph of traces.

    :param trace: A trace
    :param depth: Depth in the graph
    :param features: One of ["cumul", "raw"]
    :param max_len: Max trace length (to pad to)
    :param dummies_per_insertion: Number of dummies to insert for each neighbouring node.
    """

    def __init__(
        self, trace, depth=0, features="cumul", max_len=None, dummies_per_insertion=1
    ):
        if features not in ["cumul", "raw"]:
            raise ValueError("Unknown features type: %s" % features)
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
            encoded_trace = encoded_trace

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
                trace = trace[:self.max_len]
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

    By default the distance is w.r.t. L1 norm. This means that the denominator
    has to be in terms of the Holder dual norm (`q_norm`), so L-inf. I know,
    this interface is horrible.

    NOTE: The value has to be zero if the example is already on the target side
    of the boundary.
    """
    search_params = SearchParams.get_default()
    score = search_params.clf.decision_function([x.features])[0]
    if score >= 0:
        return 0.0
    h = np.abs(score) / np.linalg.norm(
        search_params.clf.coef_, ord=search_params.q_norm
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
        print("Current depth     :", x.depth)
        print("Branches          :", len(children))
        print("Number of expands :", n)
        print(
            "Cost stats        : %f / %f / %f"
            % (min(costs), float(sum(costs)) / len(children), max(costs))
        )
        print()

    return list(zip(children, costs))


@profiled
def hash_fn(x):
    """Hash function for examples."""
    x_str = str(x.trace)
    return hash(x_str)


def run_wfp_experiment(
    data_path="data/knndata",
    features="cumul",
    target_confidence=0.5,
    output_path=None,
    p_norm=1,
    q_norm=np.inf,
    epsilon=1.,
    max_trace_len=None,
    iter_lim=None,
    max_num_examples=None,
    dummies_per_insertion=1,
):
    """Find adversarial examples for a whole dataset"""

    datasets = prepare_data(data_path, features=features, max_len=max_trace_len)
    clf = fit_logistic_regression_model(datasets)

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

    search_params = SearchParams(
        clf=clf,
        target_class=1,
        target_confidence=target_confidence,
        p_norm=p_norm,
        q_norm=q_norm,
        epsilon=epsilon,
    )
    SearchParams.set_global_default(search_params)

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

    # Indices of examples classified as negative.
    neg_indices, = np.where(
        clf.predict_proba(datasets.X_test_features)[:, 1] < target_confidence
    )
    num_adv_examples_found = 0
    for i, original_index in enumerate(tqdm(neg_indices)):
        if max_num_examples is not None and num_adv_examples_found > max_num_examples:
            break

        x = datasets.X_test_cell[original_index]

        # Instantiate a counter for expanded nodes, and a profiler.
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
            if output_path is not None:
                with open(output_path, "wb") as f:
                    pickle.dump(results, f)

        print(results.loc[i])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial examples for WFP"
    )
    parser.add_argument(
        "--features",
        choices=["cumul", "raw"],
        default="cumul",
        help="feature extraction",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.5,
        help="target confidence level for adversarial example",
    )
    parser.add_argument(
        "--num_examples", type=int, default=None, help="number of examples"
    )
    parser.add_argument(
        "--iter_lim", type=int, default=1000, help="max number of search iterations"
    )
    parser.add_argument("--epsilon", type=int, default=1., help="greediness parameter")
    parser.add_argument(
        "--output", default=None, help="path to output pickled dataframe"
    )
    parser.add_argument(
        "--data_path", default="data/knndata/", help="path to input traces"
    )
    parser.add_argument(
        # 6746 is 95-th percentile on the knndata.
        "--max_trace_len",
        type=int,
        default=6745,
        help="max trace length",
    )
    parser.add_argument(
        "--dummies_per_insertion",
        type=int,
        default=3,
        help="number of dummy packets per insersion",
    )
    args = parser.parse_args()

    results = run_wfp_experiment(
        data_path=args.data_path,
        target_confidence=args.confidence_level,
        features=args.features,
        p_norm=np.inf,
        q_norm=1,
        epsilon=args.epsilon,
        iter_lim=args.iter_lim,
        max_trace_len=args.max_trace_len,
        max_num_examples=args.num_examples,
        dummies_per_insertion=args.dummies_per_insertion,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
