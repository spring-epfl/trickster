#!/usr/bin/env python3

import os
import sys

sys.path.append("..")

# Ignore warnings.
import warnings

warnings.filterwarnings("ignore")

# Handle library imports.
import numpy as np
import pandas as pd

from trickster.search import a_star_search, ida_star_search
from trickster.adversarial_helper import *
from trickster.wfp_helper import *
from trickster.expansion import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

###########################################
###########################################
###########################################

# Handle global variables.
SEED = 2018
np.random.seed(seed=SEED)

###########################################
###########################################
###########################################


def extract_cumul(x):
    return np.array(extract(x))


# Define experiment helper functions.
def load_transform_data_fn(path):
    """
    Load and preprocess data, returning the examples and labels as numpy.
    """
    labels = []
    data = []
    max_trace_len = -1
    for fn in tqdm(os.listdir(path)):
        file_path = os.path.join(path, fn)
        if os.path.isfile(file_path):
            cell_list = load_cell(file_path)
            if "-" in str(fn):
                labels.append(1)
            else:
                labels.append(0)
            data.append(cell_list)
            if len(cell_list) > max_trace_len:
                max_trace_len = len(cell_list)
    labels = np.array(labels)
    padded_cells = np.array(
        [np.pad(c, (0, max_trace_len - len(c)), "constant") for c in data]
    )
    return padded_cells, labels, np.arange(padded_cells.shape[1])


def clf_fit_fn(X_train, y_train, **kwargs):
    """
    Fit logistic regression by performing a Grid Search with Cross Validation.
    """
    Cs = np.arange(0.1, 2, 0.025)
    class_weight = None  # balanced or None
    scoring = "f1"  # accuracy, f1 or roc_auc

    clf = LogisticRegressionCV(
        Cs=Cs,
        cv=5,
        n_jobs=-1,
        penalty="l2",
        scoring=scoring,
        class_weight=class_weight,
        random_state=SEED,
    )

    pipeline = make_pipeline(
        FunctionTransformer(lambda data: np.array([extract_cumul(x) for x in data])),
        clf,
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def insert_dummy_packets(trace, idxs):
    """
    >>> insert_dummy_packets([1, -1, 1], [0])
    [1, 1, -1, 1]
    >>> insert_dummy_packets([1, -1, 1], [3])
    [1, -1, 1, 1]
    >>> insert_dummy_packets([1, -1, 1], [0, 3])
    [1, 1, -1, 1, 1]
    """
    results = []
    for i in idxs:
        if i > 0 and trace[i - 1] == 0:
            continue
        extended = list(trace)
        extended.insert(i, 1)
        results.append(extended)
    return results


def get_expansions_fn(features, **kwargs):
    expansions = []
    for i in features:
        expansions.append(([i], insert_dummy_packets))
    return expansions, None


class TraceNode(Node):
    def __init__(self, x, depth=0, feature_extract_fn=None):
        if feature_extract_fn is None:
            feature_extract_fn = extract_cumul
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        super().__init__(x, depth, feature_extract_fn)


def example_wrapper_fn(x):
    return TraceNode(x)


@profiled
def graph_expand_fn(x):
    """Customize the cost function --- cost is over the feature space."""
    search_params = SearchParams.get_default()
    children = x.expand(search_params.expansions)
    costs = [
        np.linalg.norm(x.features - c.features, ord=search_params.p_norm)
        for c in children
    ]

    counter = ExpansionCounter.get_default()
    print("Expansion #", counter.count)

    return list(zip(children, costs))


def _get_normalizer():
    search_params = SearchParams.get_default()
    if not hasattr(search_params, "normalizer"):
        search_params.normalizer = np.linalg.norm(
            search_params.clf.steps[-1][1].coef_[0], ord=search_params.q_norm
        )
    return search_params.normalizer


@profiled
def graph_heuristic_fn(x):
    """Adapt the default heuristic to the pipeline classifier."""
    search_params = SearchParams.get_default()
    confidence = search_params.clf.predict_proba([x.src])[0, search_params.target_class]
    if confidence >= search_params.target_confidence:
        return 0.0
    score = search_params.clf.decision_function([x.src])[0]
    h = np.abs(score) / _get_normalizer()
    return h * search_params.epsilon


def baseline_detaset_find_examples_fn(search_funcs=None, **kwargs):
    """Perform BFS adversarial example search to baseline against A* search."""
    search_funcs.heuristic_fn = lambda *args, **lambda_kwargs: 0
    results = dataset_find_adversarial_examples(search_funcs=search_funcs, **kwargs)
    return results


###########################################
###########################################
###########################################

# Main function.
if __name__ == "__main__":
    # Setup a custom logger.
    log_file = "_logging/credit_output.log"
    logger = setup_custom_logger(log_file)

    # Define dataset location.
    data_folder = "notebooks/data/wfp_traces_toy"

    # Define the meta-experiment parameters.
    p_norm, q_norm = 1, np.inf

    # Perform the experiments.
    logger.info("Starting experiments for the toy WFP dataset.")
    search_funcs = SearchFuncs(
        example_wrapper_fn=example_wrapper_fn,
        expand_fn=graph_expand_fn,
        heuristic_fn=graph_heuristic_fn,
    )
    result = experiment_wrapper(
        load_transform_data_fn=load_transform_data_fn,
        load_kwargs=dict(path=data_folder),
        clf_fit_fn=clf_fit_fn,
        target_class=1,
        search_kwargs=dict(p_norm=np.inf, q_norm=1.),
        search_funcs=search_funcs,
        get_expansions_fn=get_expansions_fn,
        baseline_dataset_find_examples_fn=baseline_detaset_find_examples_fn,
        logger=logger,
        random_state=SEED,
    )
