#!/usr/bin/env python3

import sys
sys.path.append('..')

import os
import math
import pickle
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from itertools import groupby
from IPython.display import display, HTML

from trickster.search import a_star_search
from trickster.adversarial_helper import ExpansionCounter
from trickster.wfp_helper import extract, load_cell

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from scipy.spatial import distance
from tqdm import tqdm

from defaultcontext import with_default_context
from profiled import Profiler, profiled

seed = 1

def load_data(path):
    labels = []
    data = []
    for fn in tqdm(os.listdir(path)):
        file_path = path + fn
        if os.path.isfile(file_path):
            cell_list = load_cell(file_path)
            feature_list = extract(cell_list)
            if "-" in str(fn):
                labels.append(1)
                data.append((cell_list, feature_list))
            else:
                labels.append(0)
                data.append((cell_list, feature_list))
    labels = np.array(labels)
    data = np.array(data)
    return data, labels


X, y = load_data(path='./notebooks/data/wfp_traces_toy/')
X, y = X[:500], y[:500]
print("Shape of data: {}, Shape of labels: {}".format(X.shape, y.shape))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
print("Number train samples: {}, Number test samples: {}".format(X_train.shape[0], X_test.shape[0]))

X_train_cell, X_train_features = zip(*X_train)
X_test_cell, X_test_features = zip(*X_test)
X_train_cell, X_train_features = np.array(X_train_cell), np.array(X_train_features)
X_test_cell, X_test_features = np.array(X_test_cell), np.array(X_test_features)

# Fit logistic regression and perform CV
clf = LogisticRegressionCV(
    Cs=21,
    cv=5,
    n_jobs=-1,
    random_state=seed
)
clf.fit(X_train_features, y_train)

# Get best score and C value
mean_scores = np.mean(clf.scores_[1], axis=0)
best_idx = np.argmax(mean_scores)
best_score = mean_scores[best_idx]
best_C = clf.Cs_[best_idx]

print('Best score is: {:.2f}%. Best C is: {:.4f}.'.format(best_score*100, best_C))
print('Test score is: {:.2f}%.'.format(clf.score(X_test_features, y_test)*100))


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


class BruteNode:
    def __init__(self, trace, depth=0):
        self.trace = list(trace)
        self.features = np.array(extract(self.trace))
        self.depth = depth

    def expand(self):
        # Increment the counter of expanded nodes.
        counter = ExpansionCounter.get_default()
        counter.increment()

        children = []
        for i in range(len(self.trace)):
            trace = insert_dummy_packets(self.trace, [i])[0]
            node = BruteNode(trace, depth=self.depth + 1)
            children.append(node)
        return children

    def __repr__(self):
        return 'BruteNode({})'.format(self.trace)


def _expand_fn(x, p_norm=1):
    """Wrap the example in `Node`, expand the node, and compute the costs.

    Returns a list of tuples (child, cost)
    """
    if not isinstance(x, BruteNode):
        x = BruteNode(trace=x, depth=0)
    children = x.expand()
    costs = [np.linalg.norm(
        np.array(x.features - c.features), ord=p_norm)
             for c in children]

    # Poor man's logging.
    # n = ExpansionCounter.get_default().count
    # if n % 5 == 0:
    #     print('Current depth     :', x.depth)
    #     print('Branches          :', len(children))
    #     print('Number of expands :', n)
    #     print('Cost stats        : %f / %f / %f' % (
    #         min(costs), float(sum(costs)) / len(children), max(costs)))
    #     print()

    return list(zip(children, costs))


def _goal_fn(x, clf, target_confidence=0.5):
    """Tell whether the example has reached the goal."""
    return clf.predict_proba([x.features])[0, 1] >= target_confidence


def _heuristic_fn(x, clf, q_norm=np.inf, eps=1., offset=0):
    """Distance to the decision boundary of a logistic regression classifier.

    By default the distance is w.r.t. L1 norm. This means that the denominator
    has to be in terms of the Holder dual norm (`q_norm`), so L-inf. I know,
    this interface is horrible.

    NOTE: The value has to be zero if the example is already on the target side
    of the boundary.
    """
    score = clf.decision_function([x.features])[0]
    if score >= 0:
        return 0.0
    h = np.abs(score) / np.linalg.norm(clf.coef_, ord=q_norm)
    return eps * (h + offset)


def hash_fn(x):
    """Hash function for examples."""
    x_str = str(x.trace)
    return hash(x_str)


@profiled
def find_adversarial(x, clf, p_norm=1, q_norm=np.inf,
                     target_confidence=0.5, eps=1.0, offset=0.,
                     return_path=False):
    """Transform an example until it is classified with target confidence."""

    x = BruteNode(x)
    if clf.predict_proba([x.features])[0, 1] >= target_confidence:
        raise Exception('Initial example is already classified as positive.')
    return a_star_search(
        start_node=x,
        expand_fn=lambda x: _expand_fn(x, p_norm=p_norm),
        goal_fn=lambda x: _goal_fn(x, clf, target_confidence),
        heuristic_fn=lambda x: _heuristic_fn(
            x, clf, eps=eps, q_norm=q_norm, offset=offset),
        iter_lim=int(10e5),
        hash_fn=hash_fn,
        return_path=return_path
    )

def find_adv_examples(X_cells, X_features, target_confidence, output_path=None,
                      p_norm=1, q_norm=np.inf, eps=100., offset=0.):
    """Find adversarial examples for a whole dataset"""

    # Dataframe for storing the results.
    results = pd.DataFrame(
        columns=['index', 'found', 'confidence', 'original_confidence', 'x', 'adv_x',
                 'real_cost', 'path_cost', 'nodes_expanded', 'runtime', 'conf_level'])

    # Indices of examples classified as negative.
    neg_indices, = np.where(clf.predict_proba(X_features)[:, 1] < target_confidence)

    for i, original_index in enumerate(tqdm(neg_indices)):
        x = X_cells[original_index]

        # Instantiate a counter for expanded nodes, and a profiler.
        expanded_counter = ExpansionCounter()
        per_example_profiler = Profiler()

        with expanded_counter.as_default(), per_example_profiler.as_default():
            x_adv, path_cost = find_adversarial(
                    x, clf, target_confidence=target_confidence,
                    p_norm=p_norm, q_norm=q_norm, eps=eps, offset=offset)

        nodes_expanded = expanded_counter.count
        runtime = per_example_profiler.compute_stats()['find_adversarial']['tot']
        original_confidence = clf.predict_proba([extract(x)])[0, 1]

        if x_adv is None:
            # If an adversarial example was not found, only record index, runtime, and
            # the number of expanded nodes.
            results.loc[i] = [original_index, False, None, original_confidence, x, None,
                              None, None, nodes_expanded, runtime, target_confidence]
        else:
            confidence = clf.predict_proba([x_adv.features])[0, 1]
            real_cost = len(x_adv.trace) - len(x)

            results.loc[i] = [original_index, True, confidence, original_confidence, x, x_adv.trace,
                              real_cost, path_cost, nodes_expanded, runtime, target_confidence]
            print(results.loc[i])
            if output_path is not None:
                with open(output_path, 'wb') as f:
                    pickle.dump(results, f)

    return results


def main():
    parser = argparse.ArgumentParser(description='wfp deterministic example')
    parser.add_argument('--confidence-level', type=float, default=0.5, metavar='N',
                        help='confidence level for adversarial example (default: 0.5)')
    parser.add_argument('--num-examples', type=int, default=2, metavar='N',
                        help='number of examples')
    parser.add_argument('--epsilon', type=int, default=1., metavar='N',
                        help='greediness parameter')
    parser.add_argument('--offset', type=int, default=1., metavar='N',
                        help='heuristic offset')
    args = parser.parse_args()
    output_path = 'wfp_det_eps_%1.1f_conf_l_%1.1f.pkl' % (args.epsilon, args.confidence_level)

    results = find_adv_examples(
            X_train_cell[:args.num_examples],
            X_train_features[:args.num_examples],
            args.confidence_level,
            p_norm=np.inf,
            q_norm=1,
            eps=args.epsilon,
            offset=args.offset,
            output_path=output_path)

if __name__ == "__main__":
    main()

