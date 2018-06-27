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
from trickster.wfp_helper import extract, load_cell, onehot, pad_and_onehot, reverse_onehot

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from scipy.spatial import distance
from tqdm import tqdm

from defaultcontext import with_default_context
from profiled import Profiler, profiled

seed = 2018

def load_data(path='./data/wfp_traces/'):
    labels = []
    data = []
    for fn in tqdm(os.listdir(path)):
        file_path = path + fn
        if os.path.isfile(file_path):
            cell_list = load_cell(file_path)
            if "-" in str(fn):
                labels.append(1)
                data.append(cell_list)
            else:
                labels.append(0)
                data.append(cell_list)
    labels = np.array(labels)
    data = np.array(data)
    return data, labels

X, y = load_data(path='./data/wfp_traces_toy/')
X, y = X[:500], y[:500]

trace_len, X = pad_and_onehot(X)
print("Shape of data: {}, Shape of labels: {}".format(X.shape, y.shape))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
print("Number train samples: {}, Number test samples: {}".format(X_train.shape[0], X_test.shape[0]))

# Fit logistic regression and perform CV
clf = LogisticRegression(
)
clf.fit(X_train, y_train)

# Get best score and C value
print('Test score is: {:.2f}%.'.format(clf.score(X_test, y_test)*100))



@with_default_context(use_empty_init=True)
class Counter:
    def __init__(self):
        self.cnt = 0

    def increment(self):
        self.cnt += 1

    def count(self):
        return self.cnt

# Define `BruteNode` class transformation code:
# If at level i a node contains an input of length n, there will be n+1 branches at
# level i+1 with a new request at every possible index.

class BruteNode:
    
    def __init__(self, x):
        self.root = list(reverse_onehot(x, trace_len))
    
    def expand(self):
        # Increment the counter of expanded nodes.
        counter = Counter.get_default()
        counter.increment()

        children = []
        for idx in range(len(self.root) + 1):
            expanded_node = self.root[:idx] + [1] + self.root[idx:]
            children.append(np.array(expanded_node)[:trace_len])
        
        return onehot(children)
    
    def __repr__(self):
        return 'BruteNode({})'.format(self.root)

# All the functions that need to be passed into the search, in the expected format.

def _expand_fn(x, p_norm=1):
    """Wrap the example in `Node`, expand the node, and compute the costs.

    Returns a list of tuples (child, cost)
    """
    children = BruteNode(x).expand()
    costs = [np.linalg.norm(
        np.array(reverse_onehot(c, trace_len)), ord=p_norm) - np.linalg.norm(np.abs(np.array(reverse_onehot(x, trace_len))), ord=p_norm)
             for c in children]

    # Poor man's logging.
    n = Counter().get_default().count()
    if n % 5 == 0:
        print('Current level     :', x.level)
        print('Branches          :', len(children))
        print('Number of expands :', n)
        print('Cost stats        : %f / %f / %f' % (
            min(costs), float(sum(costs)) / len(children), max(costs)))
        print()

    return list(zip(children, costs))

def _goal_fn(x, clf, target_confidence=0.5):
    """Tell whether the example has reached the goal."""
    return clf.predict_proba([x])[0, 1] >= target_confidence

def _heuristic_fn(x, clf, q_norm=np.inf, eps=1., offset=0):
    """Distance to the decision boundary of a logistic regression classifier.

    By default the distance is w.r.t. L1 norm. This means that the denominator
    has to be in terms of the Holder dual norm (`q_norm`), so L-inf. I know,
    this interface is horrible.

    NOTE: The value has to be zero if the example is already on the target side
    of the boundary.
    """
    score = clf.decision_function([x])[0]
    if score >= 0:
        return 0.0
    h = np.abs(score) / np.linalg.norm(clf.coef_, ord=q_norm)
    return eps * (h + offset)

def hash_fn(x):
    """Hash function for examples."""
    x_str = str(x.root)
    return hash(x_str)

@profiled
def find_adversarial(x, clf, p_norm=1, q_norm=np.inf,
                     target_confidence=0.5, eps=1.0, offset=0.,
                     return_path=False):
    """Transform an example until it is classified with target confidence."""

    if clf.predict_proba([x])[0, 1] >= target_confidence:
        raise Exception('Initial example is already classified as positive.')        
    return a_star_search(
        start_node=x, 
        expand_fn=lambda x: _expand_fn(x, p_norm=p_norm), 
        goal_fn=lambda x: _goal_fn(x, clf, target_confidence), 
        heuristic_fn=lambda x: _heuristic_fn(x, clf, q_norm=q_norm), 
        iter_lim=int(1e1),
        hash_fn=hash_fn,
        return_path=return_path
    )

def find_adv_examples(X, target_confidence, p_norm=1, q_norm=np.inf):
    """Find adversarial examples for a whole dataset"""

    # Dataframe for storing the results.
    results = pd.DataFrame(
        columns=['index', 'found', 'confidence', 'original_confidence', 'x', 'adv_x',
                 'real_cost', 'path_cost', 'nodes_expanded', 'runtime', 'conf_level'])

    # Indices of examples classified as negative.
    neg_indices, = np.where(clf.predict_proba(X)[:, 1] < target_confidence)
    
    for i, original_index in enumerate(neg_indices):
        x = X[original_index]
        
        # Instantiate a counter for expanded nodes, and a profiler.
        expanded_counter = Counter()
        per_example_profiler = Profiler()

        with expanded_counter.as_default(), per_example_profiler.as_default():
            x_adv, path_cost = find_adversarial(
                    x, clf, target_confidence=target_confidence, eps=eps, offset=offset)

        nodes_expanded = expanded_counter.count()
        runtime = per_example_profiler.compute_stats()['find_adversarial']['tot']

        # If an adversarial example was not found, only record index, runtime, and
        # the number of expanded nodes.
        if x_adv is None:
            results.loc[i] = [original_index, False, [], None,
                              None, None, nodes_expanded, runtime, target_confidence]
        else:
            confidence = clf.predict_proba([x_adv])[0, 1]
            original_confidence = clf.predict_proba([x])[0, 1]
            real_cost = np.linalg.norm(x_adv, ord=p_norm) - np.linalg.norm(x, ord=p_norm)
            
            results.loc[i] = [original_index, True, confidence, original_confidence,
                              real_cost, path_cost, nodes_expanded, runtime, target_confidence]
            print(results.loc[i])

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

    results_graph = find_adv_examples(X, args.confidence_level)

    with open('./data/wfp_det_conf_l_%.2f.pkl' %(args.confidence_level), 'wb') as f:
        pickle.dump(results_graph, f)

if __name__ == "__main__":
    main()
