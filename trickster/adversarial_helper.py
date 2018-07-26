import sys
sys.path.append('..')

# Ignore warnings.
import warnings
warnings.filterwarnings('ignore')

# Handle library imports.
import numpy as np
import pandas as pd
import logging

from trickster.search import a_star_search, ida_star_search
from trickster.expansion import expand
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from scipy.sparse import issparse

from collections import Counter as CollectionsCounter
from defaultcontext import with_default_context
from profiled import Profiler, profiled
from tqdm import tqdm

###########################################
###########################################
###########################################

# Handle global variables.
LOGGER_NAME = 'adversarial'

###########################################
###########################################
###########################################

# Define useful helper functions.

def setup_custom_logger(log_file='../logging/output.log'):
    '''Set up a logger object to print info to stdout and debug to file.'''
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='[%(asctime)s - %(levelname)-4s] >> %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def get_feature_coef_importance(X, clf, transformable_feature_idxs):
    '''
    Get the most important features from the transformable feature set
    based on classifier parameters.
    '''
    if issparse(X):
        X = X[:, transformable_feature_idxs].toarray()
    else:
        X = X[:, transformable_feature_idxs]

    importance = np.std(X, 0) * np.abs(clf.coef_[0][transformable_feature_idxs])
    imps_sum = np.sum(importance)
    importance_coef = [(idx, imp / imps_sum) for idx, imp in zip(transformable_feature_idxs, importance)]
    return sorted(importance_coef, key=lambda x: x[1], reverse=True)

def get_feature_diff_importance(difference, transformable_feature_idxs):
    '''
    Get the most important features from the transformable feature set
    based on the feature difference between the initial and adversarial example.
    '''
    difference = CollectionsCounter([item for sublist in difference for item in sublist])
    cnts_sum = np.sum([v for k, v in difference.items()])
    importance_diff = [(idx, cnt / cnts_sum) for idx, cnt in difference.items()]
    importance_diff += [(idx, 0) for idx in transformable_feature_idxs if idx not in difference.keys()]
    return sorted(importance_diff, key=lambda x: x[1], reverse=True)

def substring_index(xs, item):
    '''Can be used to get the index of the required substring within a list of strings.'''
    idxs = [i for (i, x) in enumerate(xs) if item in x]
    return idxs

def create_reduced_classifier(clf, x, transformable_feature_idxs):
    '''Construct a reduced classifier based on the original one.'''

    # Establish non-transformable feature indexes.
    feature_idxs = np.arange(x.size)
    non_transformable_feature_idxs = np.setdiff1d(feature_idxs, transformable_feature_idxs)

    # Create the reduced classifier.
    clf_reduced = LogisticRegressionCV()
    clf_reduced.coef_ = clf.coef_[:, transformable_feature_idxs]
    clf_reduced.intercept_ = np.dot(clf.coef_[0, non_transformable_feature_idxs], x[non_transformable_feature_idxs])
    clf_reduced.intercept_ += clf.intercept_

    assert np.array_equal(clf.predict_proba([x]).round(2), clf_reduced.predict_proba([x[transformable_feature_idxs]]).round(2))
    return clf_reduced

###########################################
###########################################
###########################################

# Define useful helper classes.
class CounterLimitExceededError(Exception):
    pass

@with_default_context(use_empty_init=True)
class Counter:

    def __init__(self, counter_lim=None, debug_freq=None, **kwargs):
        self.cnt = 0
        self.counter_lim = counter_lim
        self.debug_freq = debug_freq

    def increment(self):
        logger = logging.getLogger(LOGGER_NAME)
        if self.counter_lim is not None and self.cnt > self.counter_lim:
            raise CounterLimitExceededError('Expansion counter limit {} reached.'.format(self.counter_lim))
        if self.debug_freq is not None and self.cnt % self.debug_freq == 0:
            logger.debug('Node counter is: {}.'.format(self.cnt))
        self.cnt += 1

    def set_count(self, count):
        self.cnt = count

    def get_count(self):
        return self.cnt

###########################################
###########################################
###########################################

# Functions that perform adversarial example search.

@profiled
def find_adversarial(x, clf, epsilon, search_fn, expansions, zero_to_one,
        target_confidence=0.5, p_norm=1, q_norm=np.inf, **kwargs):
    """Transform an example until it is classified with target confidence."""

    def default_expand_fn(x, expansions, p_norm=1):
        """Expand x and compute the costs.
        Returns a list of tuples (child, cost)
        """
        # Increment the counter of expanded nodes.
        counter = Counter.get_default()
        counter.increment()

        children = expand(x, expansions)
        costs = [np.linalg.norm(x - c, ord=p_norm) for c in children]

        return list(zip(children, costs))

    def default_goal_fn(x, clf, zero_to_one, target_confidence=0.5):
        """Tell whether the example has reached the goal."""
        if zero_to_one:
            return clf.predict_proba([x])[0, 1] >= target_confidence
        else:
            return clf.predict_proba([x])[0, 1] <= target_confidence
        # return clf.predict_proba([x])[0, 1] <= target_confidence

    def default_heuristic_fn(x, clf, epsilon, zero_to_one, q_norm=np.inf):
        """Distance to the decision boundary of a logistic regression classifier.
        By default the distance is w.r.t. L1 norm. This means that the denominator
        has to be in terms of the Holder dual norm (`q_norm`), so L-inf. I know,
        this interface is horrible.
        NOTE: The value has to be zero if the example is already on the target side
        of the boundary.
        """
        confidence = clf.predict_proba([x])[0, 1]
        if zero_to_one and confidence >= target_confidence:
            return 0.0
        elif not zero_to_one and confidence <= target_confidence:
            return 0.0
        score = clf.decision_function([x])[0]
        h = np.abs(score) / np.linalg.norm(clf.coef_[0], ord=q_norm)
        return h * epsilon

    def default_hash_fn(x):
        """Hash function for examples."""
        return hash(x.tostring())

    expand_fn = kwargs.get('expand_fn', default_expand_fn)
    goal_fn = kwargs.get('goal_fn', default_goal_fn)
    heuristic_fn = kwargs.get('heuristic_fn', default_heuristic_fn)
    hash_fn = kwargs.get('hash_fn', default_hash_fn)

    return search_fn(
        start_node=x,
        expand_fn=lambda x: expand_fn(x, expansions, p_norm=p_norm),
        goal_fn=lambda x: goal_fn(x, clf, zero_to_one, target_confidence),
        heuristic_fn=lambda x: heuristic_fn(x, clf, epsilon, zero_to_one, q_norm=q_norm),
        hash_fn=hash_fn
    )

@profiled
def adversarial_search(X, idxs, clf, expansions,
    p_norm, q_norm, transformable_feature_idxs, **kwargs):
    """Find adversarial examples for specified indexes."""
    logger = logging.getLogger(LOGGER_NAME)

    # Dataframe for storing the results.
    results = pd.DataFrame(
        columns=['index', 'found', 'expansions', 'x', 'init_confidence',
                 'x_adv', 'adv_confidence', 'real_cost', 'path_cost',
                 'difference', 'nodes_expanded', 'runtime']
    )

    for i, idx in enumerate(tqdm(idxs, ascii=True)):

        logger.debug('Searching for adversarial example {}/{} using initial observation at index: {}.'
            .format(i, len(idxs), idx))

        if issparse(X):
            x = X[idx].toarray()[0]
        else:
            x = X[idx]

        # Transform x and the classifier to operate in the reduced feature space.
        x_reduced = x[transformable_feature_idxs]
        clf_reduced = create_reduced_classifier(clf, x, transformable_feature_idxs)

        # Instantiate a counter for expanded nodes, and a profiler.
        expanded_counter = Counter(**kwargs)
        per_example_profiler = Profiler()

        x_adv, x_adv_reduced, adv_found = None, None, None
        adv_confidence, difference = None, None
        real_cost, path_cost = None, None
        runtime = None

        with expanded_counter.as_default(), per_example_profiler.as_default():
            try:
                x_adv_reduced, path_cost = find_adversarial(
                    x=x_reduced,
                    clf=clf_reduced,
                    expansions=expansions,
                    p_norm=p_norm,
                    q_norm=q_norm,
                    **kwargs
                )
                adv_found = False if x_adv_reduced is None else True

            except CounterLimitExceededError as e:
                logger.debug('WARN! For observation at index {}: {}'.format(idx, e))

        # Record some basic statistics.
        nodes_expanded = expanded_counter.get_count()
        init_confidence = clf_reduced.predict_proba([x_reduced])[0, 1]
        expands = [(idxs, fn.__name__) for (idxs, fn) in expansions]
        runtime_stats = per_example_profiler.compute_stats()
        if 'find_adversarial' in runtime_stats:
            runtime = runtime_stats['find_adversarial']['tot']

        if x_adv_reduced is not None:
            logger.debug('Adversarial example found {}/{} found using initial observation at index: {}!'
                .format(i, len(idxs), idx))
            # Construct the actual adversarial example.
            x_adv = np.array(x)
            x_adv[transformable_feature_idxs] = x_adv_reduced

            # Compute further statistics.
            adv_confidence = clf.predict_proba([x_adv])[0, 1]
            real_cost = np.linalg.norm(x - x_adv, ord=p_norm)
            difference, = np.where(x != x_adv)

        results.loc[i] = [
            idx, adv_found, expands, x, init_confidence,
            x_adv, adv_confidence, real_cost, path_cost,
            difference, nodes_expanded, runtime
        ]

    return results

###########################################
###########################################
###########################################

# Define a wrapper function to perform experiments.
def experiment_wrapper(load_transform_data_fn, get_expansions_fn, clf_fit_fn,
    target_confidence, benchmark_search_fn=None, zero_to_one=True, **kwargs):
    '''Description goes here.'''
    logger = kwargs['logger'] if 'logger' in kwargs else setup_custom_logger()
    random_state = kwargs['random_state'] if 'random_state' in kwargs else 2010
    np.random.seed(seed=random_state)

    # Load and prepare data for learning.
    X, y, features = load_transform_data_fn(**kwargs)
    logger.debug('Shape of X: {}. Shape of y: {}.'.format(X.shape, y.shape))

    # Get required expansions and sorted indexes of transformable features.
    expansions, transformable_feature_idxs = get_expansions_fn(features, **kwargs)

    # Split into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
    logger.debug('Number of training points: {}. Number of test points: {}.'.format(X_train.shape[0], X_test.shape[0]))

    # Fit and evaluate the classifier.
    clf = clf_fit_fn(X_train, y_train, **kwargs)
    train_score, test_score = clf.score(X_train, y_train)*100, clf.score(X_test, y_test)*100
    logger.debug("Resulting training accuracy is: {:.2f}%. Test accuracy is: {:.2f}%.\n"
                .format(train_score, test_score))

    # Estimate contribution of each (transformable) feature to the classifier performance.
    logger.debug('Computing importance of each feature based on the classifier parameters.')
    importance_coef = get_feature_coef_importance(X_train, clf, transformable_feature_idxs)

    # Indices of examples in the original class.
    confidence_margin = kwargs.get('confidence_margin', 1)
    if zero_to_one:
        idxs, = np.where(
            (clf.predict_proba(X)[:, 1] < target_confidence) &
            (clf.predict_proba(X)[:, 1] >= target_confidence - confidence_margin)
        )
    else:
        idxs, = np.where(
            (clf.predict_proba(X)[:, 1] > target_confidence) &
            (clf.predict_proba(X)[:, 1] <= target_confidence + confidence_margin)
        )

    # Perform adversarial example search using A* search.
    logger.info('Searching for adversarial examples for {} observations using A* algorithm...'.format(len(idxs)))
    search_results = adversarial_search(
        X=X, idxs=idxs, clf=clf, expansions=expansions,
        p_norm = 1, q_norm = np.inf, target_confidence=target_confidence,
        transformable_feature_idxs=transformable_feature_idxs, epsilon=1,
        search_fn=a_star_search, zero_to_one=zero_to_one, **kwargs
    )

    # Compute feature importance based on the count of feature transformations.
    logger.debug('Computing importance of each feature based on difference between "x" and adversarial "x".')
    importance_diff = get_feature_diff_importance(search_results['difference'], transformable_feature_idxs)

    # Perform adversarial example search using a benchmark search.
    benchmark_results = None
    if benchmark_search_fn is not None:
        logger.info('Searching for adversarial examples for {} observations using a benchmark search...'.format(len(idxs)))
        benchmark_results = benchmark_search_fn(
            X=X, idxs=idxs, clf=clf, expansions=expansions,
            p_norm = 1, q_norm = np.inf, target_confidence=target_confidence,
            transformable_feature_idxs=transformable_feature_idxs, epsilon=1,
            search_fn=a_star_search, zero_to_one=zero_to_one,
            logger_name=LOGGER_NAME, **kwargs
        )

    # Output result.
    result = {
        'feature_count': X.shape[1],
        'features': features,
        'transformable_feature_idxs': transformable_feature_idxs,
        'clf_test_score': test_score,
        'coef_importance': importance_coef,
        'diff_importance': importance_diff,
        'success_rates': search_results['found'].mean(),
        'avg_init_confidences': search_results['init_confidence'].mean(),
        'avg_adv_confidences': search_results['adv_confidence'].mean(),
        'avg_path_costs': search_results['path_cost'].mean(),
        'avg_real_cost': search_results['real_cost'].mean(),
        'avg_counter': search_results['nodes_expanded'].mean(),
        'avg_runtime': search_results['runtime'].mean(),
        'benchmark_results': benchmark_results,
        'search_results': search_results
    }

    return result
