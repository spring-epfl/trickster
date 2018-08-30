#!/usr/bin/env python3

import sys

sys.path.append("..")

# Ignore warnings.
import warnings

warnings.filterwarnings("ignore")

# Handle library imports.
import numpy as np
import pandas as pd
import itertools
import logging
import ast

from trickster.search import a_star_search, ida_star_search
from trickster.expansion import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from scipy.spatial import distance
from tqdm import tqdm_notebook, tqdm

from defaultcontext import with_default_context
from profiled import Profiler, profiled

###########################################
###########################################
###########################################

# Handle global variables.

COUNTER_LIM = 50000
DEBUG_FREQ = 500

FEATURES = [
    "user_tweeted",
    "user_retweeted",
    "user_favourited",
    "user_replied",
    "likes_per_tweet",
    "retweets_per_tweet",
    "lists_per_user",
    "follower_friend_ratio",
    "tweet_frequency",
    "favourite_tweet_ratio",
    "age_of_account_in_days",
    "sources_count",
    "urls_count",
    "cdn_content_in_kb",
    "source_identity",
]

logger = None

SEED = 2018

np.random.seed(seed=SEED)

###########################################
###########################################
###########################################

# Define useful helper functions.


def setup_custom_logger():
    # Set up a logger object to print info to stdout and debug to file.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="[%(asctime)s - %(levelname)-4s] >> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.FileHandler("output.log", mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_data(data_files):
    # Load data for humans.
    df1 = pd.read_csv(data_files[0])
    df1 = df1.drop("screen_name", axis=1)  # remove screen_name column
    df1 = df1.assign(is_bot=0)

    # Load data for bots.
    df2 = pd.read_csv(data_files[1])
    df2 = df2.drop("screen_name", axis=1)  # remove screen_name column
    df2 = df2.assign(is_bot=1)

    # Concatenate dataframes.
    df = df1.append(df2, ignore_index=True)
    return df


def _transform_source_identity(X_k):
    """Helper to transform the source_identity field."""
    X_k = X_k.apply(lambda x: x.replace(";", ","))
    X_k = X_k.apply(ast.literal_eval)

    # Get the value of the maximum element in X_k.
    arg_max = np.max([item for sublist in X_k.values.tolist() for item in sublist])

    N = X_k.shape[0]
    X_k_transformed = np.zeros((N, arg_max + 1), dtype="intc")

    # Set 1 if the source is present for the user.
    for i in range(N):
        for item in X_k[i]:
            X_k_transformed[i, item] = 1

    return X_k_transformed


def feature_transform(df, features, bins=10):
    # Perform feature quantization.

    for feat in FEATURES:

        # Source identity is a special case.
        if feat == "source_identity":
            continue

        # Drop feature if it is not in the provided feature set.
        if feat not in features:
            df = df.drop(feat, axis=1)
            continue

        # Drop feature if there is only 1 distinct value.
        if np.unique(df.loc[:, feat]).size == 1:
            df = df.drop(feat, axis=1)
            continue

        series = df.loc[:, feat]
        df.loc[:, feat] = pd.qcut(series, bins, duplicates="drop")

    # Encode 'source_identity' field by setting '1's if source is present.
    transformed = _transform_source_identity(df.loc[:, "source_identity"])
    df = df.drop("source_identity", axis=1)

    df["source_identity_other"] = transformed[:, 0]
    df["source_identity_browser"] = transformed[:, 1]
    df["source_identity_mobile"] = transformed[:, 2]
    df["source_identity_osn"] = transformed[:, 3]
    df["source_identity_automation"] = transformed[:, 4]
    df["source_identity_marketing"] = transformed[:, 5]

    # Perform one-hot encoding
    df = pd.get_dummies(df)

    # Separate features from targets
    df_X = df.iloc[:, 1:]
    df_y = df.iloc[:, 0]

    return df_X, df_y


def fit_validate(X_train, y_train):
    # Fit logistic regression by performing a Grid Search with Cross Validation.
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

    clf.fit(X_train, y_train)
    return clf


###########################################
###########################################
###########################################

# Define useful helper classes.


class LogisticRegressionScikitSaliencyOracle:
    def __init__(self, model):
        self.model = model

    def eval(self, _):
        return self.model.coef_[0]


@with_default_context(use_empty_init=True)
class Counter:
    def __init__(self):
        self.cnt = 0

    def increment(self):
        self.cnt += 1

    def count(self):
        return self.cnt


class Node:
    def __init__(self, x):
        self.root = x

    def expand(self, manifest_set):
        """Generate all children of the current node."""
        # Increment the counter of expanded nodes.
        counter = Counter.get_default()
        counter.increment()

        children = []

        for feat_idx in manifest_set:

            # Skip if the feature is already set.
            if self.root[feat_idx] == 1:
                continue

            child = np.array(self.root)
            child[feat_idx] = 1
            children.append(child)

        return children

    def __repr__(self):
        return "Node({})".format(self.root)


###########################################
###########################################
###########################################

# Provide implemention of Algorithm 1 from Grosse et al. paper.


@profiled
def find_adversarial_grosse(
    x, clf, oracle, manifest_set, target_confidence=0.5, k=20, return_path=False
):
    if clf.predict_proba([x])[0, 1] <= target_confidence:
        raise Exception("Initial example is already classified as bening.")

    if return_path:
        path = [x]

    x_star = np.array(x, dtype="intc")
    distortions = 0

    while clf.predict_proba([x_star])[0, 1] > target_confidence and distortions < k:
        derivative = oracle.eval(x_star)
        idxs = np.argsort(derivative)

        for i, idx in enumerate(idxs):

            # Check if changing the feature is permitted.
            if x_star[idx] == 0 and idx in manifest_set:
                x_star[idx] = 1
                if return_path:
                    path.append(np.array(x_star))
                break

            if i == len(idxs) - 1:
                e = "Adversarial example is impossible to create. Tried {} distortions.".format(
                    distortions
                )
                raise Exception(e)

        distortions += 1

    if distortions == k:
        e = "Distortion bound {} reached.".format(k)
        raise Exception(e)

    if return_path:
        return x_star, distortions, path
    else:
        return x_star, distortions


# Provide implemention of our algorithm using heuristic and A* search.


@profiled
def find_adversarial(
    x,
    clf,
    search_fn,
    manifest_set,
    epsilon,
    p_norm=1,
    q_norm=np.inf,
    target_confidence=0.5,
    return_path=False,
    **kwargs
):
    """Transform an example until it is classified with target confidence."""

    def expand_fn(x, manifest_set, p_norm=1, **kwargs):
        """Wrap the example in `Node`, expand the node, and compute the costs.

        Returns a list of tuples (child, cost)
        """
        counter = Counter.get_default()
        count = counter.count()

        # Stop searching if counter limit is exceeded.
        if count > COUNTER_LIM:
            raise Exception("Counter limit reached.")

        # Debug.
        if count % DEBUG_FREQ == 0:
            logger.debug(">> (expand_fn) Node counter is: {}.".format(count))

        node = Node(x, **kwargs)
        children = node.expand(manifest_set)

        costs = [np.linalg.norm(x - c, ord=p_norm) for c in children]
        zipped = list(zip(children, costs))

        return zipped

    def goal_fn(x, clf, target_confidence=0.5):
        """Tell whether the example has reached the goal."""
        # Debug.
        counter = Counter.get_default()
        count = counter.count()
        if count % DEBUG_FREQ == 0:
            logger.debug(">> (goal_fn) Node counter is: {}.".format(count))

        is_goal = clf.predict_proba([x])[0, 1] <= target_confidence

        return is_goal

    def heuristic_fn(x, clf, manifest_set, epsilon, q_norm=np.inf):
        """Distance to the decision boundary of a logistic regression classifier.

        By default the distance is w.r.t. L1 norm. This means that the denominator
        has to be in terms of the Holder dual norm (`q_norm`), so L-inf. I know,
        this interface is horrible.

        NOTE: The value has to be zero if the example is already on the target side
        of the boundary.
        """
        # Debug.
        counter = Counter.get_default()
        count = counter.count()
        if count % DEBUG_FREQ == 0:
            logger.debug(">> (heuristic_fn start) Node counter is: {}.".format(count))

        score = clf.decision_function([x])[0]
        if score <= 0:
            return 0.0
        h = np.abs(score) / np.linalg.norm(clf.coef_[0, list(manifest_set)], ord=q_norm)

        return h * epsilon

    def hash_fn(x):
        """Hash function for examples."""
        # Debug.
        counter = Counter.get_default()
        count = counter.count()
        if count % DEBUG_FREQ == 0:
            logger.debug(">> (hash_fn start) Node counter is: {}.".format(count))

        hashed = hash(x.tostring())

        return hashed

    if clf.predict_proba([x])[0, 1] <= target_confidence:
        raise Exception("Initial example is already classified as bening.")

    return search_fn(
        start_node=x,
        expand_fn=lambda x: expand_fn(x, manifest_set, p_norm=p_norm, **kwargs),
        goal_fn=lambda x: goal_fn(x, clf, target_confidence),
        heuristic_fn=lambda x: heuristic_fn(
            x, clf, manifest_set, epsilon, q_norm=q_norm
        ),
        hash_fn=hash_fn,
        return_path=return_path,
    )


###########################################
###########################################
###########################################

# Write code to run experiments


def jsma_wrapper(
    X, neg_indices, clf, oracle, manifest_subset, target_confidence, k=20, debug=""
):
    jsma_results = {}

    # Find adversarial examples using JSMA and record their costs.
    for idx in tqdm(neg_indices, ascii=True):

        logger.debug(
            ">> {} Locating adversarial example for sample at index: {}...".format(
                debug, idx
            )
        )

        x = X[idx].toarray()[0]

        # Instantiate a profiler to analyse runtime.
        per_example_profiler = Profiler()

        with per_example_profiler.as_default():
            try:
                x_adv_jsma, cost_jsma = find_adversarial_grosse(
                    x,
                    clf,
                    oracle,
                    manifest_subset,
                    target_confidence=target_confidence,
                    k=k,
                )

                runtime_jsma = per_example_profiler.compute_stats()[
                    "find_adversarial_grosse"
                ]["tot"]
                jsma_results[idx] = (x_adv_jsma, cost_jsma, runtime_jsma)

            except Exception as e:
                logger.debug(
                    ">> {} WARN! JSMA failed for sample at index {} with the following message:\n{}".format(
                        debug, idx, e
                    )
                )
                continue

    return jsma_results


def find_adv_examples(
    X,
    target_confidence,
    confidence_margin,
    feat_count,
    epsilon,
    p_norm=1,
    q_norm=np.inf,
):
    # Define the file location to store results for given epsilon and feature count.
    file_path = "results/malware_{}_{}.pickle".format(epsilon, feat_count)

    # List for storing the results.
    results = []

    # Indices of examples classified in the (target_confidence, target_confidence+0.1) range.
    neg_indices, = np.where(
        (clf.predict_proba(X)[:, 1] > target_confidence)
        & (clf.predict_proba(X)[:, 1] < target_confidence + confidence_margin)
    )

    # Specify how many different subsets of features to choose.
    sampling_count = 25

    for i in range(sampling_count):

        batch_msg = "(Batch: {}; Feats: {}; Epsilon: {})".format(i, feat_count, epsilon)
        logger.info(
            ">> {} Using JSMA to find adversarial examples for {} samples.".format(
                batch_msg, len(neg_indices)
            )
        )

        # Choose randomly 'feat_count' features to perturb.
        manifest_subset = set(
            np.random.choice((list(MANIFEST_SET)), size=feat_count, replace=False)
        )
        assert manifest_subset.issubset(MANIFEST_SET)

        # Oracle required by the JSMA algorithm.
        oracle = LogisticRegressionScikitSaliencyOracle(clf)

        # Start by finding adversarial examples using JSMA and record their costs.
        jsma_results = jsma_wrapper(
            X,
            neg_indices,
            clf,
            oracle,
            manifest_subset,
            target_confidence,
            k=1000,
            debug=batch_msg,
        )

        logger.info(
            ">> {} JSMA found adversarial examples for {} samples.".format(
                batch_msg, len(jsma_results)
            )
        )

        # Skip this batch if no results are found by JSMA.
        if not len(jsma_results):
            logger.warning(
                ">> {} WARN! Insufficient adversarial examples returned by JSMA. Skipping...".format(
                    batch_msg
                )
            )
            continue

        # Keep only those results that have path_costs > 2.
        jsma_results = {k: v for k, v in jsma_results.items() if v[1] > 2}

        if not len(jsma_results):
            logger.warning(
                ">> {} WARN! JSMA did not find adversarial examples with required path cost. Skipping...".format(
                    batch_msg
                )
            )
            continue

        # Now only look at the malware samples with lowest path cost according to JSMA.
        jsma_results_sorted = sorted(jsma_results.items(), key=lambda d: d[1][1])[:10]

        logger.info(
            ">> {} Using IDA* search with heuristic to find adversarial examples for {} samples.".format(
                batch_msg, len(jsma_results_sorted)
            )
        )

        for idx, (x_adv_jsma, cost_jsma, runtime_jsma) in tqdm(
            jsma_results_sorted, ascii=True
        ):

            x = X[idx].toarray()[0]

            # Instantiate a counter for expanded nodes, and a profiler.
            expanded_counter = Counter()
            per_example_profiler = Profiler()

            logger.debug(
                ">> {} Locating adversarial example for sample at index: {}...".format(
                    batch_msg, idx
                )
            )

            with expanded_counter.as_default(), per_example_profiler.as_default():
                try:
                    x_adv, cost = find_adversarial(
                        x,
                        clf,
                        ida_star_search,
                        manifest_subset,
                        epsilon,
                        p_norm=1,
                        q_norm=np.inf,
                        target_confidence=target_confidence,
                    )

                except Exception as e:
                    logger.debug(
                        ">> {} WARN! IDA* search failed for sample at index {} with the following message:\n{}".format(
                            batch_msg, idx, e
                        )
                    )
                    continue

            nodes_expanded = expanded_counter.count()
            runtime = per_example_profiler.compute_stats()["find_adversarial"]["tot"]

            confidence_jsma = clf.predict_proba([x_adv_jsma])[0, 1]
            confidence = clf.predict_proba([x_adv])[0, 1]

            result = {
                "index": idx,
                "feat_count": feat_count,
                "manifest_subset": manifest_subset,
                "x_adv_jsma": x_adv_jsma,
                "path_cost_jsma": cost_jsma,
                "confidence_jsma": confidence_jsma,
                "runtime_jsma": runtime_jsma,
                "x_adv": x_adv,
                "path_cost": cost,
                "confidence": confidence,
                "nodes_expanded": nodes_expanded,
                "epsilon": epsilon,
                "runtime": runtime,
                "sampling_count": i,
            }

            results.append(result)

            logger.debug(
                ">> {} Saving intermediary results to '{}'.".format(
                    batch_msg, file_path
                )
            )

            with open(file_path, "wb") as f:
                pickle.dump(results, f)

            logger.debug(
                ">> {} Intermediary results saved to '{}'.".format(batch_msg, file_path)
            )

    return results


###########################################
###########################################
###########################################

# Main function
if __name__ == "__main__":
    logger = setup_custom_logger()

    # Define dataset locations.
    data_folder = "../data/twitter_bots/"

    human_datasets = [
        "humans.1k.csv",
        "humans.100k.csv",
        "humans.1M.csv",
        "humans.10M.csv",
    ]
    human_datasets = [data_folder + x for x in human_datasets]

    bot_datasets = ["bots.1k.csv", "bots.100k.csv", "bots.1M.csv", "bots.10M.csv"]
    bot_datasets = [data_folder + x for x in bot_datasets]
    data_files = [(x, y) for (x, y) in zip(human_datasets, bot_datasets)]

    # Define feature datasets to compare performance.
    filtered_features = [
        "follower_friend_ratio",
        "tweet_frequency",
        "favourite_tweet_ratio",
    ]
    reduced_feature_set = [x for x in FEATURES if x not in filtered_features]
    # feature_sets = [FEATURES, reduced_feature_set]
    feature_sets = [FEATURES]

    # Define bin counts to use.
    bin_counts = np.arange(5, 101, 5)

    # Perform the experiments.
    for data_file in data_files:

        logger.info(
            "Looking at files representing the {} popularity band.".format(
                data_file[0].split(".")[-2]
            )
        )

        for features in feature_sets:

            for bins in bin_counts:

                logger.info(">> Loading and preprocessing input data...")
                df = load_data(data_file)

                logger.info("Using {} bins to quantize features.".format(bins))
                df_X, df_y = feature_transform(df, features, bins=bins)

                logger.info(
                    "Examples are represented as {}-dimensional vectors.".format(
                        df_X.shape[1]
                    )
                )

                # Convert to numpy.
                X = df_X.values.astype("intc")
                y = df_y.values.astype("intc")
                logger.info("Shape of X: {}. Shape of y: {}.".format(X.shape, y.shape))

                # Split into training and test sets.
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.1, random_state=SEED
                )
                logger.info(
                    "Number of training points: {}. Number of test points: {}.".format(
                        X_train.shape[0], X_test.shape[0]
                    )
                )

                clf = fit_validate(X_train, y_train)
                train_score, test_score = (
                    clf.score(X_train, y_train) * 100,
                    clf.score(X_test, y_test) * 100,
                )
                logger.info(
                    "Resulting training accuracy is: {:.2f}%. Test accuracy is: {:.2f}%.\n".format(
                        train_score, test_score
                    )
                )

    # file_path = 'tmp/preprocessed.pickle'

    # # Try loading saved preprocessed data and classifier.
    # try:
    #     with open(file_path, 'rb') as f:
    #         logger.info(">> Loading saved preprocessed data...")
    #
    #         obj = pickle.load(f)
    #         X, y = obj['X'], obj['y']
    #         label_encoder = obj['label_encoder']
    #         clf = obj['clf']
    #
    #         # Split into training and test sets
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
    #
    # except IOError:
    #     # Load the data and record the feature set.
    #     logger.info(">> Loading data from DREBIN dataset...")
    #     data_folder = '../data/drebin/'
    #     hashes_csv = '../data/drebin_malware_sha256.csv'
    #     data, labels, features = load_data(data_folder, hashes_csv, subset=None)
    #
    #     # Fit a label encoder and transform the input data.
    #     logger.info(">> Label encoding input data...")
    #     encoded, label_encoder = fit_transform(data, features)
    #
    #     # Prepare input data for learning.
    #     logger.info(">> Preparing data for learning...")
    #     X, y = process_data(encoded, labels, features)
    #
    #     # Split into training and test sets
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
    #
    #     # Fit logistic regression by performing a Grid Search with Cross Validation.
    #     logger.info(">> Fitting logistic regression...")
    #     clf = fit_validate(X_train, y_train)
    #
    #     # Save preprocessed data to speed up subsequent experiments.
    #     obj = {
    #         'X': X,
    #         'y': y,
    #         'label_encoder': label_encoder,
    #         'clf': clf
    #     }
    #
    #     with open(file_path, 'wb') as f:
    #         pickle.dump(obj, f)
    #
    # logger.info(">> Bening samples: {}. Malware samples: {}. Total: {}.".format(y.size - sum(y), sum(y), y.size))
    #
    # # Validate the resulting classifier.
    # logger.info(">> Resulting training accuracy is: {:.2f}%. Test accuracy is: {:.2f}%."
    #             .format(clf.score(X_train, y_train)*100, clf.score(X_test, y_test)*100))
    #
    # # Set indexes for the features found in the Android manifest.
    # set_manifest_set(label_encoder)
    #
    # # Number of features to perturb.
    # feat_counts = np.arange(10, 21, 1)
    #
    # # Epsilons, i.e. coefficients for the A* search heuristic.
    # epsilons = [1]
    #
    # # Run experiments to compare Grosse et al. with JSMA against our heuristic.
    # logger.info("\n>>>> Comparing JSMA against our heuristic...")
    #
    # for feat_count in feat_counts:
    #
    #     for epsilon in epsilons:
    #
    #         logger.info(">> Running experiments for epsilon: {} and feature count: {}."
    #                         .format(epsilon, feat_count))
    #
    #         result = find_adv_examples(
    #             X,
    #             target_confidence=0.5,
    #             confidence_margin=0.2,
    #             feat_count=feat_count,
    #             epsilon=epsilon,
    #             p_norm=1,
    #             q_norm=np.inf
    #         )
