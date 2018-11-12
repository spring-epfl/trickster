#!/usr/bin/env python3

import sys

sys.path.append("..")

# Ignore warnings.
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle

from trickster.search import a_star_search, ida_star_search
from trickster.adversarial_helper import *
from trickster.expansions import *
from sklearn.linear_model import LogisticRegressionCV


SEED = 1
np.random.seed(seed=SEED)


def load_transform_data_fn(data_file, bins, **kwargs):
    """
    Load and preprocess data, returning the examples and labels as numpy.
    """
    # Load the file
    df = pd.read_csv(data_file)

    # Remove the index column.
    df = df.drop(df.columns[0], axis=1)

    # Quantize credit amount, duration and age.
    features_to_quantize = ["Credit amount", "Duration", "Age"]
    for feat in features_to_quantize:
        series = df.loc[:, feat]
        df.loc[:, feat] = pd.qcut(series, bins, duplicates="drop")

    # Set Job type to object for one-hot encoding
    df.loc[:, "Job"] = df.loc[:, "Job"].astype(object)

    # Perform one-hot encoding
    df = pd.get_dummies(df)
    # Drop binary features
    df = df.drop(columns=["Sex_male", "Risk_bad"])

    # Separate features from targets
    df_X = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    # Convert to numpy.
    X = df_X.values.astype("float")
    y = df_y.values.astype("float")

    return X, y, df_X.columns


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

    clf.fit(X_train, y_train)
    return clf


def get_expansions_fn(features, expand_quantized_fn, **kwargs):
    """
    Define expansions to perform on features and obtain feature indexes.
    """
    # Find indexes of required features in the original feature space.
    idxs_credit = find_substring_occurences(features, "Credit amount")
    idxs_duration = find_substring_occurences(features, "Duration")
    idxs_purpose = find_substring_occurences(features, "Purpose")

    # Concatenate indexes of transformable features.
    transformable_feature_idxs = sorted(idxs_credit + idxs_duration + idxs_purpose)
    reduced_features = features[transformable_feature_idxs]

    # Find indexes of required features in the reduced feature space.
    idxs_credit = find_substring_occurences(reduced_features, "Credit amount")
    idxs_duration = find_substring_occurences(reduced_features, "Duration")
    idxs_purpose = find_substring_occurences(reduced_features, "Purpose")

    # Set required expansions for features in the reduced feature space.
    expansions = [
        (idxs_credit, expand_quantized_fn),
        (idxs_duration, expand_quantized_fn),
        (idxs_purpose, expand_categorical),
    ]

    return expansions, transformable_feature_idxs


def baseline_dataset_find_examples_fn(search_funcs=None, **kwargs):
    """Perform BFS adversarial example search to baseline against A* search."""
    search_funcs.heuristic_fn = lambda *args, **lambda_kwargs: 0
    results = dataset_find_adversarial_examples(search_funcs=search_funcs, **kwargs)
    return results


# Main function.
if __name__ == "__main__":
    # Setup a custom logger.
    log_file = "log/credit_output.log"
    logger = setup_custom_logger(log_file)

    # Dataset location.
    data_file = "notebooks/data/german_credit_data.csv"

    # Meta-experiment parameters.
    bin_counts = [5, 50] + list(range(100, 1001, 100))
    p_norm, q_norm = 1, np.inf
    epsilons = [0, 1, 2.5, 5, 10e+5]

    results = []

    # Perform the experiments.
    logger.info("Starting experiments for the credit fraud dataset.")

    for epsilon in epsilons:

        logger.info(
            "Loading and preprocessing input data for epsilon: {}...".format(epsilon)
        )

        for bins in bin_counts:

            logger.info(
                "Loading and preprocessing input data for {} bins...".format(bins)
            )
            result = experiment_wrapper(
                load_transform_data_fn=load_transform_data_fn,
                load_kwargs=dict(data_file=data_file, bins=bins),
                search_kwargs=dict(p_norm=p_norm, q_norm=q_norm, epsilon=epsilon),
                clf_fit_fn=clf_fit_fn,
                target_class=1,
                get_expansions_fn=get_expansions_fn,
                get_expansions_kwargs=dict(expand_quantized_fn=expand_quantized),
                baseline_dataset_find_examples_fn=baseline_dataset_find_examples_fn,
                logger=logger,
                random_state=SEED,
            )

            result["bins"] = bins
            result["epsilon"] = epsilon
            result["p_norm"] = p_norm
            result["q_norm"] = q_norm

            results.append(result)

    output_file = "out/reports/credit.pkl"
    logger.info("Saving output to {}.".format(output_file))

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

