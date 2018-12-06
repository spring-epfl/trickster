#!/usr/bin/env python3

import sys

sys.path.append("..")

# Ignore warnings.
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle

from trickster.search import a_star_search
from trickster.optim import LpCategoricalProblemContext
from trickster.optim import run_experiment
from trickster.domain.categorical import expand_categorical, expand_quantized
from trickster.domain.categorical import FeatureExpansionSpec
from trickster.utils.log import setup_custom_logger

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split


SEED = 1
np.random.seed(seed=SEED)


def find_substring_occurences(xs, item):
    """Can be used to get the indexes of the required substring within a list of strings.

    >>> find_substring_occurences(['ab', 'bcd', 'de'], 'd')
    [1, 2]

    """

    idxs = [i for (i, x) in enumerate(xs) if item in x]
    return idxs


def load_transform_data(data_file, bins):
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


def fit_clf(X_train, y_train, seed=1):
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
        random_state=seed,
    )

    clf.fit(X_train, y_train)
    return clf


def get_expansions_specs(features):
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
        FeatureExpansionSpec(idxs_credit, expand_quantized),
        FeatureExpansionSpec(idxs_duration, expand_quantized),
        FeatureExpansionSpec(idxs_purpose, expand_categorical),
    ]

    return expansions, transformable_feature_idxs


# Main function.
if __name__ == "__main__":
    # Setup a custom logger.
    log_file = "log/credit_output.log"
    logger = setup_custom_logger(log_file)

    # Dataset location.
    data_file = "data/german_credit/german_credit_data.csv"

    # Meta-experiment parameters.
    bin_levels = [5, 50] + list(range(100, 1001, 100))
    p_norm = 1
    epsilons = [0, 1, 2.5, 5, 10e5]

    results = []

    # Perform the experiments.
    logger.info("Starting experiments for the credit fraud dataset.")

    for epsilon in epsilons:

        logger.info(
            "Loading and preprocessing input data for epsilon: {}...".format(epsilon)
        )

        for bins in bin_levels:

            logger.info(
                "Loading and preprocessing input data for {} bins...".format(bins)
            )
            X, y, feature_names = load_transform_data(data_file=data_file, bins=bins)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=SEED
            )

            logger.info("Fitting a model.")
            clf = fit_clf(X_train, y_train, seed=SEED)
            expansion_specs, transformable_feature_idxs = get_expansions_specs(
                feature_names
            )

            problem_ctx = LpCategoricalProblemContext(
                clf=clf, target_class=0, target_confidence=0.5, lp_space=p_norm,
                expansion_specs=expansion_specs
            )

            logger.info("Running the attack...")
            result = run_experiment(
                data=(X_test, y_test),
                problem_ctx=problem_ctx,
                transformable_feature_idxs=transformable_feature_idxs,
                logger=logger,
            )

            result["bins"] = bins
            result["epsilon"] = epsilon
            result["p_norm"] = p_norm

            results.append(result)

    output_file = "out/reports/manual_credit.pkl"
    logger.info("Saving output to {}.".format(output_file))

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

