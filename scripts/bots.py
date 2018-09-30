#!/usr/bin/env python3

import sys

sys.path.append("..")

# Ignore warnings.
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import ast

from trickster.search import a_star_search, ida_star_search
from trickster.adversarial_helper import *
from trickster.expansion import *
from sklearn.linear_model import LogisticRegressionCV


SEED = 1
np.random.seed(seed=SEED)


def _transform_source_identity(X_k):
    """
    Helper to transform the source_identity field.
    """
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


def load_transform_data_fn(human_dataset, bot_dataset, drop_features, bins, **kwargs):
    """
    Load and preprocess data, returning the examples and labels as numpy.
    """
    # Load data for humans.
    df1 = pd.read_csv(human_dataset)
    df1 = df1.drop("screen_name", axis=1)  # remove screen_name column
    df1 = df1.assign(is_bot=0)

    # Load data for bots.
    df2 = pd.read_csv(bot_dataset)
    df2 = df2.drop("screen_name", axis=1)  # remove screen_name column
    df2 = df2.assign(is_bot=1)

    # Concatenate dataframes.
    df = df1.append(df2, ignore_index=True)

    # Drop unwanted features.
    df = df.drop(drop_features, axis=1)

    for column in df:

        # Source identity and is_bot are not quantizable.
        if column == "source_identity" or column == "is_bot":
            continue

        # Drop feature if there is only 1 distinct value.
        if np.unique(df[column]).size == 1:
            df = df.drop(column, axis=1)
            continue

        df[column] = pd.qcut(df[column], bins, duplicates="drop")

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
    df_X = df.drop("is_bot", axis=1)
    df_y = df["is_bot"]

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


def get_expansions_fn(features, **kwargs):
    """
    Define expansions to perform on features and obtain feature indexes.
    """

    # Find indexes of required features in the original feature space.
    idxs_source_identity = find_substring_occurences(features, "source_identity")
    idxs_tweeted = find_substring_occurences(features, "user_tweeted")
    idxs_retweeted = find_substring_occurences(features, "user_retweeted")
    idxs_favourited = find_substring_occurences(features, "user_favourited")
    idxs_replied = find_substring_occurences(features, "user_replied")
    idxs_likes_per_tweet = find_substring_occurences(features, "likes_per_tweet")
    idxs_retweets_per_tweet = find_substring_occurences(features, "retweets_per_tweet")
    idxs_lists = find_substring_occurences(features, "lists_per_user")
    idxs_age_of_account = find_substring_occurences(features, "age_of_account_in_days")
    idxs_sources_count = find_substring_occurences(features, "sources_count")
    idxs_urls = find_substring_occurences(features, "urls_count")
    idxs_cdn_content = find_substring_occurences(features, "cdn_content_in_kb")

    # Concatenate indexes of transformable features.
    transformable_feature_idxs = sorted(
        idxs_source_identity
        + idxs_tweeted
        + idxs_retweeted
        + idxs_favourited
        + idxs_replied
        + idxs_likes_per_tweet
        + idxs_retweets_per_tweet
        + idxs_lists
        + idxs_age_of_account
        + idxs_sources_count
        + idxs_urls
        + idxs_cdn_content
    )
    reduced_features = features[transformable_feature_idxs]

    # Find indexes of required features in the reduced feature space.
    idxs_source_identity = find_substring_occurences(
        reduced_features, "source_identity"
    )
    idxs_tweeted = find_substring_occurences(reduced_features, "user_tweeted")
    idxs_retweeted = find_substring_occurences(reduced_features, "user_retweeted")
    idxs_favourited = find_substring_occurences(reduced_features, "user_favourited")
    idxs_replied = find_substring_occurences(reduced_features, "user_replied")
    idxs_likes_per_tweet = find_substring_occurences(
        reduced_features, "likes_per_tweet"
    )
    idxs_retweets_per_tweet = find_substring_occurences(
        reduced_features, "retweets_per_tweet"
    )
    idxs_lists = find_substring_occurences(reduced_features, "lists_per_user")
    idxs_age_of_account = find_substring_occurences(
        reduced_features, "age_of_account_in_days"
    )
    idxs_sources_count = find_substring_occurences(reduced_features, "sources_count")
    idxs_urls = find_substring_occurences(reduced_features, "urls_count")
    idxs_cdn_content = find_substring_occurences(reduced_features, "cdn_content_in_kb")

    # Set required expansions for features in the reduced feature space.
    expansions = [
        (idxs_source_identity, expand_collection),
        (idxs_tweeted, expand_quantized),
        (idxs_retweeted, expand_quantized),
        (idxs_favourited, expand_quantized),
        (idxs_replied, expand_quantized),
        (idxs_likes_per_tweet, expand_quantized),
        (idxs_retweets_per_tweet, expand_quantized),
        (idxs_lists, expand_quantized),
        (idxs_age_of_account, expand_quantized_increment),
        (idxs_sources_count, expand_quantized),
        (idxs_urls, expand_quantized),
        (idxs_cdn_content, expand_quantized),
    ]

    return expansions, transformable_feature_idxs


def baseline_detaset_find_examples_fn(search_funcs=None, **kwargs):
    """Perform BFS adversarial example search to baseline against A* search."""
    search_funcs.heuristic_fn = lambda *args, **lambda_kwargs: 0
    results = dataset_find_adversarial_examples(search_funcs=search_funcs, **kwargs)
    return results


# Main function.
if __name__ == "__main__":
    # Setup a custom logger.
    log_file = "log/bots_output.log"
    logger = setup_custom_logger(log_file)

    # Perform experiments for different popularity bands.
    popularity_bands = ["1k", "100k", "1M", "10M"]

    for popularity_band in popularity_bands:

        # Define dataset locations.
        human_dataset = "data/twitter_bots/humans.{}.csv".format(popularity_band)
        bot_dataset = "data/twitter_bots/bots.{}.csv".format(popularity_band)

        # Define the meta-experiment parameters.
        bin_counts = np.arange(5, 101, 5)
        p_norm, q_norm = 1, np.inf
        epsilons = [1, 2, 3, 5, 10]

        # Define features that will be removed.
        drop_features = [
            "follower_friend_ratio",
            "tweet_frequency",
            "favourite_tweet_ratio",
        ]

        results = []

        # Perform the experiments.
        logger.info("Starting experiments for the twitter bot dataset.")

        for epsilon in epsilons:

            logger.info(
                "Loading and preprocessing input data for epsilon: {}...".format(
                    epsilon
                )
            )

            for bins in bin_counts:

                logger.info(
                    "Loading and preprocessing input data for {} bins...".format(bins)
                )
                result = experiment_wrapper(
                    load_transform_data_fn=load_transform_data_fn,
                    load_kwargs=dict(
                        human_dataset=human_dataset,
                        bot_dataset=bot_dataset,
                        drop_features=drop_features,
                        bins=bins,
                    ),
                    search_kwargs=dict(p_norm=p_norm, q_norm=q_norm, epsilon=epsilon),
                    clf_fit_fn=clf_fit_fn,
                    target_class=0,
                    get_expansions_fn=get_expansions_fn,
                    baseline_dataset_find_examples_fn=baseline_detaset_find_examples_fn,
                    logger=logger,
                    random_state=SEED,
                )

                result["bins"] = bins
                result["epsilon"] = epsilon
                result["p_norm"] = p_norm
                result["q_norm"] = q_norm

                results.append(result)

        output_file = "out/bots_{}.pickle".format(popularity_band)
        logger.info("Saving output to {}.".format(output_file))

        with open(output_file, "wb") as f:
            pickle.dump(results, f)
