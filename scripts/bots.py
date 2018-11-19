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
import click

from trickster.search import a_star_search, ida_star_search
from trickster.adversarial_helper import *
from trickster.expansions import *
from trickster.utils.norms import get_holder_conjugates
from sklearn.linear_model import LogisticRegressionCV


def _transform_source_identity(X_k, sources_count=7):
    """
    Helper to transform the source_identity field.
    """
    X_k = X_k.apply(lambda x: x.replace(";", ","))
    X_k = X_k.apply(ast.literal_eval)

    N, K = X_k.shape[0], sources_count * 2
    X_k_transformed = np.zeros((N, K), dtype="intc")

    # Set (1, 0) if the source is present for the user and (0, 1) if absent.
    for i in range(N):
        for j in range(sources_count):
            if j in X_k[i]:
                X_k_transformed[i, j * 2] = 1
            else:
                X_k_transformed[i, j * 2 + 1] = 1

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

    df["source_identity_other_present"] = transformed[:, 0]
    df["source_identity_other_absent"] = transformed[:, 1]
    df["source_identity_browser_present"] = transformed[:, 2]
    df["source_identity_browser_absent"] = transformed[:, 3]
    df["source_identity_mobile_present"] = transformed[:, 4]
    df["source_identity_mobile_absent"] = transformed[:, 5]
    df["source_identity_osn_present"] = transformed[:, 6]
    df["source_identity_osn_absent"] = transformed[:, 7]
    df["source_identity_automation_present"] = transformed[:, 8]
    df["source_identity_automation_absent"] = transformed[:, 9]
    df["source_identity_marketing_present"] = transformed[:, 10]
    df["source_identity_marketing_absent"] = transformed[:, 11]
    df["source_identity_news_present"] = transformed[:, 12]
    df["source_identity_news_absent"] = transformed[:, 13]

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
        # FIXME: Use the supplied seed.
        random_state=1,
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


# Not used.
def baseline_detaset_find_examples_fn(search_funcs=None, **kwargs):
    """Perform BFS adversarial example search to baseline against A* search."""
    search_funcs.heuristic_fn = lambda *args, **lambda_kwargs: 0
    results = dataset_find_adversarial_examples(search_funcs=search_funcs, **kwargs)
    return results


@click.command()
@click.argument(
    "epsilons",
    nargs=-1,
    type=float
)
@click.option(
    "--log_file",
    default="log/bots_output.log",
    type=click.Path(),
    help="Log file path.",
)
@click.option("--seed", default=1, type=int, help="Random seed.")
@click.option(
    "--popularity_band",
    default="1k",
    show_default=True,
    type=click.Choice(["1k", "100k", "1M", "10M"]),
    help="Popularity band (dataset parameter)"
)
@click.option(
    "--human_dataset_template",
    default="data/twitter_bots/humans/humans.{}.csv",
    show_default=True,
)
@click.option(
    "--bot_dataset_template",
    default="data/twitter_bots/bots/bots.{}.csv",
    show_default=True,
)

@click.option(
    "--p_norm",
    default="1",
    type=click.Choice(["1", "2", "inf"]),
    help="The p parameter of the Lp norm for computing the cost.",
)
@click.option(
    "--confidence_level",
    default=0.5,
    show_default=True,
    help="Target confidence level.",
)
@click.option(
    "--output_pickle",
    type=click.Path(exists=False, dir_okay=False),
    help="Output results dataframe pickle.",
)
def generate(
    epsilons,
    log_file,
    seed,
    popularity_band,
    human_dataset_template,
    bot_dataset_template,
    p_norm,
    confidence_level,
    output_pickle,
):
    np.random.seed(seed=seed)
    logger = setup_custom_logger(log_file)
    p_norm, q_norm = get_holder_conjugates(p_norm)

    # Dataset locations.
    human_dataset = human_dataset_template.format(popularity_band)
    bot_dataset = bot_dataset_template.format(popularity_band)

    # The meta-experiment parameters.
    bin_counts = np.arange(5, 101, 5)

    # Features that will be removed.
    drop_features = [
        "follower_friend_ratio",
        "tweet_frequency",
        "favourite_tweet_ratio",
    ]

    results = []

    logger.info("Starting experiments for the Twitter bot dataset.")

    # Perform the experiments.
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
                load_kwargs=dict(
                    human_dataset=human_dataset,
                    bot_dataset=bot_dataset,
                    drop_features=drop_features,
                    bins=bins,
                ),
                search_kwargs=dict(p_norm=p_norm, q_norm=q_norm, epsilon=epsilon),
                clf_fit_fn=clf_fit_fn,
                target_class=0,
                target_confidence=confidence_level,
                get_expansions_fn=get_expansions_fn,
                logger=logger,
                random_state=seed,
            )

            # Record extra data.
            result["bins"] = bins
            result["epsilon"] = epsilon
            result["p_norm"] = p_norm
            result["q_norm"] = q_norm

            results.append(result)

    logger.info("Saving output to {}.".format(output_pickle))

    with open(output_pickle, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    generate()
