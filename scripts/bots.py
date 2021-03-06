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
import pprint
import click
import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from trickster.optim import run_experiment
from trickster.optim import SpecExpandFunc
from trickster.optim import CategoricalLpProblemContext
from trickster.linear import LinearGridHeuristic, LinearHeuristic
from trickster.utils.log import setup_custom_logger
from trickster.search import a_star_search
from trickster.domain.categorical import *


def find_substring_occurences(xs, item):
    """Can be used to get the indexes of the required substring within a list of strings.

    >>> find_substring_occurences(['ab', 'bcd', 'de'], 'd')
    [1, 2]

    """

    idxs = [i for (i, x) in enumerate(xs) if item in x]
    return idxs


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


def load_transform_data(human_dataset, bot_dataset, drop_features, bins, logger, **kwargs):
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
            logger.warn("Dropping feature because only one unique value: %s" % column)
            df = df.drop(column, axis=1)
            continue

        df[column] = pd.qcut(df[column], bins, duplicates="drop")

    logger.info("Features:\n %s" % pprint.pformat(list(df.columns)))

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


def fit_lr(X_train, y_train, seed=1):
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


def fit_svmrbf(X_train, y_train, seed=1):
    """
    Fit an SVM-RBF with pre-selected hyperparameters.
    """
    clf = SVC(C=10, gamma=0.01, kernel="rbf", probability=True, random_state=seed)
    clf.fit(X_train, y_train)
    return clf


def get_expansions_specs(features=None):
    """
    Define expansions to perform on features and obtain feature indexes.

    :param features: (Quantized) feature names.
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
        FeatureExpansionSpec(idxs=idxs, expand_fn=fn, feature_name=name)
        for (name, idxs, fn) in [
            ("source_identity", idxs_source_identity, expand_collection),
            ("tweeted", idxs_tweeted, expand_quantized),
            ("retweeted", idxs_retweeted, expand_quantized),
            ("favourited", idxs_favourited, expand_quantized),
            ("replied", idxs_replied, expand_quantized),
            ("likes_per_tweet", idxs_likes_per_tweet, expand_quantized),
            ("retweets_per_tweet", idxs_retweets_per_tweet, expand_quantized),
            ("lists", idxs_lists, expand_quantized),
            ("age_of_account", idxs_age_of_account, expand_quantized_increment),
            ("sources_count", idxs_sources_count, expand_quantized),
            ("urls", idxs_urls, expand_quantized),
            ("cdn_content", idxs_cdn_content, expand_quantized),
        ]
    ]

    return expansions, transformable_feature_idxs


def get_boundaries(bucket_names):
    """
    Get boundaries of pandas-quantized features.

    >>> bucket_names = ["whatever_(100.0, 1000.0)", "whatever_(-100.0, 100.0)"]
    >>> get_boundaries(bucket_names)
    [(100.0, 1000.0), (-100.0, 100.0)]
    """
    boundaries = []
    for bucket_name in bucket_names:
        boundaries_str = bucket_name.split("_")[-1]
        lo_str, hi_str = boundaries_str.strip("[]() ").split(",")
        boundaries.append((float(lo_str), float(hi_str)))

    return boundaries


@attr.s
class FeatureDollarCostExtras:
    boundaries = attr.ib()
    price_per_unit = attr.ib()
    min_unit = attr.ib(default=1)
    normalized = attr.ib(default=False)
    override_weight = attr.ib(default=None)


def compute_buyretweet_weight_vec(example, specs, normalizer=1):
    """
    Get a weight vector for buyretweet graph from a given example and feature expansion specs.

    >>> spec = FeatureExpansionSpec(idxs=[0, 1, 2, 3], expand_fn="dummy")
    >>> spec.extras = FeatureDollarCostExtras([(1, 2), (2, 3), (3, 10), (10, 100)], 10)
    >>> example = np.array([0, 1, 0, 0, 0])
    >>> compute_buyretweet_weight_vec(example, [spec])
    array([ 0.,  0., 10., 70.,  0.])
    """
    if not isinstance(example, np.ndarray):
        example = np.array(example)

    weights = np.zeros(example.shape)
    for spec in specs:
        current_val_index = np.argmax(example[spec.idxs])
        _, current_hi_boundary = spec.extras.boundaries[current_val_index]
        for weight_index, orig_index in enumerate(spec.idxs):
            if spec.extras.override_weight is not None:
                weights[orig_index] = spec.extras.override_weight
            elif weight_index <= current_val_index:
                weights[orig_index] = 0.0
            else:
                lo_boundary, _ = spec.extras.boundaries[weight_index]
                diff = lo_boundary - current_hi_boundary
                diff = max(diff, spec.extras.min_unit)
                if diff < 1:
                    import ipdb; ipdb.set_trace()
                if spec.extras.normalized:
                    diff *= normalizer
                weights[orig_index] = diff * spec.extras.price_per_unit

    return weights


def get_buyretweet_expansions_specs(features=None):
    """
    Define expansions to perform on features and obtain feature indexes.

    :param features: (Quantized) feature names.
    """

    # Find indexes of required features in the original feature space.
    idxs_tweeted = find_substring_occurences(features, "user_tweeted")
    idxs_retweeted = find_substring_occurences(features, "user_retweeted")
    idxs_replied = find_substring_occurences(features, "user_replied")
    idxs_likes_per_tweet = find_substring_occurences(features, "likes_per_tweet")
    idxs_retweets_per_tweet = find_substring_occurences(features, "retweets_per_tweet")

    # Concatenate indexes of transformable features.
    transformable_feature_idxs = sorted(
        idxs_tweeted + idxs_retweeted + idxs_replied +
        idxs_likes_per_tweet + idxs_retweets_per_tweet)
    reduced_features = features[transformable_feature_idxs]

    # Find indexes of required features in the reduced feature space.
    idxs_tweeted = find_substring_occurences(reduced_features, "user_tweeted")
    idxs_retweeted = find_substring_occurences(reduced_features, "user_retweeted")
    idxs_replied = find_substring_occurences(reduced_features, "user_replied")
    idxs_likes_per_tweet = find_substring_occurences(
        reduced_features, "likes_per_tweet"
    )
    idxs_retweets_per_tweet = find_substring_occurences(
        reduced_features, "retweets_per_tweet"
    )

    expansions = [
        FeatureExpansionSpec(
            feature_name="user_tweeted",
            idxs=idxs_tweeted,
            expand_fn=expand_quantized_increase,
            extras=FeatureDollarCostExtras(
                boundaries=get_boundaries(reduced_features[idxs_tweeted]),
                price_per_unit=2.0,
                min_unit=1,
            ),
        ),
        # This feature is not present in the transformation graph. Only need to keep it
        # for metadata.
        FeatureExpansionSpec(
            feature_name="user_retweeted",
            idxs=idxs_retweeted,
            expand_fn=noop,
            extras=FeatureDollarCostExtras(
                boundaries=get_boundaries(reduced_features[idxs_retweeted]),
                price_per_unit=0,
                override_weight=0
            ),
        ),
        FeatureExpansionSpec(
            feature_name="user_replied",
            idxs=idxs_replied,
            expand_fn=expand_quantized_increase,
            extras=FeatureDollarCostExtras(
                boundaries=get_boundaries(reduced_features[idxs_replied]),
                price_per_unit=2.0,
                min_unit=1,
            ),
        ),
        FeatureExpansionSpec(
            feature_name="likes_per_tweet",
            idxs=idxs_likes_per_tweet,
            expand_fn=expand_quantized_increase,
            extras=FeatureDollarCostExtras(
                boundaries=get_boundaries(reduced_features[idxs_likes_per_tweet]),
                price_per_unit=0.025,
                min_unit=1,
                normalized=True
            ),
        ),
        FeatureExpansionSpec(
            feature_name="retweets_per_tweet",
            idxs=idxs_retweets_per_tweet,
            expand_fn=expand_quantized_increase,
            extras=FeatureDollarCostExtras(
                boundaries=get_boundaries(reduced_features[idxs_retweets_per_tweet]),
                price_per_unit=0.025,
                min_unit=1,
                normalized=True
            ),
        ),
    ]

    return expansions, transformable_feature_idxs


class NoCostExpandFunc(SpecExpandFunc):
    """Expand function that drops the costs."""

    def __call__(self, *args, **kwargs):
        neighbours = super().__call__(*args, **kwargs)
        return [(n, 0) for (n, c) in neighbours]


@attr.s
class RandomHeuristicProblemContext(CategoricalLpProblemContext):
    seed = attr.ib(default=1)

    def get_graph_search_problem(self, x):
        graph_search_problem = super().get_graph_search_problem(x)

        # [1, 1] is a unit difference vector for this transformation graph.
        grid_step = np.linalg.norm([1, 1], ord=self.lp_space.p)

        graph_search_problem.expand_fn = NoCostExpandFunc(problem_ctx=self)
        graph_search_problem.heuristic_fn = lambda x: random.random() * grid_step
        return graph_search_problem


@attr.s
class GridHeuristicProblemContext(CategoricalLpProblemContext):
    def get_graph_search_problem(self, x):
        graph_search_problem = super().get_graph_search_problem(x)

        # [1, 1] is a unit difference vector for this transformation graph.
        grid_step = np.linalg.norm([1, 1], ord=self.lp_space.p)

        raw_heuristic = LinearGridHeuristic(problem_ctx=self, grid_step=grid_step)
        graph_search_problem.heuristic_fn = lambda x: raw_heuristic(x.features)
        return graph_search_problem


def get_num_total_tweets(x, specs):
    """Return a lower bound on the number of tweets and retweets."""
    result = 1
    for spec in specs:
        if spec.feature_name in ["user_tweeted", "user_retweeted"]:
            idx = np.argmax(x[spec.idxs])
            lo, _ = spec.extras.boundaries[idx]
            result += lo
    return result


@attr.s
class DollarCostProblemContext(CategoricalLpProblemContext):
    features = attr.ib(default=None)

    def get_graph_search_problem(self, x):
        assert self.features is not None
        graph_search_problem = super().get_graph_search_problem(x)

        num_tweets = get_num_total_tweets(x, self.expansion_specs)
        weight_vec = compute_buyretweet_weight_vec(x, self.expansion_specs,
                normalizer=num_tweets)

        expand_fn = SpecExpandFunc(problem_ctx=self, weight_vec=weight_vec)
        graph_search_problem.expand_fn = expand_fn
        raw_heuristic = LinearHeuristic(problem_ctx=self, weight_vec=weight_vec)
        graph_search_problem.heuristic_fn = lambda x: raw_heuristic(x.features)
        return graph_search_problem


@click.command()
@click.argument("epsilons", nargs=-1, required=True, type=float)
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
    help="Popularity band (dataset parameter)",
)
@click.option(
    "--bins",
    default=None,
    show_default=True,
    type=int,
    help="Number of discretization bins.",
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
    "--reduce_classifier/--no_reduce_classifier",
    default=True,
    help="Whether to use classifier reduction optimization.",
)
@click.option(
    "--classifier",
    default="lr",
    type=click.Choice(["lr", "svmrbf"]),
    help="Target classifier",
)
@click.option(
    "--graph",
    default="all",
    type=click.Choice(["all", "buyretweet"]),
    help="Transformation graph.",
)
@click.option(
    "--heuristic",
    default="dist",
    type=click.Choice(["dist", "dist_grid", "random", "dollar"]),
    help="Heuristic",
)
@click.option(
    "--heuristic_seed",
    default="1",
    type=int,
    help="If using random heuristic, its seed.",
)
@click.option(
    "--p_norm",
    default="1",
    type=click.Choice(["1", "2", "inf"]),
    help="The p parameter of the Lp norm for computing the cost.",
)
@click.option("--beam_size", default=None, type=int, help="Size of the A* fringe.")
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
@click.option(
    "--iter_lim",
    type=int,
    default=None,
    show_default=True,
    help="Max number of search iterations until before giving up.",
)
@click.pass_context
def generate(
    ctx,
    epsilons,
    log_file,
    seed,
    popularity_band,
    bins,
    human_dataset_template,
    bot_dataset_template,
    reduce_classifier,
    p_norm,
    beam_size,
    classifier,
    graph,
    heuristic,
    heuristic_seed,
    confidence_level,
    output_pickle,
    iter_lim,
):
    np.random.seed(seed=seed)
    logger = setup_custom_logger(log_file)
    logger.info("Params:\n%s" % pprint.pformat(ctx.params))

    # Dataset locations.
    human_dataset = human_dataset_template.format(popularity_band)
    bot_dataset = bot_dataset_template.format(popularity_band)

    # Data discretization parameter.
    if bins is None:
        bin_counts = np.arange(5, 101, 5)
    else:
        bin_counts = [bins]

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
            X, y, feature_names = load_transform_data(
                human_dataset=human_dataset,
                bot_dataset=bot_dataset,
                drop_features=drop_features,
                bins=bins,
                logger=logger,
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=seed
            )

            logger.info("X_train.shape: %s" % str(X_train.shape))
            logger.info("X_test.shape: %s" % str(X_test.shape))
            logger.info("Fitting a model.")
            if classifier == "lr":
                clf = fit_lr(X_train, y_train, seed=seed)
            elif classifier == "svmrbf":
                clf = fit_svmrbf(X_train, y_train, seed=seed)

            logger.info("Model accuracy on test set: %2.2f" % clf.score(X_test, y_test))
            logger.info("Baseline on test set: %2.2f" % max(y_test.mean(), 1-y_test.mean()))

            problem_ctx_params = dict(
                clf=clf,
                target_class=0,
                target_confidence=confidence_level,
                lp_space=p_norm,
                epsilon=epsilon,
            )

            if graph == "all":
                expansion_specs, transformable_feature_idxs = get_expansions_specs(
                    feature_names
                )
                problem_ctx_params["expansion_specs"] = expansion_specs

                if heuristic == "dist":
                    problem_ctx = CategoricalLpProblemContext(**problem_ctx_params)

                elif heuristic == "dist_grid":
                    problem_ctx = GridHeuristicProblemContext(**problem_ctx_params)

                elif heuristic == "random":
                    problem_ctx = RandomHeuristicProblemContext(
                        seed=heuristic_seed, **problem_ctx_params
                    )

            elif graph == "buyretweet":
                expansion_specs, transformable_feature_idxs = get_buyretweet_expansions_specs(
                    feature_names
                )
                problem_ctx_params["expansion_specs"] = expansion_specs
                problem_ctx = DollarCostProblemContext(
                    features=feature_names, **problem_ctx_params
                )

            logger.info("Running the attack...")
            result = run_experiment(
                data=(X_test, y_test),
                problem_ctx=problem_ctx,
                graph_search_kwargs=dict(iter_lim=iter_lim, beam_size=beam_size),
                reduce_classifier=reduce_classifier,
                transformable_feature_idxs=transformable_feature_idxs,
                logger=logger,
            )

            logger.info("Success rate: %2.2f" % result["search_results"].found.mean())

            # Record extra data.
            result["features"] = feature_names
            result["bins"] = bins
            result["epsilon"] = epsilon
            result["p_norm"] = p_norm

            results.append(result)

    if output_pickle is not None:
        logger.info("Saving output to {}.".format(output_pickle))
        with open(output_pickle, "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    generate()
