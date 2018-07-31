#!/usr/bin/env python3

import sys
sys.path.append('..')

# Ignore warnings.
import warnings
warnings.filterwarnings('ignore')

# Handle library imports.
import numpy as np
import pandas as pd

from trickster.adversarial_helper import *
from trickster.expansion import *
from sklearn.linear_model import LogisticRegressionCV

###########################################
###########################################
###########################################

# Handle global variables.
SEED = 2018
np.random.seed(seed=SEED)

###########################################
###########################################
###########################################

# Define experiment helper functions.
def load_transform_data_fn(data_file, bins, **kwargs):
    '''
    Load and preprocess data, returning the examples and labels as numpy.
    '''
    # Load the file
    df = pd.read_csv(data_file)

    # Remove the index column.
    df = df.drop(df.columns[0], axis=1)

    # Quantize credit amount, duration and age.
    features_to_quantize = ['Credit amount', 'Duration', 'Age']
    for feat in features_to_quantize:
        series = df.loc[:, feat]
        df.loc[:, feat] = pd.qcut(series, bins, duplicates='drop')

    # Set Job type to object for one-hot encoding
    df.loc[:, 'Job'] = df.loc[:, 'Job'].astype(object)

    # Perform one-hot encoding
    df = pd.get_dummies(df)
    # Drop binary features
    df = df.drop(columns=['Sex_male', 'Risk_bad'])

    # Separate features from targets
    df_X = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    # Convert to numpy.
    X = df_X.values.astype('float')
    y = df_y.values.astype('float')

    return X, y, df_X.columns

def clf_fit_fn(X_train, y_train, **kwargs):
    '''
    Fit logistic regression by performing a Grid Search with Cross Validation.
    '''
    Cs = np.arange(0.1, 2, 0.025)
    class_weight = None # balanced or None
    scoring = 'f1' # accuracy, f1 or roc_auc

    clf = LogisticRegressionCV(
        Cs=Cs,
        cv=5,
        n_jobs=-1,
        penalty='l2',
        scoring=scoring,
        class_weight=class_weight,
        random_state=SEED
    )

    clf.fit(X_train, y_train)
    return clf

def get_expansions_fn(features, expand_quantized_fn, **kwargs):
    '''
    Define expansions to perform on features and obtain feature indexes.
    '''
    # Find indexes of required features in the original feature space.
    idxs_credit = find_substring_occurences(features, 'Credit amount')
    idxs_duration = find_substring_occurences(features, 'Duration')
    idxs_purpose = find_substring_occurences(features, 'Purpose')

    # Concatenate indexes of transformable features.
    transformable_feature_idxs = sorted(idxs_credit + idxs_duration + idxs_purpose)
    reduced_features = features[transformable_feature_idxs]

    # Find indexes of required features in the reduced feature space.
    idxs_credit = find_substring_occurences(reduced_features, 'Credit amount')
    idxs_duration = find_substring_occurences(reduced_features, 'Duration')
    idxs_purpose = find_substring_occurences(reduced_features, 'Purpose')

    # Set required expansions for features in the reduced feature space.
    expansions = [
        (idxs_credit, expand_quantized_fn),
        (idxs_duration, expand_quantized_fn),
        (idxs_purpose, expand_categorical)
    ]

    return expansions, transformable_feature_idxs

def benchmark_search_fn(**kwargs):
    '''Perform BFS adversarial example search to benchmark against A* search.'''
    heuristic_fn = lambda x, clf, epsilon, zero_to_one, q_norm: 0
    results = adversarial_search(heuristic_fn=heuristic_fn, **kwargs)
    return results

###########################################
###########################################
###########################################

# Main function.
if __name__ == "__main__":
    # Setup a custom logger.
    log_file = '../logging/credit_output.log'
    logger = setup_custom_logger(log_file)

    # Define dataset location.
    data_file = '../data/german_credit_data.csv'

    # Define experiment parameters.
    bin_counts = np.arange(5, 101, 5)
    p_norm, q_norm = 1, np.inf

    results = []

    # Perform the experiments.
    logger.info('Starting experiments for the credit fraud dataset.')

    for bins in bin_counts:

        logger.info("Loading and preprocessing input data for {} bins...".format(bins))
        result = experiment_wrapper(
            load_transform_data_fn=load_transform_data_fn,
            data_file=data_file,
            bins=bins,
            p_norm=p_norm,
            q_norm=q_norm,
            clf_fit_fn=clf_fit_fn,
            get_expansions_fn=get_expansions_fn,
            expand_quantized_fn=expand_quantized,
            benchmark_search_fn=benchmark_search_fn,
            target_confidence=0.5,
            zero_to_one=True,
            random_state=SEED,
            logger=logger
        )

        result['bins'] = bins

        results.append(result)

    import pdb; pdb.set_trace()
