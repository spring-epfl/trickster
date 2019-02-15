#!/bin/bash

NPROC=${NPROC:=4}

BAND=1k
BINS=20
CONFIDENCE_LEVEL=50

PATH_PREFIX="bots__band_${BAND}__target_${CONFIDENCE_LEVEL}__model_svm"

EPSILONS="1 2 3 5 10 100 500 1000"

echo "Generating adversarial examples with UCS..."
    scripts/bots.py \
        0 \
        --log_file "log/${PATH_PREFIX}__ucs_only.log" \
        --popularity_band $BAND \
        --confidence_level 0.${CONFIDENCE_LEVEL} \
        --bins ${BINS} \
        --classifier svmrbf \
        --no_reduce_classifier \
        --output_pickle "out/reports/${PATH_PREFIX}__ucs_only.pkl"

echo "Generating adversarial examples with A*..."
    scripts/bots.py \
        $EPSILONS \
        --log_file "log/${PATH_PREFIX}.log" \
        --popularity_band $BAND \
        --confidence_level 0.${CONFIDENCE_LEVEL} \
        --bins ${BINS} \
        --classifier svmrbf \
        --no_reduce_classifier \
        --output_pickle "out/reports/${PATH_PREFIX}.temp.pkl"

# Adding the UCS results to the A* results.
scripts/utils/report_cat.py \
    "out/reports/${PATH_PREFIX}__ucs_only.pkl" \
    "out/reports/${PATH_PREFIX}.temp.pkl" > \
    "out/reports/${PATH_PREFIX}.pkl" && \
    rm \
    "out/reports/${PATH_PREFIX}.temp.pkl"

