#!/bin/bash

BAND=1k
BINS=20
CONFIDENCE_LEVEL=75

PATH_PREFIX="bots__band_${BAND}__target_${CONFIDENCE_LEVEL}"

EPSILONS="1 2 3 5 10 100 500 1000"

echo "Generating adversarial examples with UCS..."
scripts/bots.py \
    0 \
    --log_file "log/${PATH_PREFIX}__ucs_only.log" \
    --popularity_band $BAND \
    --bins $BINS \
    --confidence_level 0.${CONFIDENCE_LEVEL} \
    --output_pickle "out/reports/${PATH_PREFIX}__ucs_only.pkl"

echo "Generating adversarial examples..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/${PATH_PREFIX}.log" \
    --bins $BINS \
    --popularity_band ${BAND} \
    --confidence_level 0.${CONFIDENCE_LEVEL} \
    --output_pickle "out/reports/${PATH_PREFIX}.temp.pkl"

# Adding the UCS results to the A* results.
scripts/utils/report_cat.py \
    "out/reports/${PATH_PREFIX}__ucs_only.pkl" \
    "out/reports/${PATH_PREFIX}.temp.pkl" > \
    "out/reports/${PATH_PREFIX}.pkl" && \
    rm \
    "out/reports/${PATH_PREFIX}.temp.pkl"

echo "Generating adversarial examples using hill climbing..."
scripts/bots.py \
    1 \
    --log_file "log/${PATH_PREFIX}__hill_climbing.log" \
    --bins $BINS \
    --popularity_band ${BAND} \
    --confidence_level 0.${CONFIDENCE_LEVEL} \
    --beam_size 1 \
    --output_pickle "out/reports/${PATH_PREFIX}__hill_climbing.pkl"
