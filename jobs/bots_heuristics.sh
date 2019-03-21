#!/bin/bash

NPROC=${NPROC:=4}
BINS=20
BAND=1k
CONFIDENCE_LEVEL=50

PATH_PREFIX="bots__band_${BAND}__target_${CONFIDENCE_LEVEL}"

EPSILONS="1"

echo "Generating adversarial examples with L2 heuristic..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/${PATH_PREFIX}__h_l2.log" \
    --popularity_band ${BAND} \
    --confidence_level 0.${CONFIDENCE_LEVEL} \
    --bins $BINS \
    --p_norm 2 \
    --output_pickle "out/reports/${PATH_PREFIX}__h_l2.pkl"

echo "Generating adversarial examples with Linf heuristic..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/${PATH_PREFIX}__h_linf.log" \
    --popularity_band ${BAND} \
    --confidence_level 0.${CONFIDENCE_LEVEL} \
    --bins $BINS \
    --p_norm inf \
    --output_pickle "out/reports/${PATH_PREFIX}__h_linf.pkl"

echo "Generating adversarial examples with hill climbing/L1 heuristic..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/${PATH_PREFIX}__hill_climbing.log" \
    --popularity_band ${BAND} \
    --confidence_level 0.${CONFIDENCE_LEVEL} \
    --bins $BINS \
    --beam_size 1 \
    --output_pickle "out/reports/${PATH_PREFIX}__hill_climbing.pkl"

echo "Generating adversarial examples with random heuristic..."
seq 1 10 | parallel -j $NPROC \
    scripts/bots.py \
        $EPSILONS \
        --log_file "log/${PATH_PREFIX}__h_random_seed_{}.log" \
        --bins $BINS \
        --confidence_level 0.${CONFIDENCE_LEVEL} \
        --popularity_band ${BAND} \
        --heuristic_seed {} \
        --heuristic random \
        --output_pickle "out/reports/${PATH_PREFIX}__h_random_seed_{}.pkl"
