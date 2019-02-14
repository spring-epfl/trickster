#!/bin/bash

NPROC=${NPROC:=4}
BINS=20
BAND=1k

EPSILONS="1"

echo "Generating adversarial examples with L1 grid heuristic..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/bots__band_${BAND}__target_50__h_grid.log" \
    --popularity_band ${BAND} \
    --bins $BINS \
    --heuristic "dist_grid" \
    --output_pickle "out/reports/bots__band_${BAND}__target_50__h_grid.pkl"

echo "Generating adversarial examples with L2 heuristic..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/bots__band_${BAND}__target_50__h_l2.log" \
    --popularity_band ${BAND} \
    --bins $BINS \
    --p_norm 2 \
    --output_pickle "out/reports/bots__band_${BAND}__target_50__h_l2.pkl"

echo "Generating adversarial examples with Linf heuristic..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/bots__band_${BAND}__target_50__h_linf.log" \
    --popularity_band ${BAND} \
    --bins $BINS \
    --p_norm inf \
    --output_pickle "out/reports/bots__band_${BAND}__target_50__h_linf.pkl"

echo "Generating adversarial examples with hill climbing/L1 heuristic..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/bots__band_${BAND}__target_50__hill_climbing.log" \
    --popularity_band ${BAND} \
    --bins $BINS \
    --beam_size 1 \
    --output_pickle "out/reports/bots__band_${BAND}__target_50__hill_climbing.pkl"

echo "Generating adversarial examples with random heuristic..."
seq 1 10 | parallel -j $NPROC \
    scripts/bots.py \
        $EPSILONS \
        --log_file "log/bots__band_${BAND}__target_50__h_random_seed_{}.log" \
        --bins $BINS \
        --popularity_band ${BAND} \
        --heuristic_seed {} \
        --heuristic random \
        --output_pickle "out/reports/bots__band_${BAND}__target_50__h_random_seed_{}.pkl"

