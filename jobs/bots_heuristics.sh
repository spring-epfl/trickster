#!/bin/bash

NPROC=${NPROC:=4}
CONFIDENCE_LEVEL=50
BINS=20

EPSILONS="1"

echo "Generating adversarial examples with L1 grid heuristic..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/bots__band_{}__target_50__h_grid.log" \
    --popularity_band 1k \
    --bins $BINS \
    --heuristic "dist_grid" \
    --output_pickle "out/reports/bots__band_1k__target_50__h_grid.pkl"

echo "Generating adversarial examples with L2 heuristic..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/bots__band_1k__target_50__h_l2.log" \
    --popularity_band 1k \
    --bins $BINS \
    --p_norm 2 \
    --output_pickle "out/reports/bots__band_1k__target_50__h_l2.pkl"

echo "Generating adversarial examples with Linf heuristic..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/bots__band_1k__target_50__h_linf.log" \
    --popularity_band 1k \
    --bins $BINS \
    --p_norm inf \
    --output_pickle "out/reports/bots__band_1k__target_50__h_linf.pkl"

echo "Generating adversarial examples with random heuristic..."
seq 1 10 | parallel -j $NPROC \
    scripts/bots.py \
        $EPSILONS \
        --log_file "log/bots__band_1k__target_50__h_random_seed_{}.log" \
        --bins $BINS \
        --popularity_band 1k \
        --heuristic_seed {} \
        --heuristic random \
        --output_pickle "out/reports/bots__band_1k__target_50__h_random_seed_{}.pkl"

