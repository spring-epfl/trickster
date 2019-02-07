#!/bin/bash

FOCUS_BINS="20"

# This one must not have commas.
EPSILONS="0 1 2 3 5 10 100 500 1000"

echo "Generating adversarial examples..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/bots_band_1k_target_0.75.log" \
    --bins $FOCUS_BINS \
    --popularity_band 1k \
    --confidence_level 0.75 \
    --no_reduce_classifier \
    --output_pickle "out/reports/bots_band_1k_target_0.75_more.pkl"

