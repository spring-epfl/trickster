#!/bin/bash

BINS=20

# This one must not have commas.
EPSILONS="0 1 2 3 5 10 100 500 1000"

echo "Generating adversarial examples..."
scripts/bots.py \
    $EPSILONS \
    --log_file "log/bots__band_1k__target_75.log" \
    --bins $BINS \
    --popularity_band 1k \
    --confidence_level 0.75 \
    --output_pickle "out/reports/bots__band_1k__target_75.pkl"

