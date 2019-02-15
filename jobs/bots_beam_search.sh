#!/bin/bash

NPROC=${NPROC:=4}


echo "Generating adversarial examples with hill climbing..."
scripts/bots.py \
    1 \
    --log_file "log/bots__band_1k__target_75__hill_climbing.log" \
    --popularity_band 1k \
    --confidence_level 0.5 \
    --bins 20 \
    --beam_size 1 \
    --confidence_level 0.75 \
    --output_pickle "out/reports/bots__band_1k__target_75__hill_climbing.pkl"
