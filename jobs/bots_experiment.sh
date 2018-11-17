#!/bin/bash

NPROC=${NPROC:=4}
BANDS="'1k', '100k', '1M', '10M'"
EPSILONS="0, 1, 2, 3, 5, 10, 5, 100, 500, 1000"
CONFIDENCE_LEVELS="0.5, 0.6, 0.7, 0.8, 0.9"


echo "Generating adversarial examples..."
python -c "from itertools import product; for band, level in product( \
           [$BANDS], [$CONFIDENCE_LEVELS]): print(band, level)" | \
    parallel -j $NPROC --colsep ' ' \
    scripts/bots.py \
        --log_file "log/bots_band_{1}_target_{2}.log" \
        --popularity_band {1} \
        --epsilons $EPSILONS \
        --confidence_level $CONFIDENCE_LEVELS \
        --output_pickle "out/reports/bots_band_{1}_target_{2}.pkl"
