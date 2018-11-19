#!/bin/bash

NPROC=${NPROC:=4}

# Should be separated by commas. Choices of bands: "'1k', '100k', '1M', '10M'"
BANDS="'1k'"

# Should be separated by commas.
CONFIDENCE_LEVELS="0.5, 0.6, 0.7, 0.8, 0.9"

# This one must not have commas. Sorry for the inconsistency.
EPSILONS="0 1 2 3 5 10 100 500 1000"


echo "Generating adversarial examples..."
echo "import itertools
for band, level in itertools.product([$BANDS], [$CONFIDENCE_LEVELS]): \
print(band, level)" | python | \
    parallel -j $NPROC --colsep ' ' \
    scripts/bots.py \
        $EPSILONS \
        --log_file "log/bots_band_{1}_target_{2}.log" \
        --popularity_band {1} \
        --confidence_level {2} \
        --output_pickle "out/reports/bots_band_{1}_target_{2}.pkl"
