#!/bin/bash

NPROC=${NPROC:=4}

FOCUS_BINS="20"

# Should be separated by commas. Choices of bands: "'1k', '100k', '1M', '10M'"
BANDS="'1k'"

# Should be separated by commas.
CONFIDENCE_LEVELS="0.75, 0.95"

# This one must not have commas. Sorry for the inconsistency.
EPSILONS="0 1 5 10"


echo "Generating adversarial examples..."
echo "import itertools
for band, level in itertools.product([$BANDS], [$CONFIDENCE_LEVELS]): \
print(band, level)" | python | \
    parallel -j $NPROC --colsep ' ' \
    scripts/bots.py \
        $EPSILONS \
        --log_file "log/bots_band_{1}_target_{2}.log" \
        --bins $FOCUS_BINS \
        --popularity_band {1} \
        --confidence_level {2} \
        --output_pickle "out/reports/bots_band_{1}_target_{2}.pkl"
