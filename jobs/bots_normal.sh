#!/bin/bash

NPROC=${NPROC:=4}
CONFIDENCE_LEVEL="0.5"

# Should be separated by commas. Choices of bands: "'1k', '100k', '1M', '10M'"
BANDS="'1k'"

# This one must not have commas. Sorry for the inconsistency.
EPSILONS="0 1 2 3 5 10 100 500 1000"

# NOTE: The filename format is consistent with bots_high_confidence.sh
echo "Generating adversarial examples..."
python -c "for band in [$BANDS]: print(band)" | \
    parallel -j $NPROC \
    scripts/bots.py \
        $EPSILONS \
        --log_file "log/bots_band_{}_target_${CONFIDENCE_LEVEL}.log" \
        --popularity_band {} \
        --confidence_level ${CONFIDENCE_LEVEL} \
        --output_pickle "out/reports/bots_band_{}_target_${CONFIDENCE_LEVEL}.pkl"
