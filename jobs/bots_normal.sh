#!/bin/bash

NPROC=${NPROC:=4}

# Should be separated by commas. Choices of bands: "'1k', '100k', '1M', '10M'"
BANDS="'1k', '100k', '1M', '10M'"

# This one must not have commas. Sorry for the inconsistency.
EPSILONS="1 2 3 5 10 100 500 1000"

echo "Generating adversarial examples with UCS..."
python -c "for band in [$BANDS]: print(band)" | \
    parallel -j $NPROC \
    scripts/bots.py \
        0 \
        --log_file "log/bots__band_{}__target_50__ucs_only.log" \
        --popularity_band {} \
        --confidence_level 0.5 \
        --output_pickle "out/reports/bots__band_{}__target_50__ucs_only.pkl"

echo "Generating adversarial examples with A*..."
python -c "for band in [$BANDS]: print(band)" | \
    parallel -j $NPROC \
    scripts/bots.py \
        $EPSILONS \
        --log_file "log/bots__band_{}__target_50.log" \
        --popularity_band {} \
        --confidence_level 0.5 \
        --output_pickle "out/reports/bots__band_{}__target_50.temp.pkl"

# Adding the UCS results to the A* results.
python -c "for band in [$BANDS]: print(band)" | \
    parallel -j $NPROC \
    'scripts/utils/report_cat.py \
        "out/reports/bots__band_{}__target_50__ucs_only.pkl" \
        "out/reports/bots__band_{}__target_50.temp.pkl" > \
        "out/reports/bots__band_{}__target_50.pkl" && \
        rm \
        "out/reports/bots__band_{}__target_50.temp.pkl"'

