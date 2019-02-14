#!/bin/bash

NPROC=${NPROC:=4}

EPSILONS="1 2 3 5 10 100 500 1000"

echo "Generating adversarial examples with UCS..."
    scripts/bots.py \
        0 \
        --log_file "log/bots__band_1k__target_50__model_svm__ucs_only.log" \
        --popularity_band 1k \
        --confidence_level 0.5 \
        --bins 20 \
        --classifier svmrbf \
        --no_reduce_classifier \
        --output_pickle "out/reports/bots__band_1k__target_50__model_svm__ucs_only.pkl"

echo "Generating adversarial examples with A*..."
    scripts/bots.py \
        $EPSILONS \
        --log_file "log/bots__band_1k__target_50__model_svm.log" \
        --popularity_band 1k \
        --confidence_level 0.5 \
        --bins 20 \
        --classifier svmrbf \
        --no_reduce_classifier \
        --output_pickle "out/reports/bots__band_1k__target_50__model_svm.temp.pkl"

# Adding the UCS results to the A* results.
scripts/utils/report_cat.py \
    "out/reports/bots__band_1k__target_50__model_svm__ucs_only.pkl" \
    "out/reports/bots__band_1k__target_50__model_svm.temp.pkl" > \
    "out/reports/bots__band_1k__target_50__model_svm.pkl" && \
    rm \
    "out/reports/bots__band_1k__target_50__model_svm.temp.pkl"

