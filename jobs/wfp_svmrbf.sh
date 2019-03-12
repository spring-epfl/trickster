#!/bin/bash

NPROC=${NPROC:=4}
ITER_LIM=5000
GENERATION_MAX_TRACE_LEN=2000
DUMMIES_PER_INSERTION=1

MODEL_TYPE=svmrbf jobs/train_cumul.sh

PATH_PREFIX="wfp_full__model_svmrbf__dummies_${DUMMIES_PER_INSERTION}__tracelen_${GENERATION_MAX_TRACE_LEN}__iter_${ITER_LIM}"

echo "Generating adversarial examples using hill climbing..."
scripts/wfp_attacks.py generate \
    --log_file "log/${PATH_PREFIX}.log" \
    --model_pickle "out/models/model_full_svmrbf_cumul.pkl" \
    --cost "zero" \
    --heuristic "confidence" \
    --epsilon 1 \
    --iter_lim $ITER_LIM \
    --confidence_level 0.52 \
    --features cumul \
    --max_trace_len $GENERATION_MAX_TRACE_LEN \
    --dummies_per_insertion ${DUMMIES_PER_INSERTION} \
    --sort_by_len \
    --beam_size 1 \
    --output_pickle "out/reports/${PATH_PREFIX}.pkl"

echo "Generating adversarial examples using random search-hill climbing..."
seq 1 10 | parallel -j $NPROC \
    scripts/wfp_attacks.py generate \
        --log_file "log/${PATH_PREFIX}__random__seed_{}.log" \
        --model_pickle "out/models/model_full_svmrbf_cumul.pkl" \
        --cost "zero" \
        --heuristic "random" \
        --heuristic_seed {} \
        --epsilon 1 \
        --iter_lim $ITER_LIM \
        --features cumul \
        --max_trace_len $GENERATION_MAX_TRACE_LEN \
        --dummies_per_insertion ${DUMMIES_PER_INSERTION} \
        --sort_by_len \
        --beam_size 1 \
        --output_pickle "out/reports/${PATH_PREFIX}__random_{}.pkl"

