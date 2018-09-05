#!/bin/bash

NPROC=${NPROC:=4}
NUM_MODELS=20
TRAIN_DATASET_SIZE=112
GENERATION_DATASET_SIZE=100
GENERATION_MAX_TRACE_LEN=500
ITER_LIM=100

echo "Training small models..."
seq $NUM_MODELS | \
    parallel -j $NPROC \
    scripts/wfp.py train \
        --seed {} \
        --log_file "log/wfo_small_models_{}.log" \
        --shuffle \
        --num_traces $TRAIN_DATASET_SIZE \
        --model lr \
        --features cumul \
        --model_pickle "models/small_model_{}.pkl"

echo "Generating adversarial examples..."
seq $NUM_MODELS | \
    parallel -j $NPROC \
    scripts/wfp.py generate \
        --log_file "log/wfo_small_models_{}.log" \
        --model_pickle "models/small_model_{}.pkl" \
        --iter_lim $ITER_LIM \
        --features cumul \
        --num_traces $GENERATION_DATASET_SIZE \
        --max_trace_len $GENERATION_MAX_TRACE_LEN \
        --output_pickle "out/results_small_model_{}.pkl"
