#!/bin/bash

MODEL_TYPE=${MODEL_TYPE="lr"}
MODEL_NAME="model_full_$MODEL_TYPE_cumul"

echo "Training a model..."
scripts/lr_cumul_wfp.py train \
    --model $MODEL_TYPE \
    --features cumul \
    --log_file "log/$MODEL_NAME.log" \
    --model_pickle "out/models/$MODEL_NAME.pkl"

