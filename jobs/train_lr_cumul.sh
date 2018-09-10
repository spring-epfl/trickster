#!/bin/bash

MODEL_NAME="model_full_lr_cumul"

echo "Training a model..."
scripts/wfp.py train \
    --model lr \
    --features cumul \
    --log_file "log/$MODEL_NAME.log" \
    --model_pickle "models/$MODEL_NAME.pkl"

