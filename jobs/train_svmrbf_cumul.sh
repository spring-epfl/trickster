#!/bin/bash

MODEL_NAME="model_full_svmrbf_cumul"

echo "Training a model..."
scripts/lr_cumul_wfp.py train \
    --model svmrbf \
    --features cumul \
    --log_file "log/$MODEL_NAME.log" \
    --model_pickle "out/models/$MODEL_NAME.pkl"

