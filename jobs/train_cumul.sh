#!/bin/bash

echo "Training LR..."
# LR should be pretty fast.
scripts/lr_cumul_wfp.py train \
    --model lr \
    --features cumul \
    --log_file "log/model_full_lr_cumul.log" \
    --model_pickle "out/models/model_full_lr_cumul.pkl"

echo "Training SVM-RBF..."
# SVM-RBF training should take a while.
scripts/lr_cumul_wfp.py train \
    --model svmrbf \
    --features cumul \
    --log_file "log/model_full_svmrbf_cumul.log" \
    --model_pickle "out/models/model_full_svmrbf_cumul.pkl"
