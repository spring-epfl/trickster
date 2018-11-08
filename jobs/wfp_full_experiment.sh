#!/bin/bash

NPROC=${NPROC:=4}
EPSILONS="[3, 5, 7, 10, 15, 20, 25]"
ITER_LIM=5000
GENERATION_MAX_TRACE_LEN=500

jobs/train_lr_cumul.sh

echo "Generating adversarial examples..."
python -c "for eps in $EPSILONS: print(eps)" | \
    parallel -j $NPROC \
    scripts/lr_cumul_wfp.py generate \
        --log_file "log/wfp_full_eps_{}_tracelen_${GENERATION_MAX_TRACE_LEN}_iter_${ITER_LIM}.log" \
        --model_pickle "out/models/model_full_lr_cumul.pkl" \
        --epsilon {} \
        --iter_lim $ITER_LIM \
        --features cumul \
        --max_trace_len $GENERATION_MAX_TRACE_LEN \
        --sort_by_len \
        --output_pickle "out/reports/results_full_eps_{}_tracelen_${TRAINING_MAX_TRACE_LEN}_iter_${ITER_LIM}.pkl"

