#!/bin/bash

NPROC=${NPROC:=8}
EPSILONS="[100, 500, 1000, 5000, 10000]"
ITER_LIM=5000
GENERATION_MAX_TRACE_LEN=500

echo "Generating adversarial examples..."
python -c "for eps in $EPSILONS: print(eps)" | \
    parallel -j $NPROC \
    scripts/lr_cumul_wfp.py generate \
        --log_file "log/wfp_full_svm_eps_{}_tracelen_${GENERATION_MAX_TRACE_LEN}_iter_${ITER_LIM}.log" \
        --model_pickle "out/models/model_full_svmrbf_cumul.pkl" \
        --epsilon {} \
        --iter_lim $ITER_LIM \
        --heuristic svmrbf \
        --features cumul \
        --max_trace_len $GENERATION_MAX_TRACE_LEN \
        --sort_by_len \
        --dummies_per_insertion 1 \
        --output_pickle "out/reports/results_full_svm_eps_{}_tracelen_${GENERATION_MAX_TRACE_LEN}_iter_${ITER_LIM}.pkl"

