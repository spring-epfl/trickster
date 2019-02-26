#!/bin/bash

NPROC=${NPROC:=4}

# Regular experiments.
cd notebooks
papermill bots_experiment.ipynb \
    ../out/notebooks/bots_experiment__band_1k__target_50.ipynb \
    -p BAND '1k' \
    -p SAVE_PLOTS True

# High confidence experiments.
papermill bots_experiment.ipynb \
    ../out/notebooks/bots_experiment__band_1k__target_75.ipynb \
    -p BAND '1k' \
    -p TARGET_CONFIDENCE 75 \
    -p SAVE_PLOTS True \

# Heuristics comparison experiments.
papermill bots_comparisons.ipynb \
    ../out/notebooks/bots_comparisons__band_1k__target_50.ipynb \
    -p BAND '1k' \
    -p TARGET_CONFIDENCE 50 \
    -p HEURISTICS True \
    -p SAVE_PLOTS True \

# Transferability experiments.
papermill bots_transferability.ipynb \
    ../out/notebooks/bots_transferability__band_1k__target_50.ipynb \
    -p BAND '1k' \
    -p TARGET_CONFIDENCE 50

papermill bots_transferability.ipynb \
    ../out/notebooks/bots_transferability__band_1k__target_75.ipynb \
    -p BAND '1k' \
    -p TARGET_CONFIDENCE 75 \

# Non-optimal experiments
papermill bots_comparisons.ipynb \
    ../out/notebooks/bots_comparisons__band_1k__target_75.ipynb \
    -p BAND '1k' \
    -p TARGET_CONFIDENCE 75 \
    -p HEURISTICS False \
    -p SAVE_PLOTS True \

papermill bots_experiment.ipynb \
    ../out/notebooks/bots_experiment__band_1k__target_50__model_svm.ipynb \
    -p REPORT_PATH '../out/reports/bots__band_1k__target_50__model_svm.pkl' \
    -p TARGET_CONFIDENCE 50 \
    -p BAND '1k' \
    -p MODEL 'svm' \
    -p SAVE_PLOTS True
