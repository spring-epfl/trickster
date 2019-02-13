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
    -p SAVE_PLOTS True \
    -p TARGET_CONFIDENCE 75 \

# Heuristics comparison experiments.
papermill bots_heuristics.ipynb \
    ../out/notebooks/bots_heuristics__band_1k.ipynb \
    -p BAND '1k' \
    -p SAVE_PLOTS True \
    -p TARGET_CONFIDENCE 50 \

# Transferability experiments.
papermill bots_transferability.ipynb \
    ../out/notebooks/bots_transferability__band_1k__target_50.ipynb \
    -p BAND '1k' \
    -p TARGET_CONFIDENCE 50

papermill bots_transferability.ipynb \
    ../out/notebooks/bots_transferability__band_1k__target_75.ipynb \
    -p BAND '1k' \
    -p TARGET_CONFIDENCE 75 \
    -p REPORT_PATH '../out/reports/bots__band_1k__target_75.pkl' \

