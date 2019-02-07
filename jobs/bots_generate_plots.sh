#!/bin/bash

NPROC=${NPROC:=4}

# Regular experiments.
cd notebooks
papermill bots_experiment.ipynb \
    ../out/notebooks/bots_experiment_band_1k.ipynb \
    -p BAND '1k' \
    -p SAVE_PLOTS True

# High confidence experiments.
papermill bots_experiment.ipynb \
    ../out/notebooks/bots_experiment_band_1k_target_0.75.ipynb \
    -p BAND '1k' \
    -p SAVE_PLOTS True \
    -p TARGET_CONFIDENCE 0.75 \
