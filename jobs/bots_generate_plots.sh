#!/bin/bash

NPROC=${NPROC:=4}

# Regular experiments.
cd notebooks
papermill bots_experiment.ipynb \
    ../out/notebooks/bots_experiment__band_1k.ipynb \
    -p BAND '1k' \
    -p SAVE_PLOTS True

# High confidence experiments.
papermill bots_experiment.ipynb \
    ../out/notebooks/bots_experiment__band_1k__target_75.ipynb \
    -p BAND '1k' \
    -p SAVE_PLOTS True \
    -p TARGET_CONFIDENCE 75 \

# Heuristics comparison experiments.
papermill bots_heuristics_experiment.ipynb \
    ../out/notebooks/bots_heuristics_experiment__band_1k.ipynb \
    -p BAND '1k' \
    -p SAVE_PLOTS True \
    -p TARGET_CONFIDENCE 75 \
