#!/bin/bash

NPROC=${NPROC:=4}
BANDS="['1k', '100k', '1M', '10M']"

# Regular experiments.
cd notebooks
python -c "for band in $BANDS: print(band)" | \
    parallel -j $NPROC \
        papermill bots_experiment.ipynb \
            ../out/notebooks/bots_experiment_{}.ipynb \
            -p BAND {} \
            -p SAVE_PLOTS True

# High confidence experiments.
papermill bots_experiment.ipynb \
    ../out/notebooks/bots_experiment_target_0.75.ipynb \
    -p BAND '1k' \
    -p SAVE_PLOTS True \
    -p TARGET_CONFIDENCE 0.75 \
