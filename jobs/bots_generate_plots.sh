#!/bin/bash

NPROC=${NPROC:=4}
BANDS="['1k', '100k', '1M', '10M']"

cd notebooks
python -c "for band in $BANDS: print(band)" | \
    parallel -j $NPROC \
        papermill bots_experiment.ipynb \
            ../out/notebooks/bots_experiment_{}.ipynb \
            -p BAND {} \
            -p SAVE_PLOTS True
