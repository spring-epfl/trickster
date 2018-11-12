#!/bin/bash

NPROC=${NPROC:=4}
BINS="[50, 100, 500, 1000]"

cd notebooks
python -c "for bins in $BINS: print(bins)" | \
    parallel -j $NPROC \
        papermill credit_experiment.ipynb \
            ../out/notebooks/credit_experiment_bin_{}.ipynb \
            -p FOCUS_BINS {} \
            -p SAVE_PLOTS True
