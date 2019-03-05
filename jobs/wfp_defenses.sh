#!/bin/bash

NPROC=${NPROC:=4}

TRACE_LEN=2000
DATA_PATH=data/wfp/batch
FILTERED_DATA_PATH=data/wfp/batch__tracelen_${TRACE_LEN}
OUT_PREFIX=out/wfp_defenses
DEFENSES="'buflo', 'cs_buflo', 'tamaraw', 'decoy'"

echo "Filtering traces..."
python scripts/wfp_defenses/filter_data.py ${TRACE_LEN}

echo "Running defenses: $DEFENSES..."
python -c "for defense in [$DEFENSES]: print(defense)" | \
parallel -j $NPROC \
    python scripts/wfp_defenses/{}.py --data_path ${FILTERED_DATA_PATH} --out_path ${OUT_PREFIX}/{}

echo "Running WTF-Pad..."
python scripts/wfp_defenses/wtfpad/src/main.py ${FILTERED_DATA_PATH} > /dev/null
WTF_PAD_RESULTS=scripts/wfp_defenses/wtfpad/results
WTF_PAD_TRACE_DIR=$(ls -t ${WTF_PAD_RESULTS} | head -1)
mv ${WTF_PAD_RESULTS}/${WTF_PAD_TRACES}/* ${OUT_PREFIX}/wtfpad/

