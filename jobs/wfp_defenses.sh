#!/bin/bash

NPROC=${NPROC:=4}

TRACE_LEN=2000
DATA_PATH=data/wfp/batch
FILTERED_DATA_PATH=data/wfp/batch__tracelen_${TRACE_LEN}
OUT_PREFIX=out/wfp_defenses/trace_len_${TRACE_LEN}

# echo "Removing existing traces..."
rm -rf ${OUT_PREFIX}/
mkdir -p ${OUT_PREFIX}

# echo "Filtering traces..."
python scripts/wfp_defenses/filter_data.py ${TRACE_LEN}

SIMPLE_DEFENSES="'buflo', 'cs_buflo', 'decoy'"
echo "Running defenses: $SIMPLE_DEFENSES..."
python -c "for defense in [$SIMPLE_DEFENSES]: print(defense)" | \
parallel -j $NPROC \
    python scripts/wfp_defenses/{}.py --data_path ${FILTERED_DATA_PATH} --out_path ${OUT_PREFIX}/{}

echo "Running WTF-Pad..."
WTF_PAD_RESULTS=scripts/wfp_defenses/wtfpad/results
python scripts/wfp_defenses/wtfpad/src/main.py ${FILTERED_DATA_PATH} --config normal_rcv > /dev/null
WTF_PAD_TRACE_DIR=$(ls -t ${WTF_PAD_RESULTS} | head -1)
mkdir -p ${OUT_PREFIX}/wtfpad
mv ${WTF_PAD_RESULTS}/${WTF_PAD_TRACE_DIR}/* ${OUT_PREFIX}/wtfpad

