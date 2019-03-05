"""
Filter traces by their length.
"""

import os
import sys

from tqdm import tqdm
from shutil import copyfile

max_trace_len = int(sys.argv[1])

DATA_PATH = "data/wfp/batch"
FILTERED_DATA_PATH = "data/wfp/batch__tracelen_%i" % max_trace_len


if not os.path.exists(FILTERED_DATA_PATH):
    os.makedirs(FILTERED_DATA_PATH)


for fname in tqdm(os.listdir(DATA_PATH)):
    infname = os.path.join(DATA_PATH, fname)
    outfname = os.path.join(FILTERED_DATA_PATH, fname)

    with open(infname, "r") as f:
        lines = f.readlines()

    # Filter.
    if len(lines) > max_trace_len:
        continue

    copyfile(infname, outfname)
