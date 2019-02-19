# Based on: http://home.cse.ust.hk/~taow/wf/defenses/
# Author: Tao Wang and collaborators.


import os
import sys
import numpy as np
from shutil import copyfile

from tqdm import tqdm


def openandfind(path, name):
    if os.path.exists(path + str(name)):
        f2 = open(path + str(count), "r")
        lines2 = f2.readlines()
        f2.close()
        if len(lines2) > 500:
            return False
        else:
            return True
    else:
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Need max trace length parameter")

    max_trace_len = int(sys.argv[1])

    full_data_path = "data/wfp/batch/"
    data_path = "data/wfp/batch__tracelen_%i/" % max_trace_len
    out_path = "wfp/decoy__tracelen_%i/" % max_trace_len

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if not os.path.exists(data_path):
        print("batch folder needs to exist")
        sys.exit(0)

    for site in tqdm(range(0, 100)):
        for inst in range(0, 90):
            if not os.path.exists(data_path + str(site) + "-" + str(inst)):
                continue
            f1 = open(data_path + str(site) + "-" + str(inst), "r")
            lines1 = f1.readlines()
            f1.close()
            found = False
            while found == False:
                count = np.random.randint(0, 9000)
                found = openandfind(full_data_path, count)

            f2 = open(full_data_path + str(count), "r")
            lines2 = f2.readlines()
            f2.close()
            start1 = float(lines1[0].split("\t")[0])
            start2 = float(lines2[0].split("\t")[0])
            packets = []
            for x in lines1:
                x = x.split("\t")
                packets.append([float(x[0]) - start1, x[1]])
            for x in lines2:
                x = x.split("\t")
                packets.append([float(x[0]) - start2, x[1]])
            packets = sorted(packets, key=lambda packets: packets[0])
            fout = open(out_path + str(site) + "-" + str(inst), "w")
            for x in packets:
                fout.write(str(x[0]) + "\t" + x[1])
            fout.close()
