# Based on: http://home.cse.ust.hk/~taow/wf/defenses/
# Author: Tao Wang and collaborators.

import os
import sys
import random

import click
from shutil import copyfile

from tqdm import tqdm


@click.command()
@click.option("--data_path")
@click.option("--out_path")
@click.option("--seed", default=1)
def main(data_path, out_path, seed):
    random.seed(seed)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if not os.path.exists(data_path):
        print("batch folder needs to exist")
        sys.exit(0)

    num_traces = os.listdir(data_path)

    for site in tqdm(range(0, 100)):
        for inst in range(0, 90):
            inst_path = os.path.join(data_path, "%i-%i" % (site, inst))
            if not os.path.exists(inst_path):
                continue

            with open(inst_path, "r") as f1:
                lines1 = f1.readlines()

            non_monitored_traces = [f for f in os.listdir(data_path) if "-" not in f]
            camo_trace_path = random.choice(non_monitored_traces)
            with open(os.path.join(data_path, camo_trace_path), "r") as f2:
                lines2 = f2.readlines()

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

            with open(os.path.join(out_path, "%i-%i" % (site, inst)), "w") as fout:
                for x in packets:
                    fout.write(str(x[0]) + "\t" + x[1])


if __name__ == "__main__":
    main()
