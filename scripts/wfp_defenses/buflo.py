# Based on: http://home.cse.ust.hk/~taow/wf/defenses/
# Author: Tao Wang and collaborators.

# Generates 1-X from 0-X.

import os
import sys
import math
import random

import click

from tqdm import tqdm


def fsign(num):
    if num > 0:
        return 0
    else:
        return 1


def defend(list1, list2, r_in=0.02, r_out=0.02, mintime=42):
    datasize = 1

    buf = [0, 0]
    listind = 0  # marks the next packet to send
    starttime = list1[0][0]
    lastpostime = starttime
    lastnegtime = starttime
    curtime = starttime
    count = [0, 0]
    lastind = [0, 0]
    for i in range(0, len(list1)):
        if list1[i][1] > 0:
            lastind[0] = i
        else:
            lastind[1] = i
    defintertime = [[r_out], [r_in]]
    while listind < len(list1) or buf[0] + buf[1] > 0 or curtime < starttime + mintime:
        # print "Send packet, buffers", buf[0], buf[1], "listind", listind
        # decide which packet to send

        # if one direction packets end, and curtime is larger than threshold, then end that direction with the other direction continue to transmit in constant interval

        if curtime >= starttime + mintime:
            for j in range(0, 2):
                if listind > lastind[j]:
                    defintertime[j][0] = 10000
        ind = int((curtime - starttime) * 10)
        if ind >= len(defintertime[0]):
            ind = len(defintertime[0]) // 2  # ?????
        if lastpostime + defintertime[0][ind] < lastnegtime + defintertime[1][ind]:
            cursign = 0
            curtime = lastpostime + defintertime[0][ind]
            lastpostime += defintertime[0][ind]
        else:
            cursign = 1
            curtime = lastnegtime + defintertime[1][ind]
            lastnegtime += defintertime[1][ind]
        # check if there's data remaining to be sent

        # tosend: a packet ship describing the room left
        # if a packet is larger than the room left, buf contains the remains util
        # next ship comes to clear the buf.
        tosend = datasize
        if buf[cursign] > 0:
            if buf[cursign] <= datasize:
                tosend -= buf[cursign]
                buf[cursign] = 0
                listind += 1
            else:
                tosend = 0
                buf[cursign] -= datasize
        if listind < len(list1):
            while (
                list1[listind][0] <= curtime
                and fsign(list1[listind][1]) == cursign
                and tosend > 0
            ):
                if tosend >= abs(list1[listind][1]):
                    tosend -= abs(list1[listind][1])
                    listind += 1
                else:
                    buf[cursign] = abs(list1[listind][1]) - tosend
                    tosend = 0
                if listind >= len(list1):
                    break
        if cursign == 0:
            list2.append([curtime, datasize])
        else:
            list2.append([curtime, -datasize])
        count[cursign] += 1


@click.command()
@click.option("--data_path")
@click.option("--out_path")
@click.option("--r_in", default=0.02)
@click.option("--r_out", default=0.02)
@click.option("--mintime", default=42)
def main(data_path, out_path, r_in, r_out, mintime):
    # src pathes to consider
    crawl_src = list()

    s_sitenum = 0
    e_sitenum = 100
    instnum = 90

    max_time = 0

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for site in tqdm(range(s_sitenum, e_sitenum)):
        for inst in range(0, instnum):
            packets = []
            inst_path = os.path.join(data_path, "%i-%i" % (site, inst))
            if os.path.isfile(inst_path):
                with open(inst_path, "r") as f:
                    lines = f.readlines()

                if float(lines[-1].split("\t")[0]) > max_time:
                    max_time = float(lines[-1].split("\t")[0])
                starttime = float(lines[0].split("\t")[0])
                for x in lines:
                    x = x.split("\t")
                    packets.append([float(x[0]) - starttime, int(x[1])])

                list2 = []
                defend(packets, list2, r_in=r_in, r_out=r_out, mintime=mintime)
                list2 = sorted(list2, key=lambda list2: list2[0])

                with open(os.path.join(out_path, "%i-%i" % (site, inst)), "w") as d:
                    for x in list2:
                        d.write(repr(x[0]) + "\t" + repr(x[1]) + "\n")


if __name__ == "__main__":
    main()

