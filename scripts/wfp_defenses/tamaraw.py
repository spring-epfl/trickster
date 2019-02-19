# Based on: http://home.cse.ust.hk/~taow/wf/defenses/
# Author: Tao Wang and collaborators.


# Anoa consists of two components:
# 1. Send packets at some packet rate until data is done.
# 2. Pad to cover total transmission size.
# The main logic decides how to send the next packet.
# Resultant anonymity is measured in ambiguity sizes.
# Resultant overhead is in size and time.
# Maximizing anonymity while minimizing overhead is what we want.

import os
import sys
import math
import random
import numpy as np

from shutil import copyfile

from tqdm import tqdm

DATASIZE = 800


def fsign(num):
    if num > 0:
        return 0
    else:
        return 1


def rsign(num):
    if num == 0:
        return 1
    else:
        return abs(num) / num


def AnoaTime(parameters):
    direction = parameters[0]  # 0 out, 1 in
    method = parameters[1]
    if method == 0:
        if direction == 0:
            return 0.02
            # return 0.04
        if direction == 1:
            return 0.02
            # return 0.012


def AnoaPad(list1, list2, padL, method):
    lengths = [0, 0]
    times = [0, 0]
    for x in list1:
        if x[1] > 0:
            lengths[0] += 1
            times[0] = x[0]
        else:
            lengths[1] += 1
            times[1] = x[0]
        list2.append(x)
    for j in range(0, 2):
        curtime = times[j]
        topad = -int(
            math.log(random.uniform(0.00001, 1), 2) - 1
        )  # 1/2 1, 1/4 2, 1/8 3, ... #check this
        if method == 0:
            topad = (lengths[j] / padL + topad) * padL
        while lengths[j] < topad:
            curtime += AnoaTime([j, 0])
            if j == 0:
                list2.append([curtime, DATASIZE])
            else:
                list2.append([curtime, -DATASIZE])
            lengths[j] += 1
    return list2


def Anoa(list1, list2, parameters):  # inputpacket, outputpacket, parameters
    # Does NOT do padding, because ambiguity set analysis.
    # list1 WILL be modified! if necessary rewrite to tempify list1.
    starttime = list1[0][0]
    times = [starttime, starttime]  # lastpostime, lastnegtime
    curtime = starttime
    lengths = [0, 0]
    datasize = DATASIZE
    method = 0
    if method == 0:
        parameters[0] = (
            "Constant packet rate: "
            + str(AnoaTime([0, 0]))
            + ", "
            + str(AnoaTime([1, 0]))
            + ". "
        )
        parameters[0] += "Data size: " + str(datasize) + ". "
    if method == 1:
        parameters[0] = "Time-split varying bandwidth, split by 0.1 seconds. "
        parameters[0] += "Tolerance: 2x."
    listind = 0  # marks the next packet to send
    while listind < len(list1):
        # print(listind, len(list1), len(list2))
        # decide which packet to send
        if times[0] + AnoaTime([0, method, times[0] - starttime]) < times[1] + AnoaTime(
            [1, method, times[1] - starttime]
        ):
            cursign = 0
        else:
            cursign = 1
        times[cursign] += AnoaTime([cursign, method, times[cursign] - starttime])
        curtime = times[cursign]

        tosend = datasize
        while (
            list1[listind][0] <= curtime
            and fsign(list1[listind][1]) == cursign
            and tosend > 0
        ):
            if tosend >= abs(list1[listind][1]):
                tosend -= abs(list1[listind][1])
                listind += 1
            else:
                list1[listind][1] = (abs(list1[listind][1]) - tosend) * rsign(
                    list1[listind][1]
                )
                tosend = 0
            if listind >= len(list1):
                break
        if cursign == 0:
            list2.append([curtime, datasize])
        else:
            list2.append([curtime, -datasize])
        lengths[cursign] += 1
        # if list1[listind][0] == list1[listind+1][0]:
        #    print(listind, list1[listind], list2[-5:], len(list2[:listind]), len(list1[:listind]))
        # listind += 1
    return list2


if __name__ == "__main__":
    ##    parameters = [100] #padL
    ##    AnoaPad(list2, lengths, times, parameters)

    # for x in sys.argv[2:]:
    #    parameters.append(float(x))

    max_trace_len = int(sys.argv[1])
    sitenum = 100
    instnum = 90

    data_path = "data/wfp/batch__tracelen_%i/" % max_trace_len
    out_path = "out/wfp/tamaraw_002__tracelen_%i/" % max_trace_len

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tardist = [[], []]
    defpackets = []

    packets = []
    desc = ""
    anoad = []
    anoadpad = []

    unmod = []
    mod = []
    added = []
    for site in tqdm(range(0, sitenum)):
        for inst in range(0, instnum):
            packets = []
            if not os.path.exists(data_path + str(site) + "-" + str(inst)):
                continue
            with open(data_path + str(site) + "-" + str(inst), "r") as f:
                lines = f.readlines()
                starttime = float(lines[0].split("\t")[0])
                for x in lines:
                    x = x.split("\t")
                    packets.append([float(x[0]) - starttime, int(x[1])])
            if len(packets) > max_trace_len:
                continue
            list2 = []
            parameters = [""]
            # if site == 81 and inst ==37:
            #    print(len(packets))
            #    list22 = Anoa(packets, list2, parameters)
            #    print(len(list22))
            # else:
            #    continue
            list22 = Anoa(packets, list2, parameters)
            list22 = sorted(list22, key=lambda list22: list22[0])
            anoad.append(list2)

            list3 = []

            list33 = AnoaPad(list22, list3, 200, 0)

            list33 = sorted(list33, key=lambda list33: list33[0])
            # if site == 81 and inst ==37:
            #    print(packets[:10])
            #    print(list22[:10])
            #    print(list33[:10])
            #    print()
            #    print(len(packets))
            #    print(len(list22))
            #    print(len(list33))
            #    print(len([x for x in list33 if x[1]<0]))
            #    print(len([x for x in list33 if x[1]>0]))
            #    exit()

            fout = open(out_path + str(site) + "-" + str(inst), "w")
            # for x in list22:
            for x in list33:
                if x[1] <= -1:
                    direction = -1
                else:
                    direction = 1
                fout.write(str(x[0]) + "\t" + str(direction) + "\n")
            fout.close()
