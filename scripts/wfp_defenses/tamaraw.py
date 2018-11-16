#Anoa consists of two components:
#1. Send packets at some packet rate until data is done.
#2. Pad to cover total transmission size.
#The main logic decides how to send the next packet. 
#Resultant anonymity is measured in ambiguity sizes.
#Resultant overhead is in size and time.
#Maximizing anonymity while minimizing overhead is what we want. 
import math
import random

DATASIZE = 800

tardist = [[], []]
defpackets = []

def fsign(num):
    if num > 0:
        return 0
    else:
        return 1

def rsign(num):
    if num == 0:
        return 1
    else:
        return abs(num)/num

def AnoaTime(parameters):
    direction = parameters[0] #0 out, 1 in
    method = parameters[1]
    if (method == 0):
        if direction == 0:
            return 0.04
        if direction == 1:
            return 0.012
        

def AnoaPad(list1, list2, padL, method):
    lengths = [0, 0]
    times = [0, 0]
    for x in list1:
        if (x[1] > 0):
            lengths[0] += 1
            times[0] = x[0]
        else:
            lengths[1] += 1
            times[1] = x[0]
        list2.append(x)
    for j in range(0, 2):
        curtime = times[j]
        topad = -int(math.log(random.uniform(0.00001, 1), 2) - 1) #1/2 1, 1/4 2, 1/8 3, ... #check this
        if (method == 0):
            topad = (lengths[j]/padL + topad) * padL
        while (lengths[j] < topad):
            curtime += AnoaTime([j, 0])
            if j == 0:
                list2.append([curtime, DATASIZE])
            else:
                list2.append([curtime, -DATASIZE])
            lengths[j] += 1

def Anoa(list1, list2, parameters): #inputpacket, outputpacket, parameters
    #Does NOT do padding, because ambiguity set analysis. 
    #list1 WILL be modified! if necessary rewrite to tempify list1.
    starttime = list1[0][0]
    times = [starttime, starttime] #lastpostime, lastnegtime
    curtime = starttime
    lengths = [0, 0]
    datasize = DATASIZE
    method = 0
    if (method == 0):
        parameters[0] = "Constant packet rate: " + str(AnoaTime([0, 0])) + ", " + str(AnoaTime([1, 0])) + ". "
        parameters[0] += "Data size: " + str(datasize) + ". "
    if (method == 1):
        parameters[0] = "Time-split varying bandwidth, split by 0.1 seconds. "
        parameters[0] += "Tolerance: 2x."
    listind = 0 #marks the next packet to send
    while (listind < len(list1)):
        #decide which packet to send
        if times[0] + AnoaTime([0, method, times[0]-starttime]) < times[1] + AnoaTime([1, method, times[1]-starttime]):
            cursign = 0
        else:
            cursign = 1
        times[cursign] += AnoaTime([cursign, method, times[cursign]-starttime])
        curtime = times[cursign]
        
        tosend = datasize
        while (list1[listind][0] <= curtime and fsign(list1[listind][1]) == cursign and tosend > 0):
            if (tosend >= abs(list1[listind][1])):
                tosend -= abs(list1[listind][1])
                listind += 1
            else:
                list1[listind][1] = (abs(list1[listind][1]) - tosend) * rsign(list1[listind][1])
                tosend = 0
            if (listind >= len(list1)):
                break
        if cursign == 0:
            list2.append([curtime, datasize])
        else:
            list2.append([curtime, -datasize])
        lengths[cursign] += 1
        
##    parameters = [100] #padL
##    AnoaPad(list2, lengths, times, parameters)

import sys
import os
for x in sys.argv[2:]:
    parameters.append(float(x))

sitenum = 100
instnum = 90
fold = "../../data/batch/"
foldout = "../../data/batch-tamaraw/"

if not os.path.exists(foldout):
    os.makedirs(foldout)

max_trace_len = int(sys.argv[1]) 


packets = []
desc = ""
anoad = []
anoadpad = []
for site in range(0, sitenum):
    print site
    for inst in range(0, instnum):
        packets = []
        with open("../../data/batch/" + str(site) + "-" + str(inst), "r") as f:
            lines = f.readlines()
            starttime = float(lines[0].split("\t")[0])
            for x in lines:
                x = x.split("\t")
                packets.append([float(x[0]) - starttime, int(x[1])])
        if len(packets) > max_trace_len:
            continue
        list2 = []
        parameters = [""]
        
        Anoa(packets, list2, parameters)
        list2 = sorted(list2, key = lambda list2: list2[0])
        anoad.append(list2)

        list3 = []
        
        AnoaPad(list2, list3, 100, 0)

        fout = open(foldout + str(site) + "-" + str(inst), "w")
        for x in list3:
            fout.write(str(x[0]) + "\t" + str(x[1]) + "\n")
        fout.close()

