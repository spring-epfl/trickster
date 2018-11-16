#Generates 1-X from 0-X.
import math
import random

tardist = []

defpackets = []

def fsign(num):
    if num > 0:
        return 0
    else:
        return 1

def defend(list1, list2, method, parameter):
    #method is integer from 1 to n
    #parameters is [class, ....]
    #list2 is empty and where stuff goes
    if (method == 1): #giant padding
        for x in list1:
            list2.append([x[0], x[1]/abs(x[1]) * 1500])
    if (method == 2): #exponential padding
        for x in list1:
            padx = int(math.pow(2, math.ceil(math.log(abs(x[1]), 2))))
            padx = min(padx, 1500)
            list2.append([x[0], x[1]/abs(x[1]) * padx])

    if (method == 3): #traffic morphing
        tar = 0
        for x in list1:
            remainder = abs(x[1])
            if x[1] > 0:
                sign = 0
            else:
                sign = 1
            while (remainder > 0):
                s = random.sample(tardist[tar][sign], 1)
                list2.append([x[0], abs(x[1])/x[1] * s[0]])
                remainder -= s[0]

import sys
import os
parameters = [0, 0]

packets = []

#method = int(sys.argv[1])

method = 3

if not os.path.exists("batchusenix-tdef"):
    os.makedirs("batchusenix-tdef")

if not os.path.exists("batchtcp"):
    print "batchtcp folder needs to exist"
    sys.exit(0)

#Preprocessing
for i in range(0, 100):
    with open("batchtcp/" + str(i) + "-0", "r") as f:
        tardist.append([[], []])
        for x in f.readlines():
            x = x.split("\t")[1]
            if (int(x) > 0):
                tardist[-1][0].append(abs(int(x)))
            elif (int(x) < 0):
                tardist[-1][1].append(abs(int(x)))

for j in range(0, 100):
    print j
    for i in range(0, 90):
        packets = []
        with open("batchtcp/" + str(j) + "-" + str(i), "r") as f:
            for x in f.readlines():
                x = x.split("\t")
                packets.append([float(x[0]), int(x[1])])
        with open("batchusenix-tdef/" + str(j) + "-" + str(i), "w") as f:
            list2 = []
            parameters[0] = j
            parameters[1] = i
            defend(packets, list2, method, parameters)
            list2 = sorted(list2, key = lambda list2: list2[0])
            for x in list2:
                f.write(repr(x[0]) + "\t" + repr(x[1]) + "\n")
            
