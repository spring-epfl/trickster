#Generates 1-X from 0-X.
import os
import sys
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

    if (method == 5): #HTTPOS2
        for x in list1:
            if (x[1] < 0):
                if (x[1] <= -1400):
                    list2.append([x[0], x[1]])
                else:
                    if (abs(x[1]) > 1):
                        r = random.randint(1, abs(x[1])-1)
                        list2.append([x[0], -r])
                        list2.append([x[0], x[1] + r])
                    else:
                        list2.append(x)
            if (x[1] > 0):
                list2.append([x[0], 1500])
        list2 = sorted(list2, key = lambda list2: list2[0])

                

import sys
import os

if not os.path.exists("../../data/batchusenix-hdef"):
    os.makedirs("../../data/batchusenix-hdef")

if not os.path.exists("../../data/batch"):
    print "batchtcp folder needs to exist"
    sys.exit(0)

parameters = [0, 0]

packets = []

method = 5
for j in range(0, 100):
    print j
    for i in range(0, 90):
        packets = []
        with open("../../data/batch/" + str(j) + "-" + str(i), "r") as f:
            for x in f.readlines():
                x = x.split("\t")
                packets.append([float(x[0]), int(x[1])])
        with open("../../data/batchusenix-hdef/" + str(j) + "-" + str(i), "w") as f:
            list2 = []
            parameters[0] = j
            parameters[1] = i
            defend(packets, list2, method, parameters)
            list2 = sorted(list2, key = lambda list2: list2[0])
            for x in list2:
                f.write(repr(x[0]) + "\t" + repr(x[1]) + "\n")
            
