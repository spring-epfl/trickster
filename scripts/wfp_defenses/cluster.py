import random
import math

fold = "/home/tao/temp/tor/code/distchecker/batchusenix/"

SITE_NUM = 40
INST_NUM = 20
TESTSITE_NUM = 30
TESTINST_NUM = 30
CLUSTER_NUM = 20

#For method 3 only
#SUPCLUSTER_NUM = int(math.ceil(math.sqrt(CLUSTER_NUM)))
SUPCLUSTER_NUM = 2
STOPPOINTS_NUM = int(math.ceil(float(CLUSTER_NUM)/SUPCLUSTER_NUM))

sizes = []
times = []
testsizes = []

method = 1

clusters = [-1]*SITE_NUM*INST_NUM #-1 is no cluster
stoppoints = []

superseqlens = []

sseq = []
tseq = []

def dist(sinste1, sinste2):
        
    #Returns distance between sinste and sinste2
    #Does true supersequence finding
    lastline = [0] * len(sizes[sinste1])
    thisline = [0] * len(sizes[sinste1])

    for i in range(0, len(sizes[sinste2])):
        if sizes[sinste1][0] in sizes[sinste2][:i]:
            thisline[0] = 1
        else:
            thisline[0] = 0
        for j in range(1, len(sizes[sinste1])):
            if sizes[sinste1][j] == sizes[sinste2][i]:
                thisline[j] = 1 + lastline[j-1]
            else:
                thisline[j] = max(lastline[j], thisline[j-1])
        for j in range(0, len(sizes[sinste1])):
            lastline[j] = thisline[j]
    return len(sizes[sinste1]) + len(sizes[sinste2]) - 2 * lastline[j]

def samelendist(sinste1, sinste2):
        
    #Returns distance between sinste and sinste2
    #Does true supersequence finding

    minsize = min(len(sizes[sinste1]), len(sizes[sinste2]))
    
    lastline = [0] * minsize
    thisline = [0] * minsize

    for i in range(0, minsize):
        if sizes[sinste1][0] in sizes[sinste2][:i]:
            thisline[0] = 1
        else:
            thisline[0] = 0
        for j in range(1, minsize):
            if sizes[sinste1][j] == sizes[sinste2][i]:
                thisline[j] = 1 + lastline[j-1]
            else:
                thisline[j] = max(lastline[j], thisline[j-1])
        for j in range(0, minsize):
            lastline[j] = thisline[j]
    return 2 * minsize - 2 * lastline[j]


def mindist(site, inst):
    #Returns instance number with minimum distance
    sinste = site * INST_NUM + inst
    mindist = dist(sinste, (sinste + 1) % SITE_NUM*INST_NUM)
    minindex = -1
    for i in range(0, SITE_NUM):
        for j in range(0, INST_NUM):
            sinste2 = i * INST_NUM + j
            if (i != site):
                d = dist(sinste, sinste2)
                if (d <= mindist):
                    minindex = sinste2
                    mindist = d
    return minindex

def superseq(clus):
    pointers = []

    done = 0
    totaldone = 0

    time = 0
    lasttime = [0, 0]
    gap = [0.03, 0.01]
    #use all sets that have been loaded:
    for i in range(0, len(clus)):
        pointers.append(0)
        totaldone += 1

    while (len(sseq) > 0):
        sseq.pop(0)
    while (len(tseq) > 0):
        tseq.pop(0)
        
    while (done != totaldone):
        vote = 0
        #make your votes
        for i in range(0, len(clus)):
            if (pointers[i] != -1):
                if (sizes[clus[i]][pointers[i]] > 0):
                    vote += 3.5
                else:
                    vote += -1
        
        #make a decision

        if (vote > 0):
            thispacket = 1
            sign = 0
        else:
            thispacket = -1
            sign = 1
        sseq.append(thispacket)

        #progress pointers

        passtimes = []
        
        for i in range(0, len(clus)):
            if (pointers[i] != -1):
                if (sizes[clus[i]][pointers[i]] * thispacket > 0):
                    passtimes.append(times[clus[i]][pointers[i]])

        passtimes = sorted(passtimes)
        time = max(passtimes[int(len(passtimes) * 0.8)], time)
        
        for i in range(0, len(clus)):
            if (pointers[i] != -1):
                if (sizes[clus[i]][pointers[i]] * thispacket > 0 and times[clus[i]][pointers[i]] <= time):
                    pointers[i] += 1
                    
            if (pointers[i] == len(sizes[clus[i]])):
                pointers[i] = -1
                done += 1

        time = max(lasttime[sign] + gap[sign], time)
        tseq.append(time)
        lasttime[sign] = time

##    fout = open("st.txt", "w")
##    fout.write("1, " + str(len(sseq)) + ",\n")
##    for i in range(0, len(sseq)):
##        fout.write(str(sseq[i] * 1500 * -1) + ",")
##    fout.write("\n")
##    for i in range(0, len(sseq)):
##        fout.write(str(int(tseq[i] * 1000000)) + ",")
##    fout.write("\n")
    
##    fout.close()

    return sseq


def transmit(smallseq, superseq):
    #Returns number of packets required in total to send the smallseq from superseq
    pt = 0
    for x in smallseq:
        if (pt == len(superseq)):
            superseq.append(1)
            superseq.append(-1)
            superseq.append(-1)
            superseq.append(-1)
            superseq.append(-1)
        while (superseq[pt] * x < 0):
            pt += 1
            if (pt == len(superseq)):
                superseq.append(1)
                superseq.append(-1)
                superseq.append(-1)
                superseq.append(-1)
                superseq.append(-1)
        pt += 1
        
    return pt

def ucluster(method):


    for i in range(0, len(clusters)):
        clusters[i] = -1

    roots = [] #clusternum should be 0, 1, 2....

    #METHOD 1: No info
    if (method == 1 or method == 2):
        for i in range(0, SITE_NUM*INST_NUM):
            clusters[i] = 0
                
    #METHOD 3: Class info
    if (method == 3):
        #First level clustering

        while (len(roots) < SUPCLUSTER_NUM):
            site = random.randint(0, SITE_NUM - 1)
            if (not (site * INST_NUM in roots)):
                for inst in range(0, INST_NUM):
                    clusters[site * INST_NUM + inst] = len(roots)
                roots.append(site * INST_NUM)
        
        rootlens = []
        for k in range(0, len(roots)):
            thisrootlens = []
            for i in range(0, SITE_NUM):
                sinste = i * INST_NUM + random.randint(0, INST_NUM-1)
                d = samelendist(roots[k], sinste)
                thisrootlens.append([d, i])
##                print d, i
            thisrootlens = sorted(thisrootlens, key = lambda thisrootlens:thisrootlens[0])
            rootlens.append([])
            for x in thisrootlens:
                rootlens[-1].append(x[1])
##            print rootlens

        tocluster = SITE_NUM - len(roots)
        thisroot = 0
        while tocluster > 0:
            site = rootlens[thisroot].pop(0)
            while (clusters[site * INST_NUM] != -1):
                site = rootlens[thisroot].pop(0)
            for inst in range(0, INST_NUM):
                clusters[site * INST_NUM + inst] = thisroot
            thisroot = (thisroot + 1) % len(roots)
            tocluster -= 1

        while(len(stoppoints) > 0):
            stoppoints.pop(0)
        #Second level clustering
        for c in range(0, SUPCLUSTER_NUM):
            clus = []
            for i in range(0, SITE_NUM*INST_NUM):
                if clusters[i] == c:
                    clus.append(i)
            sseq = superseq(clus)
            clusterlens = []
            for i in clus:
                clusterlens.append([transmit(sizes[i], sseq), i/INST_NUM])
            clusterlens = sorted(clusterlens, key = lambda clusterlens: clusterlens[0])
            stoppoints.append([])

            sitecount = []
            for i in range(0, SITE_NUM):
                if clusters[i* INST_NUM] == c:
                    sitecount.append(0)
                else:
                    sitecount.append(-1)
            
            CLUSTER_SIZE = int(math.ceil(float(SITE_NUM/SUPCLUSTER_NUM)/STOPPOINTS_NUM))
##            print "CLUSTER_SIZE", CLUSTER_SIZE
            for x in clusterlens:
                sitecount[x[1]] += 1
                finished = 1
                sm = sum(sitecount)
                for num in sitecount:
                    if (num > sm/CLUSTER_SIZE and num != -1):
                        finished = 0

                if (finished == 1):
                    stoppoints[-1].append(x[0])
                    for i in range(0, SITE_NUM):
                        if clusters[i* INST_NUM] == c:
                            sitecount.append(0)
                        else:
                            sitecount.append(-1)

##            for i in range(0, len(stoppoints[-1])):
##                print stoppoints[-1][i]
        
    #METHOD 4: Full info
    if (method == 4):
        inst = 0
        sizelist = []
        for site in range(0, SITE_NUM):
            sizelist.append([site * INST_NUM + inst, sizes[site * INST_NUM + inst]])
        sizelist = sorted(sizelist, key = lambda sizelist: sizelist[1])
        for c in range(0, CLUSTER_NUM):
            roots.append(sizelist[SITE_NUM/CLUSTER_NUM * c][0])
        tocluster = SITE_NUM
        while (tocluster > 0):
            for c in range(0, CLUSTER_NUM):
                mind = -1
                minsite = 0
                for site in range(0, SITE_NUM):
                    sinste = site * INST_NUM + inst
                    if (clusters[sinste] == -1):
                        d = dist(site * INST_NUM + inst, roots[c])
                        if (d < mind or mind == -1):
                            mind = d
                            minsite = site
                clusters[minsite * INST_NUM + inst] = c
                tocluster -= 1
        
##        for site in range(0, SITE_NUM):
##            mind = dist(site * INST_NUM + inst, roots[0])
##            minr = 0
##            for r in range(0, len(roots)):
##                d = dist(site * INST_NUM + inst, roots[r])
##                if (d < mind):
##                    mind = d
##                    minr = r
##            clusters[site * INST_NUM + inst] = minr
        

def nucluster(method):
    for i in range(0, len(clusters)):
        clusters[i] = -1

    roots = [] #clusternum should be 0, 1, 2....
    
    #METHOD 3: Class info
    if (method == 3):
        #First level clustering

        while (len(roots) < SUPCLUSTER_NUM):
            sinste = random.randint(0, SITE_NUM - 1) * INST_NUM
            if (not (sinste in roots)):
                roots.append(sinste)

        for i in range(0, SITE_NUM):
            mindist = -1
            mink = 0
            for k in range(0, len(roots)):
                if (clusters[i * INST_NUM] == -1):
                    sinste = i * INST_NUM + random.randint(0, INST_NUM-1)
                    d = samelendist(roots[k], sinste)
                    if (d < mindist or mindist == -1):
                        mindist = d
                        mink = k
            #print i * INST_NUM, mink
            for inst in range(0, INST_NUM):
                clusters[i * INST_NUM + inst] = mink

        while(len(stoppoints) > 0):
            stoppoints.pop(0)
        #Second level clustering

        for c in range(0, SUPCLUSTER_NUM):
            stopids = []
            clus = []
            for i in range(0, SITE_NUM*INST_NUM):
                if clusters[i] == c:
                    clus.append(i)
            sseq = superseq(clus)
            clusterlens = []
            for i in clus:
                clusterlens.append(transmit(sizes[i], sseq))
            clusterlens = sorted(clusterlens)
            gap = len(clusterlens)/STOPPOINTS_NUM

            #print gap, len(clusterlens), STOPPOINTS_NUM
            for i in range(0, STOPPOINTS_NUM):
                stopids.append((i+1) * gap - 1)

            #non-uniform cluster; allows adjustment
##            change = 1
##            while (change == 1):
##                change = 0
##                for i in range(0, STOPPOINTS_NUM-1):
##                    #cost of -1 to stop point
##                    cost = 0
####                    print tr, i, stopids[0], stopids[1], len(clusterlens)
##                    if i == 0:
##                        for j in range(0, len(clusterlens)):
##                            if clusterlens[j] < clusterlens[stopids[0]]:
##                                cost -= clusterlens[stopids[0]] - clusterlens[stopids[0] - 1]
##
##                    else:
##                        for j in range(0, len(clusterlens)):
##                            if clusterlens[j] > clusterlens[stopids[i-1]] and clusterlens[j] < clusterlens[stopids[i]]:
##                                cost -= clusterlens[stopids[i]] - clusterlens[stopids[i]-1]
##                    cost += clusterlens[stopids[i+1]] - clusterlens[stopids[i]]
##                    if (cost < 0):
##                        stopids[i] -= 1
##                        change = 1
##                        #print tr, i, stopids[i]
##                    #cost of +1 to stop point
##                for i in range(0, STOPPOINTS_NUM-1):
##                    cost = 0
##                    if i == 0:
##                        for j in range(0, len(clusterlens)):
##                            if clusterlens[j] <= clusterlens[stopids[0]]:
##                                cost += clusterlens[stopids[0] + 1] - clusterlens[stopids[0]]
##                    else:
##                        for j in range(0, len(clusterlens)):
##                            if clusterlens[j] > clusterlens[stopids[i-1]] and clusterlens[j] < clusterlens[stopids[i] + 1]:
##                                cost += clusterlens[stopids[i]+1] - clusterlens[stopids[i]]
##                    cost -= clusterlens[stopids[i+1]] - clusterlens[stopids[i] + 1]
##                    if (cost < 0):
##                        stopids[i] += 1
##                        change = 1
##                        #print tr, i, stopids[i]
                        
            stoppoints.append([])
            for i in range(0, STOPPOINTS_NUM):
                stoppoints[-1].append(clusterlens[stopids[i]])

                
    if (method == 4):
        inst = 0
        sizelist = []
        for site in range(0, SITE_NUM):
            sizelist.append([site * INST_NUM + inst, sizes[site * INST_NUM + inst]])
        sizelist = sorted(sizelist, key = lambda sizelist: sizelist[1])
        for c in range(0, CLUSTER_NUM):
            roots.append(sizelist[SITE_NUM/CLUSTER_NUM * c][0])

        inst = 0
        for i in range(0, SITE_NUM):
            sinste = i * INST_NUM + inst

            if (clusters[sinste] == -1):
                mind = -1
                minc = 0
                for c in range(0, CLUSTER_NUM):
                    d = dist(sinste, roots[c])
                    if (d < mind or mind == -1):
                        mind = d
                        minc = c
                clusters[sinste] = minc
                    

    
    return 0

def evaluate(method):

    
    #Returns uacc, nuacc, e
    cost = 0
    totallen = 0

    if (method == 1):
        for c in range(0, 1):
            clus = []
            for site in range(0, SITE_NUM):
                for inst in range(0, INST_NUM):
                    if (clusters[site*INST_NUM+inst] == c):
                        clus.append(site*INST_NUM+inst)
            s = superseq(clus)
            
        for site in range(0, SITE_NUM):
            for inst in range(0, INST_NUM):
                sinste = site*INST_NUM + inst
                cost += max(transmit(testsizes[sinste], s), len(s)) - len(testsizes[sinste])
                totallen += len(testsizes[sinste])

    if (method == 2):
        for c in range(0, 1):
            clus = []
            for sinste in range(0, SITE_NUM*INST_NUM):
                clus.append(sinste)
            s = superseq(clus)

        s = []
        for i in range(0, 5000):
            s.append(1)
            s.append(-1)
            s.append(-1)
            s.append(-1)
            s.append(-1)

        tlens = []
        for sinste in range(0, SITE_NUM*INST_NUM):
            tlen = transmit(sizes[sinste], s)
            tlens.append([tlen, sinste / INST_NUM])
        tlens = sorted(tlens, key = lambda tlens: -tlens[0])
##        print tlens

        sitecount = []
        for i in range(0, SITE_NUM):
            sitecount.append(0)
            
        while (len(superseqlens) > 0):
            superseqlens.pop(0)
        CLUSTER_SIZE = int(math.ceil(float(SITE_NUM)/CLUSTER_NUM))
        for x in tlens:
            sitecount[x[1]] += 1
            finished = 1
            sm = sum(sitecount)
            for num in sitecount:
                if (num > sm/CLUSTER_SIZE):
                    finished = 0

            if (finished == 1):
                superseqlens.append(x[0])
                for i in range(0, SITE_NUM):
                    sitecount[i] = 0
##        for x in superseqlens:
##            print x
            
##        fout = open("st.txt", "a+")
##        for i in range(0, len(s)):
##            if i in superseqlens:
####                print i
##                fout.write("1,")
##            else:
##                fout.write("0,")
##        fout.write("\n")

        costs = []
                    
        sdiff = tlens[0][0] / 5
        
        for site in range(0, TESTSITE_NUM):
            for inst in range(0, TESTINST_NUM):
                sinste = site*TESTINST_NUM + inst
                tlen = transmit(testsizes[sinste], s)
                #cost += tlen - len(testsizes[sinste])
                ptlen = (tlen/sdiff + 1) * sdiff
                for l in superseqlens:
                    if l >= tlen and l < ptlen:
                        ptlen = l
                #print ptlen, tlen
                cost += ptlen - len(testsizes[sinste])
                totallen += len(testsizes[sinste])
                costs.append((ptlen - len(testsizes[sinste])) / float(len(testsizes[sinste])))

##        for site in range(0, 100):
##            for inst in range(0, 90):
##                sinste = site*INST_NUM + inst
##                tlen = transmit(sizes[sinste], s)
##                #cost += tlen - len(testsizes[sinste])
##                ptlen = (tlen/sdiff + 1) * sdiff
##                for l in superseqlens:
##                    if l >= tlen and l < ptlen:
##                        ptlen = l
##                #print ptlen, tlen
##                fout = open("../batchusenix/" + str(site) + "-" + str(inst) + "d", "w")
##                for i in range(0, min(ptlen, len(sseq))):
##                    fout.write(str(tseq[i]) + "\t" + str(sseq[i]) + "\n")
##                fout.close()
##                cost += ptlen - len(sizes[sinste])
##                totallen += len(sizes[sinste])
##                costs.append((ptlen - len(sizes[sinste])) / float(len(sizes[sinste])))

        
    if (method == 3):
        costs = []
        for c in range(0, SUPCLUSTER_NUM):
            clus = []
            for sinste in range(0, SITE_NUM*INST_NUM):
                if (clusters[sinste] == c):
                    clus.append(sinste)
            s = superseq(clus)
            
            sdiff = len(s) / 5
            for inst in range(0, INST_NUM):
                for site in range(0, SITE_NUM):
                    if (clusters[site * INST_NUM] == c):
                        sinste = site * INST_NUM + inst
                        tlen = transmit(sizes[sinste], s)
                        ptlen = (tlen/sdiff + 1) * sdiff
                        for l in stoppoints[c]:
                            if l >= tlen and l < ptlen:
                                ptlen = l
                        #print ptlen, tlen
                        cost += ptlen - len(sizes[sinste])
                        totallen += len(sizes[sinste])
                costs.append(float(cost)/totallen)
                cost = 0
                totallen = 0
        import numpy
        print numpy.mean(costs), numpy.std(costs)
                

    if (method == 4):
        for c in range(0, CLUSTER_NUM):
            clus = []
            for sinste in range(0, SITE_NUM*INST_NUM):
                if (clusters[sinste] == c):
                    clus.append(sinste)
            s = superseq(clus)
            for sinste in clus:
                cost += transmit(sizes[sinste], s) - len(sizes[sinste])
                totallen += len(sizes[sinste])

##    costs = sorted(costs)
##    print costs[len(costs)/4]
##    print costs[len(costs)/2]
##    print costs[3*len(costs)/4]
    return cost/float(totallen)


print "Loading site"
for site in range(0, SITE_NUM):
    for instance in range(0, INST_NUM):
        fname = str(site) + "-" + str(instance)
        #Set up times, sizes
        f = open("../batchusenix/" + fname, "r")
        sizes.append([])
        times.append([])
        for x in f:
            x = x.split("\t")
            sizes[-1].append(int(x[1]))
            times[-1].append(float(x[0]))
        starttime = times[-1][0]
        for i in range(0, len(times[-1])):
            times[-1][i] -= starttime
        f.close()

print "Loading test"
for site in range(0, TESTSITE_NUM):   
    for instance in range(0, TESTINST_NUM):
        fname = str(site+SITE_NUM) + "-" + str(instance)
        #Set up times, sizes
        f = open("../batchusenix/" + fname, "r")
        testsizes.append([])
        for x in f:
            x = x.split("\t")
            testsizes[-1].append(int(x[1]))
        f.close()

##CLUSTER_NUM = 20
##for i in range(1, 11):
##    SUPCLUSTER_NUM = i
##    STOPPOINTS_NUM = int(math.ceil(float(CLUSTER_NUM)/SUPCLUSTER_NUM))
##    ucluster(3)
##    print SUPCLUSTER_NUM, STOPPOINTS_NUM, evaluate(3)

##for i in range(1, 21):        
##    CLUSTER_NUM = i
##    method = 2
##    ucluster(method)
##    print i, method, str(evaluate(method))

for tr in range(0, 20):
    for i in range(5, 6):
        CLUSTER_NUM = 20
        SUPCLUSTER_NUM = 2
        STOPPOINTS_NUM = i
        method = 3
        ucluster(method)
        print tr, i, method, str(evaluate(method))

##SUPCLUSTERLIST = [1, 2, 3, 4, 5, 7, 10]
##STOPPOINTSLIST = [20, 10, 8, 5, 4, 3, 2]
##for i in range(6, 7):
##    CLUSTER_NUM = 20
##    method = 3
##    SUPCLUSTER_NUM = SUPCLUSTERLIST[i]
##    STOPPOINTS_NUM = STOPPOINTSLIST[i]
##    ucluster(method)
##    print SUPCLUSTER_NUM, STOPPOINTS_NUM, method, str(evaluate(method))

##for i in range(1, 21):
##    CLUSTER_NUM = i
##    method = 4
##    nucluster(method)
##    print i, method, str(evaluate(method))
