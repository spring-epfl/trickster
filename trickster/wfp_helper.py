import sys

# Extract CUMUL features
def extract(sinste):
    #sinste: list of packet sizes

    #first 4 features

    insize = 0
    outsize = 0
    inpacket = 0
    outpacket = 0

    for i in range(0, len(sinste)):
        if sinste[i] > 0:
            outsize += sinste[i]
            outpacket += 1
        else:
            insize += abs(sinste[i])
            inpacket += 1
    features = [insize, outsize, inpacket, outpacket]

    #100 interpolants
    
    n = 100 #number of linear interpolants

    x = 0 #sum of absolute packet sizes
    y = 0 #sum of packet sizes
    graph = []
    
    for si in range(0, len(sinste)):
        x += abs(sinste[si])
        y += sinste[si]
        graph.append([x, y])


    #derive interpolants
    max_x = graph[-1][0] 
    gap = float(max_x)/(n+1)
    graph_ptr = 0

    for i in range(0, n):
        sample_x = gap * (i+1)
        while (graph[graph_ptr][0] < sample_x):
            graph_ptr += 1
            if (graph_ptr >= len(graph) - 1):
                graph_ptr = len(graph) - 1
                #wouldn't be necessary if floats were floats
                break
        next_y = graph[graph_ptr][1]
        next_x = graph[graph_ptr][0]
        last_y = graph[graph_ptr-1][1]
        last_x = graph[graph_ptr-1][0]

        if (next_x - last_x != 0):
            slope = (next_y - last_y)/float(next_x - last_x)
        else:
            slope = 1000
        sample_y = slope * (sample_x - last_x) + last_y

        features.append(sample_y)

    return features

def load_cell(fname, time=0, ext=".cell"):
    #time = 0 means don't load packet times (saves time and memory)
    data = []
    starttime = -1
    try:
        f = open(fname, "r")
        lines = f.readlines()
        f.close()

        if ext == ".htor":
            #htor actually loads into a cell format
            for li in lines:
                psize = 0
                if "INCOMING" in li:
                    psize = -1
                if "OUTGOING" in li:
                    psize = 1
                if psize != 0:
                    if time == 0:
                        data.append(psize)
                    if time == 1:
                        time = float(li.split(" ")[0])
                        if (starttime == -1):
                            starttime = time
                        data.append([time - starttime, psize])

        if ext == ".cell":
            for li in lines:
                li = li.split("\t")
                p = int(li[1])
                if time == 0:
                    data.append(p)
                if time == 1:
                    t = float(li[0])
                    if (starttime == -1):
                        starttime = t
                    data.append([t-starttime, p])
        if ext == ".burst":
            #data is like: 1,1,1,-1,-1\n1,1,1,1,-1,-1,-1
            for li in lines:
                burst = [0, 0]
                li = li.split(",")
                data.append([li.count("1"), li.count("-1")])
                for l in li:
                    if l == "1":
                        burst[0] += 1
                    if l == "-1":
                        burst[1] += 1
                data.append(burst)

        if ext == ".pairs":
            #data is like: [[3, 12], [1, 24]]
            #not truly implemented
            data = list(lines[0])            
    except:
        print("Could not load", fname)
        sys.exit(-1)
    return data
