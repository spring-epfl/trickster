import os
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


def extract(packet_sizes):
    """Extract CUMUL features.

    :param packet_sizes: (Signed) list of packet sizes
    """

    # First 4 features.
    insize = 0
    outsize = 0
    inpacket = 0
    outpacket = 0

    for i in range(0, len(packet_sizes)):
        if packet_sizes[i] > 0:
            outsize += packet_sizes[i]
            outpacket += 1
        else:
            insize += abs(packet_sizes[i])
            inpacket += 1
    features = [insize, outsize, inpacket, outpacket]

    # Reserve space for 100 interpolation features.
    n = 100  # Number of linear interpolants.
    x = 0  # Sum of absolute packet sizes.
    y = 0  # Sum of packet sizes.
    graph = []
    for packet_size in packet_sizes:
        x += abs(packet_size)
        y += packet_size
        graph.append([x, y])

    # Derive interpolants.
    max_x = graph[-1][0]
    gap = float(max_x) / (n + 1)
    graph_ptr = 0
    for i in range(0, n):
        sample_x = gap * (i + 1)
        while graph[graph_ptr][0] < sample_x:
            graph_ptr += 1
            if graph_ptr >= len(graph) - 1:
                graph_ptr = len(graph) - 1
                # Wouldn't be necessary if floats were floats.
                break
        next_y = graph[graph_ptr][1]
        next_x = graph[graph_ptr][0]
        last_y = graph[graph_ptr - 1][1]
        last_x = graph[graph_ptr - 1][0]

        if next_x - last_x != 0:
            slope = (next_y - last_y) / float(next_x - last_x)
        else:
            slope = 1000
        sample_y = slope * (sample_x - last_x) + last_y

        features.append(sample_y)

    return features


def load_cell_data(filename, time=0, ext=".cell", max_len=None):
    """Load cell data from file.

    :param time: If zero, don't load packet times (saves time and memoty).
    """
    data = []
    starttime = -1
    try:
        f = open(filename, "r")
        lines = f.readlines()
        f.close()

        if ext == ".htor":
            # htor actually loads into a cell format.
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
                        if starttime == -1:
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
                    if starttime == -1:
                        starttime = t
                    data.append([t - starttime, p])
        if ext == ".burst":
            # Data is like: 1,1,1,-1,-1\n1,1,1,1,-1,-1,-1
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
            # Data is like: [[3, 12], [1, 24]]
            # Not truly implemented...
            data = list(lines[0])

    except:
        raise Exception("Could not load cell data in %s" % filename)

    if max_len is None:
        return data
    else:
        return data[:max_len]


def one_hot_to_indices(data):
    indices = []
    for el in data:
        indices.append(list(el).index(1))
    return indices


def pad_and_onehot(data):
    max_trace_len = len(max([x for x in data], key=len)) + 200
    data = [np.pad(x, (0, max_trace_len - len(x)), "constant") for x in data]
    data = onehot(data)
    return max_trace_len, np.array(data)


def onehot(data):
    data_ = []
    for d in data:
        b = np.zeros((len(d), 3))
        b[np.arange(len(d)), d] = 1
        data_.append(b.flatten())
    return data_


def reverse_onehot(arr, trace_len):
    f = np.argmax(np.reshape(arr, (trace_len, 3)), axis=1)
    f[f == 2] = -1
    return f


def load_data(path, *args, **kwargs):
    """Load traces from a folder."""
    labels = []
    data = []
    for filename in tqdm(os.listdir(path)):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            cell_list = load_cell_data(file_path, *args, **kwargs)
            feature_list = extract(cell_list)
            if "-" in str(filename):
                labels.append(1)
                data.append((cell_list, feature_list))
            else:
                labels.append(0)
                data.append((cell_list, feature_list))
    labels = np.array(labels)
    data = np.array(data)
    return data, labels
