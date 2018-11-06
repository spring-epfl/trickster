"""
Website fingerprinting utilities: feature extraction and 'knndata' dataset loading.
"""

import os
import random

import numpy as np
from tqdm import tqdm


def extract(trace, interpolated_features=True):
    """Extract CUMUL features from a single trace.

    :param trace: (Signed) list of packet sizes.
    :param interpolated_features: Whether to include interpolated features.

    >>> trace = [1, 1, -1, -1, -1]
    >>> cumul_vector = extract(trace)
    >>> len(cumul_vector)
    104
    >>> cumul_vector[:4]
    array([3., 2., 3., 2.])

    >>> extract(trace, interpolated_features=False)
    array([3, 2, 3, 2])
    """

    # First 4 features.
    insize = 0
    outsize = 0
    inpacket = 0
    outpacket = 0

    for i in range(0, len(trace)):
        if trace[i] > 0:
            outsize += trace[i]
            outpacket += 1
        else:
            insize += abs(trace[i])
            inpacket += 1
    features = [insize, outsize, inpacket, outpacket]

    # Reserve space for 100 interpolation features.
    n = 100  # Number of linear interpolants.
    x = 0  # Sum of absolute packet sizes.
    y = 0  # Sum of packet sizes.
    graph = []
    for packet_size in trace:
        x += abs(packet_size)
        y += packet_size
        graph.append([x, y])

    # Derive interpolants.
    if interpolated_features:
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

    return np.array(features)


def load_cell_data(filename, time=0, ext=".cell", max_len=None, filter_by_len=True):
    """Load cell data from file.

    :param time: If zero, don't load packet times (saves time and memoty).
    :param ext: Data format/extension. One of ``[".htor", ".cell", ".burst"]``
    :param max_len: Max trace length.
    :param filter_by_len: If True, traces over the max_len will be dropped.
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

    # No filtering.
    elif not filter_by_len:
        return data[:max_len]

    # Filtering is on: if the trace is longer than max_len, return None.
    else:
        if len(data) <= max_len:
            return data
        else:
            return None


def load_data(
    path, shuffle=False, max_traces=None, max_trace_len=None, filter_by_len=True
):
    """Load traces from a folder.

    :param shuffle: Whether to shuffle the traces.
    :param max_traces: Max number of traces to load.

    See :py:func:`load_cell_data` for information about the other arguments.
    """
    labels = []
    data = []

    filenames = os.listdir(path)
    if shuffle:
        random.shuffle(filenames)
    else:
        filenames = sorted(filenames)

    num_traces = 0
    prog_bar = tqdm(total=len(filenames) if max_traces is None else max_traces)
    for filename in tqdm(filenames):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            trace = load_cell_data(
                file_path, max_len=max_trace_len, filter_by_len=filter_by_len
            )

            # Trace might be None if it was longer than allowed.
            if trace is not None:
                label = 1 if "-" in str(filename) else 0
                data.append(trace)
                labels.append(label)

                num_traces += 1
                prog_bar.update(1)
                if max_traces is not None and num_traces >= max_traces:
                    break

    labels = np.array(labels)
    data = np.array(data)
    return data, labels


def pad_and_onehot(data, pad_len=None, extra_padding=200):
    """Pad and one-hot encode traces.

    See :py:func:`onehot`.

    :param data: A list or array of traces.
    :param pad_len: Number of packets to pad to. If ``None``, pad to
            the max trace length.
    :param extra_padding: Extra padding.

    >>> pad_and_onehot([[1, 1, -1]], pad_len=4, extra_padding=0)
    (4, array([[0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]]))

            ^1st     ^2nd     ^3rd     ^pad
    """
    if pad_len is None:
        pad_len = max(len(x) for x in data) + extra_padding
    data = [
        onehot(np.pad(trace, (0, pad_len - len(trace)), mode="constant"))
        for trace in data
    ]
    return pad_len, np.array(data)


def onehot(trace):
    """One-hot encode a single trace.

    Each packet in the trace corresponds to three categories: none, outgoing, or ingoing.
    This encoding procedure one-hot encodes each position. This results in the binary
    vector of length ``3 * len(trace)`` for each trace.

    :param trace: Trace

    >>> onehot([1, -1])
    array([0, 1, 0, 0, 0, 1])
    """
    encoded_trace = np.zeros((len(trace), 3), dtype=int)
    encoded_trace[np.arange(len(trace)), trace] = 1
    return encoded_trace.flatten()


def reverse_onehot(encoded_trace, trace_len=None):
    """Reverse a single one-hot encoded trace.

    See :py:func:`onehot`.

    :param encoded_trace: One-hot encoded trace
    :param trace_len: Length of the original trace

    >>> encoded_trace = np.array([0, 1, 0, 0, 0, 1])
    >>> reverse_onehot(encoded_trace)
    array([ 1, -1])
    """
    if trace_len is None:
        trace_len = len(encoded_trace) // 3
    raw_trace = np.argmax(np.reshape(encoded_trace, (trace_len, 3)), axis=1)
    raw_trace[raw_trace == 2] = -1
    return raw_trace


def insert_dummy_packets(trace, index, num_dummies=1):
    """Insert dummy packets to the trace at a given position.

    :param trace: Trace
    :param index: Index before which to insert the packets
    :param num_dummies: Number of dummy packets to insert

    >>> insert_dummy_packets([1, -1, 1], 0)
    [1, 1, -1, 1]
    >>> insert_dummy_packets([1, -1, 1], 1)
    [1, 1, -1, 1]
    >>> insert_dummy_packets([1, -1, 1], 2)
    [1, -1, 1, 1]
    >>> insert_dummy_packets([1, -1, 1], 3)
    [1, -1, 1, 1]
    >>> insert_dummy_packets([1, -1, 1], 0, num_dummies=2)
    [1, 1, 1, -1, 1]
    """
    if index > 0 and trace[index - 1] == 0:
        return None
    extended = trace[:index] + [1] * num_dummies + trace[index:]
    return extended

