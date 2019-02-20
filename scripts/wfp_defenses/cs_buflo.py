# Implementation of the CS-BuFLO defense (Cai et al., 14).
# Author: Giovanni Cherubin (@gchers)

import os
import sys
import math
import random
from collections import deque
from shutil import copyfile

import click
import numpy as np

from tqdm import tqdm
from numpy import median

# Direction
IN = -1
OUT = 1
MTU = 1
INF = float("inf")

# Upstream: from the current endpoint to the defence
# Downstream: from the other endpoint to the defence
# Define MODES of packets
UPSTREAM_APPLICATION_DATA = "upstream_application_data"
DOWNSTREAM_APPLICATION_DATA = "downstream_application_data"
ON_LOAD = "onLoad"
TIMEOUT = "timeout"

# Time is specified in ms
# NOTE: a QUIET_TIME too small causes the defence to fail,
# raising a TimeTravel exception: one endpoint was silent
# for too long, and then wants to catch up with the padding
# (junk) messages it didn't send.
# NOTE: The above problem was solved by using endpoint's
# "more_data" attribute, which is set to False only when
# the endpoint has no more data to transmit
QUIET_TIME = 2000
INIT_RHO = 200
MIN_RHO = 2 ** (-4) * 1000
MAX_RHO = 2 ** 3 * 1000


class TimeTravel(Exception):
    """This exception is raised when there are time inconsistencies.
    """

    pass


class Queue(deque, object):
    """A deque that allows seeing the element at the
    left of the queue without popping it.
    """

    def seeleft(self):
        """Returns the next element at the left of the
        queue, without removing it from the queue.

        Returns None if the queue is empty.
        """
        if self:
            return self[0]
        else:
            return None


class CSBuFLOEndpoint:
    """Implements one endpoint (client, server) of the CS-BuFLO defence as
    described in:
        http://pub.cs.sunysb.edu/~rob/papers/csbuflo.pdf
    """

    def __init__(self, output_trace, direction, initial_rho=0.2, quiet_time=QUIET_TIME):
        # Direction is OUT for client, IN for server
        self.direction = direction
        # Defended packets
        self.defended_trace = output_trace
        # Internal buffer for packets to send
        self.output_buff = Queue()  # Packets to send
        # Packets that have been sent
        # NOTE: don't rely on this to see which packets were sent,
        # because the function defend() pops from this queue.
        self.sent = Queue()
        # Bytes of real and padding (junk) traffic sent
        self.real_bytes = 0
        self.junk_bytes = 0
        self.last_site_response_time = None
        # onLoad and padding_done events
        self.on_load = False
        self.padding_done = False
        # The timeout it sampled uniformly in [0, 2*self.rho]
        # rho_stats are used to adjust rho
        self.initial_rho = initial_rho
        self.rho = initial_rho
        self.rho_stats = []
        self.timeout = initial_rho
        # "As a backup mechanism, the CS-BuFLO server considers
        # the website idle if quiet-time seconds pass without receiving
        # new data from the website. We used a quiet-time of 2
        # seconds in our prototype implementation."
        # (Section 4.5)
        self.quiet_time = quiet_time
        # Triggered to True when self.done_xmitting() == True
        self.done = False
        # This was not present in the original paper. Is is
        # set to False when no more data needs to be transmitted
        # from this endpoint
        self.more_data = True

    def next_timeout(self):
        ## No timeout if it finished transmitting
        # if self.done_xmitting():
        if self.padding_done:
            return INF
        # NOTE: need to check last sent packet time on defended_trace
        # rather than self.sent, as self.sent may be changed externally
        last_time = 0.0
        if self.defended_trace:
            for i in range(len(self.defended_trace) - 1, -1, -1):
                if self.defended_trace[i][1] == MTU * self.direction:
                    # last_time, _ = self.defended_trace[i]
                    last_time, _ = self.defended_trace[i]
                    break

        return last_time + self.timeout

    def process(self, packet, mode):
        # print('process {} {}'.format(packet, mode))
        # print('rho {}'.format(self.rho))
        # print('timeout {}'.format(self.timeout))
        time, size = packet

        if mode == UPSTREAM_APPLICATION_DATA:
            # this endpoint is trying to send data
            self.output_buff.append((time, size))
            self.real_bytes += abs(size)
            self.last_site_response_time = time
            # self.padding_done = False
        elif mode == DOWNSTREAM_APPLICATION_DATA:
            # would pass data back to this endpoint
            self.rho_stats.append(None)
            self.on_load = False
            # NOTE: This is only done for the server
            # endpoint (i.e., direction == IN)
            # This was not specified by the original paper
            # but without this we would encounter instable
            # situations
            if self.direction == IN:
                self.padding_done = False
        elif mode == ON_LOAD:
            self.on_load = True
        elif mode == TIMEOUT:
            if self.output_buff:
                self.rho_stats.append(time)
            padding = self.cs_send(time)
            self.junk_bytes += padding

        if self.done_xmitting(time):
            self.padding_done = True
        else:
            # Set rho to the average time between sends to client
            if self.rho == INF:
                self.rho = self.initial_rho
            elif self.crossed_threshold():
                self.rho = self.rho_estimator()
                self.rho_stats = []

            if mode == TIMEOUT:
                # Random in [0, 2*rho]
                self.timeout = random.random() * 2 * self.rho

    def cs_send(self, time):
        padding = 0

        if self.output_buff:
            _, size = self.output_buff.popleft()
            # We use the timeout time
            packet = (time, size)
        else:
            # Padding packet
            packet = (time, self.direction * MTU)
            padding = 1

        self.sent.append(packet)
        self.defended_trace.append(packet)

        return padding

    def padding_finished(self):
        """Using "payload padding", which pads until the bytes
        sent are a multiple of the real bytes.

        Implements "Payload padding" as in paper.
        """
        if self.real_bytes == 0:
            return False

        total_bytes = self.real_bytes + self.junk_bytes
        next_multiple = 2 ** math.ceil(math.log(self.real_bytes, 2))

        return total_bytes % next_multiple == 0

    def done_xmitting(self, cur_time):
        condition1 = self.padding_finished() or self.crossed_threshold()
        # NOTE: The paper writes "length(output_buff) <- 0" as first
        # condition. Since the expression makes no sense, I assume
        # the authors meant "length(output_buff) == 0".
        condition2 = not self.output_buff
        condition3 = self.channel_idle(cur_time)
        # NOTE: I add the following condition, because if one
        # endpoint gets silent for too long and then wants to transmit
        # again the defence would fail
        condition4 = not self.more_data

        # print(self.direction, condition1, condition2, condition3, condition4)

        return condition1 and condition2 and condition3 and condition4

    def channel_idle(self, cur_time):
        if self.last_site_response_time is not None:
            is_quiet = self.last_site_response_time + self.quiet_time < cur_time
        else:
            is_quiet = False

        return self.on_load or is_quiet

    def rho_estimator(self):
        it = []
        for i in range(len(self.rho_stats) - 1):
            if self.rho_stats[i] is not None and self.rho_stats[i + 1] is not None:
                it.append(self.rho_stats[i + 1] - self.rho_stats[i])

        if not it:
            return self.rho

        med = median(it)
        if not med:
            rho = self.rho

        rho = 2 ** math.floor(math.log(med, 2))

        # NOTE: to my understanding, the original CS-BuFLO
        # implemented in SSH has a minimum and maximum RHO
        # values. By doing some experiments I observed the
        # following values allow containing overheads
        if rho < MIN_RHO:
            rho = MIN_RHO
        elif rho > MAX_RHO:
            rho = MAX_RHO

        return rho

    def crossed_threshold(self):
        # From Algorithm 1 in the paper, this function seems to be
        # called on two arguments: (real_bytes, junk_bytes).
        # However, the function only accepts one argument (Algorithm 4).
        # Thankfully, DONE_XMITTING() in Algorithm 4 calls the function
        # on real_bytes + junk_bytes, so I assume this is what was
        # intended in Algorithm 1.
        x = self.real_bytes + self.junk_bytes

        # NOTE: I'm adding this, because otherwise (_by design_)
        # when x == MTU (which does happen) the log is indefinite
        if x == MTU or x == 0:
            return False

        # The following comment does not apply anymore, and I'll
        # only keep it for record.
        ## NOTE: This is an edit from the original function.
        ## As in Figure 1, it returns True only if the number
        ## of bytes sent so far is a multiple of 2^k, for
        ## some integer k
        ##log = math.log(x, 2)
        ##return log == math.floor(log)

        return math.floor(math.log(x - MTU, 2)) < math.floor(math.log(x, 2))


class CSBuFLO:
    def __init__(self, initial_rho=INIT_RHO):
        self.initial_rho = initial_rho

    def reset(self):
        self.defended_trace = Queue()
        self.client = CSBuFLOEndpoint(self.defended_trace, OUT, self.initial_rho)
        self.server = CSBuFLOEndpoint(self.defended_trace, IN, self.initial_rho)
        # For each endpoint (client, server), there are two
        # queues of packets: those read from the trace and those
        # that were actually sent by the defence (output).
        # The output ones can be found as client.sent, server.sent,
        # the others we define as follows
        self.client_packets = Queue()
        self.server_packets = Queue()

    def defend(self, packets):
        self.reset()

        # Add packets to the respective pipelines
        for t, s in packets:
            if s > 0:
                self.client_packets.append((t, s))
            else:
                self.server_packets.append((t, s))

        # Record what events happen for debugging purposes
        nevents = {}

        # Keep time, to check for errors
        self.cur_time = 0.0

        running = True
        while running:
            # Detect onLoad
            cond1 = not self.client_packets and not self.client.output_buff
            cond2 = not self.server_packets and not self.server.output_buff
            if cond1 and cond2:
                # print('onLoad')
                self.server.process((self.cur_time, None), ON_LOAD)
                self.client.process((self.cur_time, None), ON_LOAD)

            # Notify endpoint if no more data to transmit
            if not self.client_packets:
                self.client.more_data = False
            if not self.server_packets:
                self.server.more_data = False

            # We break if either there's no more data to transmit
            # or self.process_next() returns None; the latter happens
            # when there are no more packets to transmit and
            # padding is done.
            if self.client.done_xmitting(self.cur_time) and self.server.done_xmitting(
                self.cur_time
            ):
                break

            # Keep log of events
            next_event = self.process_next()
            if next_event not in nevents:
                nevents[next_event] = 1
            else:
                nevents[next_event] += 1

            # print('client out: {}'.format(self.client.output_buff))
            # print('server out: {}'.format(self.server.output_buff))
            ##print('defended: {}'.format(defended_trace))
            # print('client rho: {}'.format(self.client.timeout))
            # print('server rho: {}'.format(self.server.timeout))

        return list(self.defended_trace)

    def process_next(self):
        """Process next event.

        Get the time of each event, and process the one
        that comes first.
        """
        # Find what is the next event
        time_c_read_packet = self.client_packets.seeleft()
        time_s_read_packet = self.server_packets.seeleft()
        # Little hack to use the "min" expression later on both
        # packets and timeouts
        c_timeout = (self.client.next_timeout(),)
        s_timeout = (self.server.next_timeout(),)

        t_events = [time_c_read_packet, time_s_read_packet, c_timeout, s_timeout]

        next_packet = min([x for x in t_events if x is not None], key=lambda x: x[0])
        next_event = t_events.index(next_packet)

        # DEBUG
        # print('t-events {}'.format(t_events))
        # print('next {}'.format(next_event))
        # print(self.client.timeout)
        # print(len(self.client_packets))

        # Check the time is consistent
        next_time = next_packet[0]
        # NOTE: if next_time is INF, it means both endpoints
        # have been silent for a while. Try increasing QUIET_TIME.
        if next_time < self.cur_time:
            print("I travelled back in time:")
            print("{} => {}".format(self.cur_time, next_time))
            print(t_events)
            raise TimeTravel
        self.cur_time = next_time

        # Process the event
        process_event = {
            0: self.client_read,
            1: self.server_read,
            2: self.client_timeout,
            3: self.server_timeout,
        }

        process_event[next_event]()
        # After event, send packet (if any) to the other
        # endpoint
        if self.client.sent.seeleft() is not None:
            self.client_sent()
        if self.server.sent.seeleft() is not None:
            self.server_sent()

        return next_event

    def client_read(self):
        packet = self.client_packets.popleft()
        self.client.process(packet, UPSTREAM_APPLICATION_DATA)

    def server_read(self):
        packet = self.server_packets.popleft()
        self.server.process(packet, UPSTREAM_APPLICATION_DATA)

    def client_sent(self):
        packet = self.client.sent.popleft()
        self.server.process(packet, DOWNSTREAM_APPLICATION_DATA)

    def server_sent(self):
        packet = self.server.sent.popleft()
        self.client.process(packet, DOWNSTREAM_APPLICATION_DATA)

    def client_timeout(self):
        packet = (self.client.next_timeout(), None)
        self.client.process(packet, TIMEOUT)

    def server_timeout(self):
        packet = (self.server.next_timeout(), None)
        self.server.process(packet, TIMEOUT)


def normalise_timings(packets):
    """Returns a packet sequence with sorted packets, and
    timing starting at 0.
    """
    if not packets:
        return []
    res = sorted(packets, key=lambda x: x[0])
    min_time = res[0][0]

    for i in range(len(res)):
        t, s = res[i]
        res[i] = (t - min_time, s)
        if t == INF:
            print("Try increasing QUIET_TIME")
            raise TimeTravel

    return res


@click.command()
@click.option("--data_path")
@click.option("--out_path")
def main(data_path, out_path):
    DEFENDED = out_path
    dataset = data_path
    outdirectory = DEFENDED

    if not os.path.exists(outdirectory):
        os.makedirs(outdirectory)

    unmod = []
    mod = []
    added = []
    count = 0

    # Defend
    for fname in tqdm(os.listdir(dataset)):
        infname = os.path.join(dataset, fname)
        outfname = os.path.join(outdirectory, fname)
        # Skip open world traces
        if "-" not in fname:
            # copyfile(infname, outfname)
            continue

        packets = []
        with open(infname, "r") as f:
            for x in f.readlines():
                t, s = x.split("\t")
                t = float(t) * 1000.0  # To milliseconds
                s = int(s)
                packets.append((t, s))
        success = False
        while not success:
            try:
                defended = CSBuFLO().defend(packets)
                success = True
            except TimeTravel:
                pass

        defended = normalise_timings(defended)

        # Store defended trace
        with open(outfname, "w") as f:
            for t, s in defended:
                t /= 1000.0  # To seconds
                s = 1 if s > 0 else -1
                f.write(repr(t) + "\t" + repr(s) + "\n")


if __name__ == "__main__":
    main()
