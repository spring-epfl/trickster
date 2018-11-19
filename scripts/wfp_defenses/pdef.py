import os
import sys

if __name__ == '__main__':
    if len(sys.argv)!= 2:
        exit("Need max trace length parameter")

    max_trace_len = int(sys.argv[1])


    if not os.path.exists("../../data/batchusenix-pdef/"):
        os.makedirs("../../data/batchusenix-pdef")

    if not os.path.exists("../../data/batch/"):
        print("batch folder needs to exist")
        sys.exit(0)

    fold = "../../data/batch/"
    foldout = "../../data/batchusenix-pdef/"
    count = 0
    for site in range(0, 100):
        print(site)
        for inst in range(0, 90):
            count += 1
            f1 = open(fold + str(site) + "-" + str(inst), "r")
            lines1 = f1.readlines()
            f1.close()
            if len(lines1) > max_trace_len:
                continue
            f2 = open(fold + str(count), "r")
            lines2 = f2.readlines()
            f2.close()
            start1 = float(lines1[0].split("\t")[0])
            start2 = float(lines2[0].split("\t")[0])
            packets = []
            for x in lines1:
                x = x.split("\t")
                packets.append([float(x[0]) - start1, x[1]])
            for x in lines2:
                x = x.split("\t")
                packets.append([float(x[0]) - start2, x[1]])
            packets = sorted(packets, key = lambda packets:packets[0])
            fout = open(foldout + str(site) + "-" + str(inst), "w")
            for x in packets:
                fout.write(str(x[0]) + "\t" + x[1])
            fout.close()

    ##for site in range(0, 9000):
    ##    if (site % 100 == 0):
    ##        print(site)
    ##    f1 = open(fold + str(site), "r")
    ##    lines1 = f1.readlines()
    ##    f1.close()
    ##    f2 = open(fold + str(9000-site), "r")
    ##    lines2 = f2.readlines()
    ##    f2.close()
    ##    start1 = float(lines1[0].split("\t")[0])
    ##    start2 = float(lines2[0].split("\t")[0])
    ##    packets = []
    ##    for x in lines1:
    ##        x = x.split("\t")
    ##        packets.append([float(x[0]) - start1, x[1]])
    ##    for x in lines2:
    ##        x = x.split("\t")
    ##        packets.append([float(x[0]) - start2, x[1]])
    ##    packets = sorted(packets, key = lambda packets:packets[0])
    ##    fout = open(foldout + str(site), "w")
    ##    for x in packets:
    ##        fout.write(str(x[0]) + "\t" + x[1])
    ##    fout.close()
