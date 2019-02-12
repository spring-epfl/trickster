#!/usr/bin/env python

import sys
import pickle


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        sys.exit(0)

    joined_report = []
    for report_path in sys.argv[1:]:
        with open(report_path, "rb") as f:
            joined_report.extend(pickle.load(f))

    sys.stdout.buffer.write(pickle.dumps(joined_report))
