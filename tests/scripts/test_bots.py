import os
import tempfile
import shlex
import subprocess
import pickle

import pytest
import pandas as pd


BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
BOTS_SCRIPT = os.path.join(BASE_DIR, "scripts/bots.py")

MAX_TRACES = 50


def build_cmd(*args, **kwargs):
    """
    >>> build_cmd('script.py', 'train', model_pickle='tmp.pkl', shuffle=True)
    'script.py train --model_pickle "tmp.pkl" --shuffle'
    """
    options = []
    for key, value in kwargs.items():
        if isinstance(value, bool):
            options.append("--%s" % key)
        elif isinstance(value, int) or isinstance(value, float):
            options.append("--%s %s" % (key, value))
        else:
            options.append('--%s "%s"' % (key, value))

    return " ".join(list(args) + options)


def invoke_bots_script(*args, **kwargs):
    """Run the bots script and return the contents of the log file."""
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        log_file = f.name
        cmd = build_cmd(BOTS_SCRIPT, *args, log_file=log_file, **kwargs)
        print(cmd)
        subprocess.run(cmd, shell=True, check=True, stderr=subprocess.STDOUT)

        f.seek(0)
        return f.read()


@pytest.mark.parametrize("heuristic", ["dist", "dist_grid", "random"])
def test_generation(file_factory, heuristic):
    with file_factory() as results_file:
        log = invoke_bots_script(
            "100",
            output_pickle=results_file.name,
            # Let's make the task easy.
            bins=20,
            confidence_level=0.5,
            iter_lim=2,
            heuristic=heuristic,
        )

        assert "found" in log
        results = pd.read_pickle(results_file.name)

    # One example is expected to be found.
    assert len(results[0]["search_results"].found) > 0
    assert len(results[0]["search_results"]) == 41

