import os
import tempfile
import shlex
import subprocess
import pickle

import pytest
import pandas as pd


BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
BOTS_SCRIPT = os.path.join(BASE_DIR, "scripts/bots.py")
DATA_PATH = os.path.join(BASE_DIR, "data/twitter_bots")


def build_cmd(*args, **kwargs):
    """
    >>> build_cmd('script.py', 'train', model_pickle='tmp.pkl', shuffle=True)
    'script.py train --model_pickle "tmp.pkl" --shuffle'
    """
    options = []
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                options.append("--%s" % key)
            else:
                options.append("--no_%s" % key)
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


@pytest.mark.parametrize("heuristic", [
    "dist",
    "dist_grid",

    ## This one is flaky
    # "random",
])
@pytest.mark.parametrize("classifier", ["lr", "svmrbf"])
def test_generation(file_factory, heuristic, classifier):
    with file_factory() as results_file:
        if classifier == "svmrbf":
            reduce_classifier = False
        else:
            reduce_classifier = True

        log = invoke_bots_script(
            "100",
            output_pickle=results_file.name,
            # Let's make the task easy.
            bins=20,
            confidence_level=0.5,
            iter_lim=2,
            heuristic=heuristic,
            classifier=classifier,
            reduce_classifier=reduce_classifier,
            human_dataset_template=DATA_PATH + "/humans/humans.{}.csv",
            bot_dataset_template=DATA_PATH + "/bots/bots.{}.csv",
        )

        assert "found" in log
        results = pd.read_pickle(results_file.name)

    # One example is expected to be found.
    assert len(results[0]["search_results"].found) > 0
    # 41 examples for LR and 42 examples for SVM.
    assert len(results[0]["search_results"]) >= 41

