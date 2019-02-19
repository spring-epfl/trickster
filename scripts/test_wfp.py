import os
import tempfile
import shlex
import subprocess
import pickle

import pytest
import pandas as pd


BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
WFP_SCRIPT = os.path.join(BASE_DIR, "scripts/wfp_attacks.py")
DATA_PATH = os.path.join(BASE_DIR, "data/wfp_traces_toy")

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


def invoke_wfp_script(*args, **kwargs):
    """Run the wfp script and return the contents of the log file."""
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        log_file = f.name
        cmd = build_cmd(WFP_SCRIPT, *args, log_file=log_file, **kwargs)
        print(cmd)
        subprocess.run(cmd, shell=True, check=True, stderr=subprocess.STDOUT)

        f.seek(0)
        return f.read()


@pytest.mark.parametrize("model", ["svmrbf", "lr"])
@pytest.mark.parametrize("features", ["cumul", "raw", "total"])
def test_training(file_factory, model, features):
    with file_factory() as pickle_file:
        log = invoke_wfp_script(
            "train",
            model_pickle=pickle_file.name,
            data_path=DATA_PATH,
            features=features,
            model=model,
            num_traces=MAX_TRACES,
            shuffle=True,
        )
        clf = pickle.load(pickle_file)

    # Check the model is fitted.
    if model == "lr":
        assert hasattr(clf, "coef_")
    elif model == "svmrbf":
        assert hasattr(clf.best_estimator_, "dual_coef_")


@pytest.fixture(params=['lr', 'svmrbf'])
def trained_model(request, file_factory):
    with file_factory() as pickle_file:
        log = invoke_wfp_script(
            "train",
            model_pickle=pickle_file.name,
            data_path=DATA_PATH,
            features="cumul",
            model=request.param,
            num_traces=MAX_TRACES,
        )
        yield pickle_file, request.param


@pytest.mark.parametrize('heuristic', ['dist', 'confidence'])
def test_generation_success(file_factory, trained_model, heuristic):
    with file_factory() as results_file:
        trained_model_pickle, model_type = trained_model
        log = invoke_wfp_script(
            "generate",
            model_pickle=trained_model_pickle.name,
            data_path=DATA_PATH,
            output_pickle=results_file.name,
            # Let's make the task easy.
            epsilon=100,
            heuristic=heuristic,
            num_adv_examples=1,
            confidence_level=0.1,
            iter_lim=10,
        )

        assert "found" in log
        results = pd.read_pickle(results_file.name)

    # One example is expected to be found.
    assert results.found.mean() >= 1.0
    # Should be one, because num_adv_examples == 1.
    assert len(results) == 1


def test_generation_sort_by_len(file_factory, trained_model):
    with file_factory() as results_file:
        trained_model_pickle, model_type = trained_model
        log = invoke_wfp_script(
            "generate",
            model_pickle=trained_model_pickle.name,
            data_path=DATA_PATH,
            output_pickle=results_file.name,
            sort_by_len=True,

            # Don't do any search.
            iter_lim=0,
        )

        results = pd.read_pickle(results_file.name)

    # Examples should be sorted by length.
    prev_len = None
    for _, row in results.iterrows():
        assert prev_len is None or len(row.x) <= prev_len
        prev_len = len(row.x)

