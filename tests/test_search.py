import pytest
import numpy as np

from trickster.search import beam_a_star_search


GRAPH_EDGES = {
    'A': [(90, 'B'), (60, 'C')],
    'B': [(90, 'Z')],
    'C': [(20, 'Z')],
    'Z': [],
}
START = 'A'
FINISH = 'Z'
DISTANCES_TO_Z = {
    'A': 150,
    'B': 90,
    'C': 20,
    'Z': 0,
}


def expand_fn(state, **kwargs):
    return [(dist, value) for dist, value in GRAPH_EDGES[state]]


def goal_predicate(state):
    return state == FINISH


def test_graph_search_dijkstra():
    score, result = beam_a_star_search(
        START,
        goal_predicate,
        expand_fn)
    assert result == FINISH


def heuristic_fn(state):
    return DISTANCES_TO_Z[state]


def test_graph_search_a_star():
    score, result = beam_a_star_search(
        START,
        goal_predicate,
        expand_fn,
        heuristic_fn=heuristic_fn)
    assert result == FINISH


def test_graph_search_iterations_limit():
    output = beam_a_star_search(
        START,
        goal_predicate,
        expand_fn,
        iter_lim=1)
    assert output is None

