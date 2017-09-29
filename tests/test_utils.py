import pytest
import numpy as np

from trickster.utils import generalized_graph_search, fast_hash


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


def generate_neighbours_fn(state, **kwargs):
    return [(-dist, value) for dist, value in GRAPH_EDGES[state]]


def goal_predicate(state):
    return state == FINISH


def test_graph_search_dijkstra():
    score, result = generalized_graph_search(
        START, 10, 10,
        generate_neighbours_fn, goal_predicate)
    assert result == FINISH


def heuristic_fn(state):
    return DISTANCES_TO_Z[state]


def fscore_fn(parent_state, new_state, gscore, rank=1):
    return gscore + heuristic_fn(new_state)


def test_graph_search_astar():
    score, result = generalized_graph_search(
        START, 10, 10,
        generate_neighbours_fn, goal_predicate,
        fscore_fn)
    assert result == FINISH


def test_graph_search_iterations_limit():
    output = generalized_graph_search(START, 1, 10,
        generate_neighbours_fn, goal_predicate)
    assert output is None


def test_fast_hash_deterministic():
    assert fast_hash('whatever') == fast_hash('whatever')


def test_fast_hash_works_with_numpy():
    array = np.arange(10)
    assert fast_hash(array) is not None

