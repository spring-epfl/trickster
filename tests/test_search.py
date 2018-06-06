import pytest

from trickster.search import a_star_search
from trickster.utils.romania import *


def heuristic_fn(node):
    return HEURISTIC_MAP[node]


def expand_fn(node):
    return GRAPH_TRANSITIONS[node]


HEURISTIC_SEARCH_FUNCS = [a_star_search]


@pytest.mark.parametrize('search_fn', HEURISTIC_SEARCH_FUNCS)
def test_optimal_search_path(search_fn):
    start_node = 'Arad'
    goal_fn = lambda x: x == 'Bucharest'

    goal, path_costs, optimal_path = search_fn(
        start_node=start_node,
        heuristic_fn=heuristic_fn,
        expand_fn=expand_fn,
        goal_fn=goal_fn,
        return_path=True
    )

    assert OPTIMAL_PATH_FROM_ARAD == optimal_path


@pytest.mark.parametrize('search_fn', HEURISTIC_SEARCH_FUNCS)
@pytest.mark.parametrize('target_node', OPTIMAL_COSTS_FROM_ARAD.keys())
def test_optimal_search_costs(search_fn, target_node):
    start_node = 'Arad'
    goal_fn = lambda x: x == target_node

    goal, path_costs = search_fn(
        start_node=start_node,
        heuristic_fn=heuristic_fn,
        expand_fn=expand_fn,
        goal_fn=goal_fn,
        return_path=False
    )

    for node, cost in path_costs.items():
        if node in optimal_path:
            assert cost == OPTIMAL_COSTS_FROM_ARAD[node]
