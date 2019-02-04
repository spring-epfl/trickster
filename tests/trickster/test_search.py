import pytest

from trickster.search import a_star_search, ida_star_search, generalized_a_star_search
from trickster.utils.romania import *


def heuristic_fn(node):
    return HEURISTIC_MAP[node]


def expand_fn(node):
    return GRAPH_TRANSITIONS[node]


HEURISTIC_SEARCH_FUNCS = [a_star_search, ida_star_search]
HASH_FUNCS = [None, hash]


@pytest.mark.parametrize("search_fn", HEURISTIC_SEARCH_FUNCS)
@pytest.mark.parametrize("hash_fn", HASH_FUNCS)
def test_optimal_search_path(search_fn, hash_fn):
    start_node = "Arad"
    goal_fn = lambda x: x == "Bucharest"

    goal, path_costs, optimal_path = search_fn(
        start_node=start_node,
        heuristic_fn=heuristic_fn,
        expand_fn=expand_fn,
        goal_fn=goal_fn,
        hash_fn=hash_fn,
        return_path=True,
    )

    assert OPTIMAL_PATH_FROM_ARAD == optimal_path


@pytest.mark.parametrize("search_fn", HEURISTIC_SEARCH_FUNCS)
@pytest.mark.parametrize("target_node", OPTIMAL_COSTS_FROM_ARAD.keys())
@pytest.mark.parametrize("hash_fn", HASH_FUNCS)
def test_optimal_search_target_node(search_fn, target_node, hash_fn):
    start_node = "Arad"
    goal_fn = lambda x: x == target_node

    # Do not use heuristic here, since it shows distance to Bucharest.
    goal, cost = search_fn(
        start_node=start_node,
        expand_fn=expand_fn,
        goal_fn=goal_fn,
        hash_fn=hash_fn,
        return_path=False,
    )

    assert goal == target_node
    assert cost == OPTIMAL_COSTS_FROM_ARAD[goal]


@pytest.mark.parametrize("search_fn", HEURISTIC_SEARCH_FUNCS)
@pytest.mark.parametrize("target_node", OPTIMAL_COSTS_FROM_ARAD.keys())
@pytest.mark.parametrize("hash_fn", HASH_FUNCS)
def test_optimal_search_costs(search_fn, target_node, hash_fn):
    start_node = "Arad"
    goal_fn = lambda x: x == target_node

    # Do not use heuristic here, since it shows distance to Bucharest.
    goal, path_costs, optimal_path = search_fn(
        start_node=start_node,
        expand_fn=expand_fn,
        goal_fn=goal_fn,
        hash_fn=hash_fn,
        return_path=True,
    )

    for node, cost in path_costs.items():
        if node in optimal_path:
            assert cost == OPTIMAL_COSTS_FROM_ARAD[node]


@pytest.mark.parametrize("beam_size,expected", [(1, False), (2, False), (5, True)])
def test_beam_search(beam_size, expected):
    # Hill climbing and beam search with beam under 5 don't find the path
    # to Fagaras.
    start_node = "Arad"
    goal_fn = lambda x: x == "Fagaras"

    goal, _, path = generalized_a_star_search(
        start_node=start_node,
        expand_fn=expand_fn,
        goal_fn=goal_fn,
        beam_size=beam_size,
        return_path=True,
    )

    assert goal is None if not expected else goal is not None
