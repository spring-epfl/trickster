#!/usr/bin/python

from trickster.search import a_star_search
from tests.test_data import *

# Test A* search implementation on the Romanian City Problem
# (Optimal path from Arad to Bucharest)


def heuristic_fn(node):
    return HEURISTIC_MAP[node]

def expand_fn(node):
    return GRAPH_TRANSITIONS[node]

def test_a_star_search():
    print('>>>> Running A* search for the Romanian City Problem.')
    start_node = 'Arad'
    goal_fn = lambda x: x == 'Bucharest'

    # Run the A* search
    goal, path_costs, optimal_path = a_star_search(
        start_node=start_node,
        heuristic_fn=heuristic_fn,
        expand_fn=expand_fn,
        goal_fn=goal_fn,
        return_path=True
    )

    # Test 1 - Check the returned path of the algorithm to be the optimal
    assert OPTIMAL_PATH_FROM_ARAD == optimal_path
    print('>> (1) Returned path is optimal.')

    # Test 2 - Check the costs of the optimal nodes
    for node in optimal_path:
        assert OPTIMAL_COSTS_FROM_ARAD[node] == path_costs[node]
    print('>> (2) Costs of expanded nodes are optimal.')

if __name__ == '__main__':
    test_a_star_search()
