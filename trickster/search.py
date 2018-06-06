from boltons.queueutils import PriorityQueue


def _get_optimal_path(predecessors, start_node, node):

    # Reconstruct the optimal path from the start to the current node
    path = [node]

    # Backtrack until starting node is found
    while node != start_node:
        node = predecessors[node]
        path.append(node)
    path.reverse()

    return path


def a_star_search(start_node, expand_fn, goal_fn,
                  heuristic_fn=None, hash_fn=None, return_path=False):
    '''
    A* search.

    Return the target node, the costs of nodes expanded by the algorithm,
    and the optimal path from the initial node to the target node.

    :param start_node: Initial node.
    :param expand_fn: Returns an iterable of tuples (neighbour, cost).
    :param goal_fn: Returns True if the current node is the target node.
    :param heuristic_fn: Returns an estimate of the cost to the target
            node. By default, is a constant 0.
    :param hash_fn: Hash function for nodes. By default equals the
            identity function f(x) = x.
    :param return_path: Whether to return the optimal path from the
            initial node to the target node. By default equals False.
    '''

    # Define default heuristic and hash functions if none given
    if heuristic_fn is None:
        heuristic_fn = lambda _: 0
    if hash_fn is None:
        hash_fn = lambda x: x

    # Define data structures to hold data
    path_costs = {}
    predecessors = {}
    open_set = PriorityQueue()
    closed_set = set()

    # Add the starting node; f-score equal to heuristic.
    path_costs[hash_fn(start_node)] = 0
    f_score = heuristic_fn(start_node)
    open_set.add(start_node, priority=-f_score)

    # Iterate until a goal node is found or open set is empty.
    while len(open_set):

        # Retrieve the node with the lowest f-score.
        node = open_set.pop()
        hashed_node = hash_fn(node)

        # Check if the current node is a goal node.
        if goal_fn(node):
            if return_path:
                optimal_path = _get_optimal_path(
                        predecessors, start_node, node)
                return node, path_costs, optimal_path
            return node, path_costs
        closed_set.add(hashed_node)

        # Iterate through all neighbours of the current node
        for neighbour, cost in expand_fn(node):
            hashed_neighbour = hash_fn(neighbour)
            if hashed_neighbour in closed_set:
                continue

            # Compute tentative path cost from the start node to the neighbour.
            tentative_cost = path_costs[hashed_node] + cost

            # Skip if the tentative path cost is larger or equal than the
            # recorded one (if the latter exists).
            if hashed_neighbour in path_costs and (
                    tentative_cost >= path_costs[hashed_neighbour]):
                continue

            # Record new path cost for the neighbour, the predecessor, and add
            # to open set.
            path_costs[hashed_neighbour] = tentative_cost
            f_score = tentative_cost + heuristic_fn(neighbour)
            open_set.add(neighbour, priority=-f_score)
            if return_path:
                predecessors[hashed_neighbour] = node

    # Goal node is unreachable.
    if return_path:
        return None, path_costs, None
    return None, path_costs

