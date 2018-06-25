from boltons.queueutils import PriorityQueue


def _get_optimal_path(predecessors, start_node, node, hash_fn):
    # Reconstruct the optimal path from the start to the current node
    path = [node]
    hashed_start = hash_fn(start_node)

    # Backtrack until starting node is found
    while hash_fn(node) != hashed_start:
        node = predecessors[hash_fn(node)]
        path.append(node)
    path.reverse()

    return path


def a_star_search(
        start_node, 
        expand_fn, 
        goal_fn,
        heuristic_fn=None, 
        hash_fn=None,
        iter_lim=None,
        return_path=False
    ):
    '''
    A* search.

    Returns the tuple (cost, target_node) if return_path is set to False.
    Otherwise, returns the target node, the costs of nodes expanded by the
    algorithm, and the optimal path from the initial node to the target node.

    :param start_node: Initial node.
    :param expand_fn: Returns an iterable of tuples (neighbour, cost).
    :param goal_fn: Returns True if the current node is the target node.
    :param heuristic_fn: Returns an estimate of the cost to the target
            node. By default, is a constant 0.
    :param hash_fn: Hash function for nodes. By default equals the
            identity function f(x) = x.
    :param iter_lim: Maximum number of iterations to try.
    :param return_path: Whether to return the optimal path from the
            initial node to the target node. By default equals False.
    '''

    # Define default heuristic and hash functions if none given
    if heuristic_fn is None:
        heuristic_fn = lambda _: 0
    if hash_fn is None:
        hash_fn = lambda x: x
       
    iter_count = 0

    # Define data structures to hold data
    path_costs = {}
    predecessors = {}
    reverse_hashes = {}
    open_set = PriorityQueue()
    closed_set = set()

    # Add the starting node; f-score equal to heuristic.
    hashed_start = hash_fn(start_node)
    path_costs[hashed_start] = 0
    f_score = heuristic_fn(start_node)
    open_set.add(hashed_start, priority=-f_score)
    reverse_hashes[hashed_start] = start_node

    # Iterate until a goal node is found, open set is empty
    # or max number of iterations has been reached.
    while len(open_set) and (iter_lim is None or iter_count < iter_lim):

        # Retrieve the node with the lowest f-score.
        hashed_node = open_set.pop()
        node = reverse_hashes[hashed_node]

        # Check if the current node is a goal node.
        if goal_fn(node):
            if return_path:
                optimal_path = _get_optimal_path(
                        predecessors, start_node, node, hash_fn)
                return node, path_costs, optimal_path
            else:
                return (node, path_costs[hashed_node])
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
            open_set.add(hashed_neighbour, priority=-f_score)
            reverse_hashes[hashed_neighbour] = neighbour
            if return_path:
                predecessors[hashed_neighbour] = node
                
        iter_count += 1

    # Goal node is unreachable.
    if return_path:
        return None, path_costs, None
    else:
        return None, None


def _bounded_search_recusive(path, path_costs, bound, expand_fn,
                             goal_fn, heuristic_fn, hash_fn, reverse_hashes):

    # Expand the last node on the search path.
    hashed_node = path[-1]
    node = reverse_hashes[hashed_node]
    f_score = path_costs[hashed_node] + heuristic_fn(node)

    # Backtrack if f-score exceeds the bound.
    if f_score > bound:
        return False, f_score, node, path
    if goal_fn(node):
        return True, f_score, node, path

    min_score = None

    # Iterate through all neighbours of the current node.
    for neighbour, cost in expand_fn(node):
        hashed_neighbour = hash_fn(neighbour)

        # Expand the neighbour only if it was not already visited.
        if hashed_neighbour not in path:
            path.append(hashed_neighbour)
            reverse_hashes[hashed_neighbour] = neighbour
            path_costs[hashed_neighbour] = path_costs[hashed_node] + cost

            # Call the search recursively on the neighbour with new path and costs.
            output = _bounded_search_recusive(
                path,
                path_costs,
                bound,
                expand_fn,
                goal_fn,
                heuristic_fn,
                hash_fn,
                reverse_hashes
            )

            is_found, score, candidate_node, candidate_path = output

            if is_found:
                return True, score, candidate_node, candidate_path

            # If score is None then leave min_score None,
            # otherwise update if score is smaller.
            if min_score is None or score is None or score < min_score:
                min_score = score

            # Remove the neighbour from the path.
            path.pop()

    return False, min_score, None, None


def _bounded_search(path, path_costs, bound, expand_fn,
                    goal_fn, heuristic_fn, hash_fn, reverse_hashes):

    # Obtain the starting node and its cost.
    hashed_node = path[-1]
    path_cost = path_costs[hashed_node]

    # Reset the path.
    path = []

    # Initialise stack to imitate recursive calls.
    # Stack holds tuples (current node, cost to current node and predecessor).
    stack = [(hashed_node, path_cost, None)]
    min_score = None

    # Iterate while stack is not empty.
    while len(stack):

        hashed_node, path_cost, predecessor = stack.pop()

        # Backtracks if the last node on the path was already expanded.
        while predecessor and predecessor != path[-1]:
            path.pop()

        path.append(hashed_node)
        node = reverse_hashes[hashed_node]
        path_costs[hashed_node] = path_cost
        f_score = path_cost + heuristic_fn(node)

        # Backtrack if f-score exceeds the bound.
        if f_score > bound:
            path.pop()
            if min_score is None or f_score < min_score:
                min_score = f_score
            continue
        if goal_fn(node):
            return True, f_score, node, path

        # Iterate through all neighbours of the current node.
        for neighbour, cost in reversed(expand_fn(node)):
            hashed_neighbour = hash_fn(neighbour)

            # Expand the neighbour only if it was not already visited.
            if hashed_neighbour not in path:
                reverse_hashes[hashed_neighbour] = neighbour
                stack.append((hashed_neighbour, path_cost + cost, hashed_node))

    return False, min_score, None, None


def ida_star_search(
        start_node, 
        expand_fn, 
        goal_fn,
        heuristic_fn=None, 
        hash_fn=None, 
        return_path=False
    ):
    '''
    IDA* search.

    Returns the tuple (cost, target_node) if return_path is set to False.
    Otherwise, returns the target node, the costs of nodes expanded by the
    algorithm, and the optimal path from the initial node to the target node.

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

    # Define default heuristic and hash functions if none given.
    if heuristic_fn is None:
        heuristic_fn = lambda _: 0
    if hash_fn is None:
        hash_fn = lambda x: x

    # Define data structures to hold data.
    path_costs = {}
    reverse_hashes = {}

    # Add the starting node; bound equal to heuristic.
    hashed_start = hash_fn(start_node)
    path_costs[hashed_start] = 0
    bound = heuristic_fn(start_node)
    path = [hashed_start]
    reverse_hashes[hashed_start] = start_node

    # Iterate until found or score is None (i.e. no children).
    while True:
        output = _bounded_search(
            path,
            path_costs,
            bound,
            expand_fn,
            goal_fn,
            heuristic_fn,
            hash_fn,
            reverse_hashes
        )

        is_found, score, candidate_node, candidate_path = output

        if is_found:
            if return_path:
                optimal_path = [reverse_hashes[x] for x in candidate_path]
                return candidate_node, path_costs, optimal_path
            else:
                return candidate_node, score

        # Goal node is unreachable.
        if score is None:
            if return_path:
                return None, path_costs, None
            else:
                return None, None

        # Set the bound to be equal to the lowest f-score encountered.
        bound = score
