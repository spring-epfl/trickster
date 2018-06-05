import attr

from boltons.queueutils import PriorityQueue

from .utils import fast_hash


def beam_a_star_search(
        initial_node,
        goal_predicate,
        expand_fn,
        heuristic_fn=None,
        iter_lim=None,
        beam_size=None,
        sort_neighbours=True,
        hash_fn=None):
    '''
    Beam A* search with swappable parts.

    :param initial_node: Initial node.
    :param expand_fn: Returns an iterable of tuples (cost, candidate).
    :param heuristic_fn: Returns an estimate of the cost to the target
            node. By default, is a constant 0.
    :param iter_lim: Max number of iterations to try.
    :param beam_size: Max number of expanded candidates at each
            iteration (beam).
    :param hash_fn: Hash function for nodes. By default set to
            :py:func:`utils.fash_hash`
    '''

    if heuristic_fn is None:
        heuristic_fn = lambda _: 0
    if hash_fn is None:
        hash_fn = lambda node: node

    iter_count = 0
    closed = set()
    pool = PriorityQueue()

    score = heuristic_fn(initial_node)
    cost_by_node = {}

    pool.add((0, initial_node), priority=-score)
    while not len(pool) == 0 and (iter_lim is None or iter_count < iter_lim):
        current_cost, node = pool.pop()
        if goal_predicate(node):
            return score, node

        hashed = hash_fn(node)
        if hashed in closed:
            continue
        closed.add(hashed)

        # Generate candidates.
        candidates = expand_fn(node)
        for index, (transition_cost, candidate) in enumerate(candidates):
            if beam_size is not None and index >= beam_size:
                break
            if candidate in closed:
                continue
            candidate_hash = hash_fn(candidate)
            candidate_cost = current_cost + transition_cost
            if candidate_hash in cost_by_node and (
                    candidate_cost > cost_by_node[candidate_hash]):
                continue

            # Compute the total score.
            score = candidate_cost + heuristic_fn(candidate)

            # Add to the pool and the cost dictionary.
            pool.add((candidate_cost, candidate), priority=-score)
            cost_by_node[candidate_hash] = candidate_cost

        iter_count += 1

