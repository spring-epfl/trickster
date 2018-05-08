import attr
try:
    from Queue import PriorityQueue
except ImportError:
    from queue import PriorityQueue

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
    :param expand_fn: Returns an iterable of tuples (cost, neighbour).
    :param heuristic_fn: Returns an estimate of the cost to the target
            node. By default, is a constant 0.
    :param iter_lim: Max number of iterations to try.
    :param beam_size: Max number of expanded neighbours at each
            iteration (beam).
    :param sort_neighbours: Whether to sort expanded neighbours at each
            iteration.
            If ``expand_fn`` is an infinite generator, you can set this
            flag to ``True`` to expand ``beam_size`` neighbours each time.
    :param hash_fn: Hash function for nodes. By default set to
            :py:func:`utils.fash_hash`
    '''

    if heuristic_fn is None:
        heuristic_fn = lambda _: 0

    iter_count = 0
    closed = set()
    pool = PriorityQueue()

    # Calculate initial score
    score = heuristic_fn(initial_node)

    pool.put((score, initial_node))
    while not pool.empty() and (iter_lim is None or iter_count < iter_lim):
        score, node = pool.get()
        if goal_predicate(node):
            return score, node

        hashed = fast_hash(node)
        if hashed in closed:
            continue
        closed.add(hashed)

        # Generate neighbours, sort by cost if not sorted already.
        neighbours = expand_fn(node)
        if sort_neighbours:
            neighbours = sorted(neighbours, key=lambda t: t[0])

        for index, (cost, neighbour) in enumerate(neighbours):
            # Neighbours that are too low in the ranking are cut off
            if beam_size is not None and index >= beam_size:
                break
            if neighbour in closed:
                continue

            # Compute the total score.
            score = cost + heuristic_fn(neighbour)
            pool.put((score, neighbour))

        iter_count += 1

