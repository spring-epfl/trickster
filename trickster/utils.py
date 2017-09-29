try:
    from Queue import PriorityQueue
except ImportError:
    from queue import PriorityQueue

import xxhash
_hash_state = xxhash.xxh64()


def fast_hash(obj):
    '''
    Fast hashing for numpy arrays.

    https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array#16592241
    '''
    _hash_state.update(obj)
    result = _hash_state.intdigest()
    _hash_state.reset()
    return result


def generalized_graph_search(
        initial_node,
        iter_lim,
        beam_size,
        generate_neighbours_fn,
        goal_predicate,
        fscore_fn=None,
        sort_neighbours=True):
    '''
    Beam search with swappable parts.

    :param initial_node: Initial node.
    :param iter_lim: Max number of iterations to try.
    :param beam_size: Max number of expanded neighbours at each
            iteration (beam).
    :param generate_neighbours_fn: For a node, returns an iterable of tuples
            (g-score, neighbour).
    :param fscore_fn: Takes a parent node, a child node, its g-score wrt to
            the parent, its rank within a beam, and returns the f-score.
    :param sort_neighbours: Whether to sort expanded neighbours at each
            iteration.
            If ``generate_neighbours_fn`` is an infinite generator,
            you can set this flag to ``True`` to expand ``beam_size``
            neighbours each time. Note that if the ``fscore_fn`` uses rank,
            it makes sense to sort the generated neighbours by f-score (either
            by setting the flag to ``True``, or generate them in the correct
            order right away).

    '''
    iter_count = 0
    closed = set()
    pool = PriorityQueue()

    # Calculate initial f-score
    if fscore_fn:
        fscore = fscore_fn(initial_node, initial_node, gscore=0, rank=1)
    else:
        fscore = 0

    pool.put((fscore, initial_node))
    while not pool.empty() and iter_count < iter_lim:
        fscore, node = pool.get()
        if goal_predicate(node):
            return fscore, node

        hashed = fast_hash(node)
        if hashed in closed:
            continue
        closed.add(hashed)

        # Generate neighbours, sort by gscore if not sorted already.
        neighbours = generate_neighbours_fn(node)
        if sort_neighbours:
            neighbours = sorted(neighbours, key=lambda t: t[0])

        for rank, (gscore, neighbour) in enumerate(neighbours):
            # Neighbours that are too low in the ranking are cut off
            if rank >= beam_size:
                break
            if neighbour in closed:
                continue

            # Compute a total score, e.g. by incorporating a heuristic.
            if fscore_fn is not None:
                fscore = fscore_fn(neighbour, gscore, rank)
            else:
                fscore = gscore
            pool.put((fscore, neighbour))

        iter_count += 1
