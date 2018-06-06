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

