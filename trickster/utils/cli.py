import functools


def add_options(options):
    """Combine several click options in a single decorator.

    https://github.com/pallets/click/issues/108#issuecomment-255547347
    """

    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options
