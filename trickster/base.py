import abc
import attr
import typing


@attr.s(auto_attribs=True)
class GraphSearchProblem:
    """
    Functions that define a graph search instance.

    :param search_fn: Graph search function.

    :param expand_fn: Returns the expanded neighbour nodes for a given node.
    :param goal_fn: Predicate that tells whether a given node is a target node.
    :param heuristic_fn: Returns an estimate of how far the given example.
    :param hash_fn: Hash function for nodes.
    :param bench_cost_fn: An alternative cost function used for analysis and reporting.
    """

    search_fn: typing.Callable
    expand_fn: typing.Callable
    goal_fn: typing.Callable
    heuristic_fn: typing.Callable
    hash_fn: typing.Callable
    bench_cost_fn: typing.Callable = None


@attr.s
class ProblemContext(metaclass=abc.ABCMeta):
    clf = attr.ib()
    target_class = attr.ib()

    @abc.abstractmethod
    def get_graph_search_problem(self):
        pass
