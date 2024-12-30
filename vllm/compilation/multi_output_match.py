import abc
import operator
from abc import abstractmethod
from typing import Iterable, List, Tuple

from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor import pattern_matcher as pm
from torch._ops import OpOverload
from torch.fx import Node

from vllm.compilation.fx_utils import find_auto_fn


class MultiOutputMatch(abc.ABC):
    """
    This class provides utilities to process multi-output matches and
    manually insert replacements.

    This is necessary because the automatic replacement for multi-output
    matches is broken: https://github.com/pytorch/pytorch/issues/137280
    """

    def __init__(self, match: pm.Match):
        self.match = match

    @abstractmethod
    def process(self):
        """
        Process a multi-output match and manually insert the replacement.

        This method should:
        1. Insert the replacement nodes after the last node in the match.
        2. Rebind the users of nodes in the match to use the new nodes.
        3. Set meta["val"] for de-functionalization.

        The result of an auto-functionalized node is a tuple of tensors.
        The first element is the return value of the function, usually None.
        The remaining elements are the mutated args of the function.

        All auto-functionalized nodes must contain a proper meta["val"],
        as it is used by de-functionalization. meta["val"] has to contain the
        value of the node (tuple of tensors) that would be returned by the
        functionalized node during tracing.

        Existing nodes in the graph all have this property set, but we have
        to set it manually for new nodes we insert.

        Example:
        # op schema: foo(a: Tensor!, b: Tensor, c: Tensor!) -> None
        at = auto_functionalized(torch.ops._C.foo.default, a, b, c)
        # at.meta["val"] = (None, a, c)
        """
        raise NotImplementedError

    @property
    def nodes(self) -> List[fx.Node]:
        return self.match.nodes

    @property
    def graph(self) -> fx.Graph:
        return self.match.graph

    def find_auto_fn(self, op) -> fx.Node:
        """
        Find the first auto_functionalized node with the given op in the match.
        """
        return find_auto_fn(self.nodes, op)

    def inserting_after_match(self):
        """
        Insert nodes after the last node in the match.
        This is done to avoid use-before-definition errors after inserting
        replacement nodes.
        """

        # match.nodes is not guaranteed to be sorted.
        # Find the last node in the match.
        for last_node_in_match in reversed(self.graph.nodes):
            if last_node_in_match in self.match.nodes:
                break
        else:
            raise ValueError("No nodes in graph")

        return self.graph.inserting_after(last_node_in_match)

    def insert_getitems(self, tuple_node: fx.Node,
                        indices: Iterable[int]) -> Tuple[fx.Node, ...]:
        """
        Insert operator.getitem nodes to extract elements from a tuple node.

        :param tuple_node: The tuple node to extract elements from.
        :param indices: The indices of the elements to extract.
        :return: Tuple of the new getitem nodes, corresponding to the indices.
        """
        with self.graph.inserting_after(tuple_node):
            return tuple(
                self.graph.call_function(operator.getitem, (tuple_node, idx))
                for idx in indices)

    def insert_auto_fn(self, op: OpOverload, kwargs) -> Node:
        """
        Insert an auto_functionalized node with the given op and kwargs.
        """
        return self.graph.call_function(auto_functionalized, (op, ),
                                        kwargs=kwargs)
