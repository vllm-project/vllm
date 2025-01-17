import operator
from typing import Iterable, Optional

from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import Match
from torch._ops import OpOverload


def is_func(node: fx.Node, target) -> bool:
    return node.op == "call_function" and node.target == target


def find_fn(nodes: Iterable[fx.Node], op) -> Optional[fx.Node]:
    for node in nodes:
        if node.op == "call_function" and node.target == op:
            return node
    return None


def find_op(nodes: Iterable[fx.Node], op: str) -> Optional[fx.Node]:
    for node in nodes:
        if node.op == op:
            return node
    return None


# Returns the first auto_functionalized node with the given op (if it exists)
def find_auto_fn_maybe(nodes: Iterable[fx.Node],
                       op: OpOverload) -> Optional[fx.Node]:
    for node in nodes:
        if is_func(node, auto_functionalized) and node.args[0] == op:  # noqa
            return node
    return None


# Returns the first auto_functionalized node with the given op
def find_auto_fn(nodes: Iterable[fx.Node], op: OpOverload) -> fx.Node:
    node = find_auto_fn_maybe(nodes, op)
    assert node is not None, f"Could not find {op} in nodes {nodes}"
    return node


# Returns the getitem node that extracts the idx-th element from node
# (if it exists)
def find_getitem_maybe(node: fx.Node, idx: int) -> Optional[fx.Node]:
    for user in node.users:
        if is_func(user, operator.getitem) and user.args[1] == idx:
            return user
    return None


# Returns the getitem node that extracts the idx-th element from node
def find_getitem(node: fx.Node, idx: int) -> fx.Node:
    ret = find_getitem_maybe(node, idx)
    assert ret is not None, f"Could not find getitem {idx} in node {node}"
    return ret


def last_node_in_match(match: Match) -> fx.Node:
    if len(match.nodes) > 0:
        graph = match.nodes[0].graph
        for n in reversed(graph.nodes):
            if n in reversed(match.nodes):
                return n
    raise ValueError("No nodes in graph")
