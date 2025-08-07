# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator
from collections.abc import Iterable, Iterator
from typing import Optional

from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._ops import OpOverload


def is_func(node: fx.Node, target) -> bool:
    return node.op == "call_function" and node.target == target


def is_auto_func(node: fx.Node, op: OpOverload) -> bool:
    return is_func(node, auto_functionalized) and node.args[0] == op


# Returns the first specified node with the given op (if it exists)
def find_specified_fn_maybe(nodes: Iterable[fx.Node],
                            op: OpOverload) -> Optional[fx.Node]:
    for node in nodes:
        if node.target == op:
            return node
    return None


# Returns the first specified node with the given op
def find_specified_fn(nodes: Iterable[fx.Node], op: OpOverload) -> fx.Node:
    node = find_specified_fn_maybe(nodes, op)
    assert node is not None, f"Could not find {op} in nodes {nodes}"
    return node


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


# An auto-functionalization-aware utility for finding nodes with a specific op
def find_op_nodes(op: OpOverload, graph: fx.Graph) -> Iterator[fx.Node]:
    if not op._schema.is_mutable:
        yield from graph.find_nodes(op="call_function", target=op)

    for n in graph.find_nodes(op="call_function", target=auto_functionalized):
        if n.args[0] == op:
            yield n


# Asserts that the node only has one user and returns it
# Even if a node has only 1 user, it might share storage with another node,
# which might need to be taken into account.
def get_only_user(node: fx.Node) -> fx.Node:
    assert len(node.users) == 1
    return next(iter(node.users))
