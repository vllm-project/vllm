# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Iterator

from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx.node import Target


def is_func(node: fx.Node, target: Target) -> bool:
    return bool(node.op == "call_function" and node.target == target)


# Returns the first auto_functionalized node with the given op (if it exists)
def find_auto_fn_maybe(nodes: Iterable[fx.Node], op: OpOverload) -> fx.Node | None:
    for node in nodes:
        if is_func(node, auto_functionalized) and node.args[0] == op:  # noqa
            return node
    return None


# Returns the first auto_functionalized node with the given op
def find_auto_fn(nodes: Iterable[fx.Node], op: OpOverload) -> fx.Node:
    node = find_auto_fn_maybe(nodes, op)
    assert node is not None, f"Could not find {op} in nodes {nodes}"
    return node


# An auto-functionalization-aware utility for finding nodes with a specific op
# Also handles op overload packets and finds all overloads
def find_op_nodes(
    op: OpOverload | OpOverloadPacket, graph: fx.Graph
) -> Iterator[fx.Node]:
    if isinstance(op, OpOverloadPacket):
        for overload in op.overloads():
            overload_op = getattr(op, overload)
            yield from find_op_nodes(overload_op, graph)
        return

    assert isinstance(op, OpOverload)

    yield from graph.find_nodes(op="call_function", target=op)

    for n in graph.find_nodes(op="call_function", target=auto_functionalized):
        if n.args[0] == op:
            yield n
