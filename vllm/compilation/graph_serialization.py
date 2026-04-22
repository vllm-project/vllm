# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast, lean GraphModule serialization.

GraphPickler is general-purpose but expensive.  For the vLLM compilation
cache we only need the graph *topology* (node ops, targets, connectivity)
and the submodule hierarchy -- no node metadata such as ``example_value``,
``source_fn_stack``, or ``nn_module_stack``.

The format is a compact JSON blob (C-level json encoder/decoder) that can
be round-tripped much faster than GraphPickler for the graphs we care
about.
"""

import importlib
import json
from typing import Any

import torch
import torch.fx as fx

# Sentinel keys used in the JSON representation.
_NODE_REF = "__r"
_TUPLE_TAG = "__t"


# ---------------------------------------------------------------------------
# Argument serialization
# ---------------------------------------------------------------------------


def _serialize_arg(arg: Any) -> Any:
    """Encode an fx.Node argument for JSON serialisation."""
    if isinstance(arg, fx.Node):
        return {_NODE_REF: arg.name}
    if isinstance(arg, tuple):
        return {_TUPLE_TAG: [_serialize_arg(a) for a in arg]}
    if isinstance(arg, list):
        return [_serialize_arg(a) for a in arg]
    if isinstance(arg, dict):
        return {k: _serialize_arg(v) for k, v in arg.items()}
    if isinstance(arg, slice):
        return {
            "__slice": [
                _serialize_arg(arg.start),
                _serialize_arg(arg.stop),
                _serialize_arg(arg.step),
            ]
        }
    if isinstance(arg, torch.dtype):
        return {"__dtype": str(arg)}
    if isinstance(arg, torch.device):
        return {"__device": str(arg)}
    if isinstance(arg, torch.layout):
        return {"__layout": str(arg)}
    if isinstance(arg, torch.memory_format):
        return {"__memory_format": str(arg)}
    if arg is ...:
        return {"__ellipsis": True}
    # JSON-primitive types (int, float, str, bool, None) pass through.
    if arg is None or isinstance(arg, (int, float, str, bool)):
        return arg
    raise TypeError(f"Unsupported argument type for graph serialization: {type(arg)}")


def _deserialize_arg(arg: Any, node_map: dict[str, fx.Node]) -> Any:
    """Decode an argument produced by ``_serialize_arg``."""
    if isinstance(arg, dict):
        if _NODE_REF in arg:
            return node_map[arg[_NODE_REF]]
        if _TUPLE_TAG in arg:
            return tuple(_deserialize_arg(a, node_map) for a in arg[_TUPLE_TAG])
        if "__slice" in arg:
            s = arg["__slice"]
            return slice(
                _deserialize_arg(s[0], node_map),
                _deserialize_arg(s[1], node_map),
                _deserialize_arg(s[2], node_map),
            )
        if "__dtype" in arg:
            return getattr(torch, arg["__dtype"].split(".")[-1])
        if "__device" in arg:
            return torch.device(arg["__device"])
        if "__layout" in arg:
            return getattr(torch, arg["__layout"].split(".")[-1])
        if "__memory_format" in arg:
            return getattr(torch, arg["__memory_format"].split(".")[-1])
        if "__ellipsis" in arg:
            return ...
        return {k: _deserialize_arg(v, node_map) for k, v in arg.items()}
    if isinstance(arg, list):
        return [_deserialize_arg(a, node_map) for a in arg]
    return arg


# ---------------------------------------------------------------------------
# Target serialization
# ---------------------------------------------------------------------------


def _target_to_str(op: str, target: Any) -> str:
    """Convert a node target to a string for serialisation."""
    if op in ("placeholder", "call_module", "call_method", "get_attr", "output"):
        return str(target)
    if op == "call_function":
        from torch.fx.node import _get_qualified_name

        return _get_qualified_name(target)
    raise ValueError(f"Unknown fx op: {op}")


def _str_to_target(op: str, target_str: str) -> Any:
    """Resolve a serialised target string back to its runtime object."""
    if op in ("placeholder", "call_module", "call_method", "get_attr", "output"):
        return target_str
    if op == "call_function":
        # Resolve the qualified name to the actual callable.
        parts = target_str.rsplit(".", 1)
        if len(parts) == 1:
            # Builtins (e.g. ``getattr``).
            import builtins

            return getattr(builtins, target_str)
        module_name, attr_name = parts
        # torch.ops.* is a dynamic namespace (not a real module), so
        # resolve it via attribute traversal instead of importlib.
        if module_name.startswith("torch.ops."):
            obj: Any = torch.ops
            for part in module_name.split(".")[2:]:
                obj = getattr(obj, part)
            return getattr(obj, attr_name)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise ValueError(f"Unknown fx op: {op}")


# ---------------------------------------------------------------------------
# Graph / GraphModule serialization
# ---------------------------------------------------------------------------


def _serialize_graph(graph: fx.Graph) -> list[list[Any]]:
    """Serialise an ``fx.Graph`` as a list of node descriptors."""
    nodes: list[list[Any]] = []
    for node in graph.nodes:
        nodes.append(
            [
                node.op,
                node.name,
                _target_to_str(node.op, node.target),
                _serialize_arg(node.args),
                _serialize_arg(node.kwargs),
            ]
        )
    return nodes


def _deserialize_graph(
    node_list: list[list[Any]],
) -> tuple[fx.Graph, dict[str, fx.Node]]:
    """Reconstruct an ``fx.Graph`` from the compact node list."""
    graph = fx.Graph()
    node_map: dict[str, fx.Node] = {}
    for op, name, target_str, raw_args, raw_kwargs in node_list:
        target = _str_to_target(op, target_str)
        args = _deserialize_arg(raw_args, node_map)
        # _deserialize_arg returns a list for the top-level args; create_node
        # expects a tuple.
        if isinstance(args, list):
            args = tuple(args)
        kwargs = _deserialize_arg(raw_kwargs, node_map)
        node = graph.create_node(op, target, args, kwargs, name=name)
        node_map[name] = node
    return graph, node_map


def _serialize_gm(gm: fx.GraphModule) -> dict[str, Any]:
    """Recursively serialise a ``GraphModule`` (graph + submodules)."""
    submodules: dict[str, dict[str, Any]] = {}
    for child_name, child in gm.named_children():
        if isinstance(child, fx.GraphModule):
            submodules[child_name] = _serialize_gm(child)
    return {
        "nodes": _serialize_graph(gm.graph),
        "submodules": submodules,
    }


def _deserialize_gm(data: dict[str, Any]) -> fx.GraphModule:
    """Reconstruct a ``GraphModule`` from the dict produced by
    ``_serialize_gm``."""
    # Reconstruct child submodules first so they're available as attributes.
    root = torch.nn.Module()
    for child_name, child_data in data["submodules"].items():
        root.add_module(child_name, _deserialize_gm(child_data))

    graph, _ = _deserialize_graph(data["nodes"])
    gm = fx.GraphModule(root, graph)
    return gm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serialize_graph_structure(gm: fx.GraphModule) -> bytes:
    """Serialise the topology of *gm* to bytes (fast, no metadata).

    Only the graph structure (node ops/targets/connectivity) and the
    submodule hierarchy are preserved.  Node metadata such as
    ``example_value`` is **not** included.

    This is intended as a drop-in replacement for
    ``GraphPickler.dumps(gm, ...)`` in the vLLM compilation cache path
    where the full generality of ``GraphPickler`` is not needed.
    """
    return json.dumps(_serialize_gm(gm), separators=(",", ":")).encode()


def deserialize_graph_structure(data: bytes) -> fx.GraphModule:
    """Reconstruct a ``GraphModule`` from bytes produced by
    :func:`serialize_graph_structure`.

    The returned ``GraphModule`` has ``recompile()`` already called.
    """
    obj = json.loads(data)
    gm = _deserialize_gm(obj)
    gm.recompile()
    return gm
