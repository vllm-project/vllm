# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hierarchical graph trace dump for FX graphs.

Reconstructs the module call hierarchy from nn_module_stack metadata
on FX graph nodes, producing an indented, human-readable representation
of the model's forward pass that shows both the module nesting and the
individual operations/kernels within each module.

Usage:
    from vllm.compilation.graph_trace_dump import dump_graph_hierarchy
    print(dump_graph_hierarchy(graph_module))

See https://github.com/vllm-project/vllm/issues/39215
"""

from __future__ import annotations

import re
from collections import OrderedDict

import torch
import torch.fx as fx
from torch._logging._internal import trace_structured

from vllm.logger import init_logger

logger = init_logger(__name__)

# Node ops that don't produce meaningful output lines
_SKIP_OPS = {"placeholder", "output"}


def _get_module_stack(
    node: fx.Node,
) -> list[tuple[str, str]]:
    """
    Extract the module call stack from a node's nn_module_stack metadata.

    Returns a list of (fully_qualified_name, class_name) tuples,
    ordered from outermost to innermost module scope.

    nn_module_stack is an OrderedDict mapping scope keys to
    (fqn: str, module_class: type) pairs. We convert the class to its
    simple name for display.
    """
    nn_module_stack: OrderedDict[str, tuple[str, type]] | None = node.meta.get(
        "nn_module_stack"
    )
    if not nn_module_stack:
        return []

    result = []
    for _key, (fqn, module_cls) in nn_module_stack.items():
        if isinstance(module_cls, type):
            cls_name = module_cls.__name__
        else:
            # Sometimes it's already a string
            cls_name = str(module_cls).rsplit(".", 1)[-1]
        result.append((fqn, cls_name))
    return result


def _format_node(node: fx.Node) -> str:
    """
    Format a single FX node as a human-readable operation string.

    For call_function nodes: target_name(arg0, arg1, ...)
    For get_attr nodes: attr_name
    For other nodes: the node's string representation
    """
    if node.op == "call_function":
        target = node.target
        if isinstance(target, torch._ops.OpOverload):
            target_str = str(target)
        elif hasattr(target, "__module__") and hasattr(target, "__qualname__"):
            module = target.__module__ or ""
            qualname = target.__qualname__
            if module == "_operator" or module == "operator":
                target_str = qualname
            elif module == "torch" or module.startswith("torch."):
                # Simplify internal class dispatch like
                # torch._VariableFunctionsClass.add -> torch.add
                simple_name = qualname.rsplit(".", 1)[-1]
                target_str = f"torch.{simple_name}"
            else:
                target_str = f"{module}.{qualname}" if module else qualname
        else:
            target_str = str(target)

        args_parts = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                args_parts.append(arg.name)
            elif isinstance(arg, (list, tuple)):
                inner = ", ".join(
                    a.name if isinstance(a, fx.Node) else repr(a) for a in arg
                )
                bracket = "[" if isinstance(arg, list) else "("
                close = "]" if isinstance(arg, list) else ")"
                args_parts.append(f"{bracket}{inner}{close}")
            else:
                args_parts.append(repr(arg))
        for k, v in node.kwargs.items():
            if isinstance(v, fx.Node):
                args_parts.append(f"{k}={v.name}")
            else:
                args_parts.append(f"{k}={repr(v)}")

        args_str = ", ".join(args_parts)
        # Truncate very long argument strings
        if len(args_str) > 120:
            args_str = args_str[:117] + "..."

        if node.name and not node.name.startswith("_"):
            return f"{node.name} = {target_str}({args_str})"
        else:
            return f"{target_str}({args_str})"

    elif node.op == "get_attr":
        return f"{node.name} = self.{node.target}"

    return str(node)


def _module_display_name(fqn: str, cls_name: str) -> str:
    """
    Create a display name for a module scope entry.

    Format: ``instance_name: ClassName`` so that multiple instances of the
    same class are easy to distinguish at a glance.

    Examples:
        ("model.layers.0", "DecoderLayer") -> "layers[0]: DecoderLayer"
        ("model.embed_tokens", "Embedding") -> "embed_tokens: Embedding"
        ("layer.linear", "InnerLinear")     -> "linear: InnerLinear"
        ("", "DeepseekV2Model")             -> "DeepseekV2Model"
    """
    if not fqn:
        return cls_name

    # Extract the last component of the fqn for the instance name
    parts = fqn.rsplit(".", 1)
    short_name = parts[-1] if len(parts) > 1 else fqn

    # If the short name is a number, it's an indexed layer (e.g., layers.0)
    # Display as "layers[0]: DecoderLayer".
    # Edge case: if fqn is just "0" (no parent), rsplit produces ["0"]
    # and we fall through to the default "0: ClassName" format.
    if short_name.isdigit():
        parent_parts = fqn.rsplit(".", 2)
        if len(parent_parts) >= 2:
            parent_name = parent_parts[-2]
            return f"{parent_name}[{short_name}]: {cls_name}"

    return f"{short_name}: {cls_name}"


# Regex to parse Python traceback frames:
#   File "/path/to/file.py", line 42, in function_name
_TRACEBACK_FRAME_RE = re.compile(
    r'File "([^"]+)", line (\d+), in (.+)'
)

# Function names from torch internals / nn.Module plumbing to skip
_SKIP_FUNCTION_NAMES = frozenset({
    "forward",
    "_call_impl",
    "_wrapped_call_impl",
    "__call__",
    "_call_with_frames_allowed",
    "<module>",
    "<lambda>",
})


def _get_source_context(node: fx.Node) -> str | None:
    """
    Extract non-module source context from a node's stack_trace metadata.

    Parses the Python traceback recorded by torch.compile/Inductor to find
    free functions and non-module class methods that are NOT already
    represented in nn_module_stack.  Returns a short annotation string
    like ``"my_free_fn"`` or ``None`` if no extra context is available.

    This addresses reviewer Request #2: showing free functions and
    non-module classes in the hierarchy dump.
    """
    stack_trace: str | None = node.meta.get("stack_trace")
    if not stack_trace:
        return None

    # Parse traceback frames
    frames = _TRACEBACK_FRAME_RE.findall(stack_trace)
    if not frames:
        return None

    # Walk frames from innermost to outermost, find the first
    # "interesting" user-defined function
    for filepath, _lineno, func_name in reversed(frames):
        if func_name in _SKIP_FUNCTION_NAMES:
            continue
        # Skip torch/dynamo internals
        if "/torch/" in filepath or "/torch_" in filepath:
            continue
        # Skip vllm compilation internals
        if "/vllm/compilation/" in filepath:
            continue
        # Found a user-defined free function or non-module method
        return func_name

    return None


def dump_graph_hierarchy(
    graph_module: fx.GraphModule,
    *,
    indent_size: int = 2,
) -> str:
    """
    Produce a hierarchical, indented dump of the FX graph that shows
    the module call structure alongside the individual operations.

    Args:
        graph_module: The FX GraphModule to dump.
        indent_size: Number of spaces per indentation level.

    Returns:
        A multi-line string with the hierarchical trace.
    """
    lines: list[str] = []
    # Track the current module stack to detect transitions
    current_stack: list[tuple[str, str]] = []

    for node in graph_module.graph.nodes:
        if node.op in _SKIP_OPS:
            continue

        node_stack = _get_module_stack(node)

        # Find the divergence point between current and new stack
        common_depth = 0
        for i, (cur, new) in enumerate(zip(current_stack, node_stack)):
            if cur == new:
                common_depth = i + 1
            else:
                break

        # Emit new module scope headers for any new levels
        for i in range(common_depth, len(node_stack)):
            fqn, cls_name = node_stack[i]
            indent = " " * (i * indent_size)
            display = _module_display_name(fqn, cls_name)
            lines.append(f"{indent}{display}")

        current_stack = node_stack

        # Emit the operation itself at the deepest module level + 1
        op_depth = len(node_stack)
        indent = " " * (op_depth * indent_size)
        op_str = _format_node(node)

        # Annotate with free function / non-module class context
        source_ctx = _get_source_context(node)
        if source_ctx:
            op_str += f"  # via {source_ctx}"

        lines.append(f"{indent}{op_str}")

    return "\n".join(lines)


def dump_graph_hierarchy_to_file(
    graph_module: fx.GraphModule,
    file_path: str,
    *,
    indent_size: int = 2,
) -> None:
    """
    Dump the hierarchical graph trace to a file.

    Args:
        graph_module: The FX GraphModule to dump.
        file_path: Path to write the dump to.
        indent_size: Number of spaces per indentation level.
    """
    content = dump_graph_hierarchy(graph_module, indent_size=indent_size)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug("Hierarchical graph trace dumped to %s", file_path)


def trace_graph_structured(
    name: str,
    graph_module: fx.GraphModule,
) -> None:
    """
    Emit a structured trace for an FX graph, including both the raw
    ``print_readable`` output and the hierarchical module trace.

    This is the **standard utility** that every graph dump site should use
    so that every dumped graph is also printed in structured (hierarchical)
    form.  It emits two ``trace_structured`` events:

    1. ``graph_dump`` with the raw ``print_readable`` output (name=*name*).
    2. ``graph_dump`` with the hierarchical module trace
       (name=*name* ``_module_trace``).

    Args:
        name: A short identifier for the graph (e.g.
            ``"vllm_graph_before_split"``).  It is used as the ``"name"``
            field in the ``trace_structured`` metadata.
        graph_module: The FX ``GraphModule`` to dump.
    """
    # 1. Raw graph dump (same as the original print_readable output)
    trace_structured(
        "graph_dump",
        metadata_fn=lambda: {"name": name},
        payload_fn=lambda: graph_module.print_readable(print_output=False),
    )

    # 2. Hierarchical module trace
    trace_structured(
        "graph_dump",
        metadata_fn=lambda: {"name": f"{name}_module_trace"},
        payload_fn=lambda: dump_graph_hierarchy(graph_module),
    )
