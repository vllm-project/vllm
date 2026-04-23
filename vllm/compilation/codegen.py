# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code generation for split_gm stitching graph execution.

Generates a plain Python function that replaces the FX GraphModule's
interpreter-based execution of the stitching graph, eliminating
nn.Module.__call__ overhead and __getattr__ dispatch.
"""

import operator
from collections.abc import Callable
from functools import partial
from typing import Any

import torch.fx
from torch._dynamo.utils import dynamo_timed
from torch._logging import trace_structured


@dynamo_timed("vllm.generate_execution_code")
def generate_execution_code(
    split_gm: torch.fx.GraphModule,
) -> tuple[str, list[str]]:
    """Generate Python source code from a split_gm's stitching graph.

    Walks split_gm.graph.nodes and produces a function that calls
    submodules via a __vllm_submods__ list, avoiding FX GraphModule overhead
    and dict lookup cost.

    Args:
        split_gm: The split graph module produced by split_graph().

    Returns:
        A tuple of (code, submod_names) where code is the Python source
        and submod_names is the ordered list of submodule target names
        corresponding to list indices used in the generated code.
    """
    lines: list[str] = []
    param_names: list[str] = []
    submod_names: list[str] = []
    submod_index: dict[str, int] = {}

    # Build node ordering for liveness analysis.
    nodes = list(split_gm.graph.nodes)
    node_order = {node: i for i, node in enumerate(nodes)}

    # For each value-producing node, find the position of its last consumer.
    # If the last consumer is the output node, skip (return handles cleanup).
    # Otherwise, schedule a del after that consumer to free memory early.
    del_after: dict[int, list[str]] = {}  # position -> names to delete
    for node in nodes:
        if node.op == "output":
            continue
        users = list(node.users.keys())
        if not users:
            continue
        last_user = max(users, key=lambda u: node_order[u])
        if last_user.op == "output":
            continue
        del_after.setdefault(node_order[last_user], []).append(node.name)

    for i, node in enumerate(nodes):
        if node.op == "placeholder":
            param_names.append(node.name)

        elif node.op == "call_module":
            target = node.target
            if target not in submod_index:
                submod_index[target] = len(submod_names)
                submod_names.append(target)
            idx = submod_index[target]
            args_str = ", ".join(_node_ref(a) for a in node.args)
            kwargs_str = ", ".join(
                f"{k}={_node_ref(v)}" for k, v in node.kwargs.items()
            )
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))
            lines.append(f"    {node.name} = __vllm_submods__[{idx}]({all_args})")

        elif node.op == "call_function" and node.target is operator.getitem:
            source = _node_ref(node.args[0])
            index = node.args[1]
            assert isinstance(index, int)
            lines.append(f"    {node.name} = {source}[{index}]")

        elif node.op == "output":
            assert len(node.args) == 1
            ret = _node_ref(node.args[0])
            lines.append(f"    return {ret}")

        else:
            raise RuntimeError(f"Unsupported node from codegen: {node.format_node()}")

        # Emit del for variables whose last use was this node.
        if i in del_after:
            names = sorted(del_after[i])
            lines.append(f"    del {', '.join(names)}")

    assert len(param_names) > 0
    params = ", ".join(param_names)
    header = f"def execution_fn({params}, *, __vllm_submods__):"
    return "import torch\n" + "\n".join([header] + lines) + "\n", submod_names


@dynamo_timed("vllm.compile_execution_fn")
def compile_execution_fn(
    code: str,
    submod_callables: dict[str, Callable[..., Any]],
    submod_names: list[str],
) -> Callable[..., Any]:
    """Compile execution code and bind submodule callables.

    Args:
        code: Python source from generate_execution_code().
        submod_callables: Mapping of submodule names to their callables.
        submod_names: Ordered list of submodule names matching the indices
            used in the generated code.

    Returns:
        A callable that executes the stitching logic.
    """
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "vllm_execution_code",
            "encoding": "string",
        },
        payload_fn=lambda: code,
    )
    namespace: dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102
    fn = namespace["execution_fn"]
    # Use .forward() directly to avoid nn.Module.__call__ overhead.
    submods_list = [
        c.forward if isinstance(c, torch.fx.GraphModule) else c
        for c in (submod_callables[name] for name in submod_names)
    ]
    return partial(fn, __vllm_submods__=submods_list)


def _node_ref(arg: Any) -> str:
    """Convert an FX node argument to a source code reference recursively."""
    if isinstance(arg, torch.fx.Node):
        return arg.name
    if isinstance(arg, list):
        return f"[{', '.join(_node_ref(x) for x in arg)}]"
    if isinstance(arg, tuple):
        items = ", ".join(_node_ref(x) for x in arg)
        return f"({items},)" if len(arg) == 1 else f"({items})"
    if isinstance(arg, dict):
        return (
            "{"
            + ", ".join(f"{_node_ref(k)}: {_node_ref(v)}" for k, v in arg.items())
            + "}"
        )
    return repr(arg)
