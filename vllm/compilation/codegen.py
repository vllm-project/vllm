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
from torch.fx.node import _get_qualified_name


def generate_execution_code_with_name(
    split_gm: torch.fx.GraphModule,
    fn_name: str,
    with_submod: bool,
) -> tuple[str, list[str]]:
    lines: list[str] = []
    param_names: list[str] = []
    submod_names: list[str] = []
    submod_index: dict[str, int] = {}

    # Build node ordering for liveness analysis.
    nodes = list(split_gm.graph.nodes)
    node_order = {node: i for i, node in enumerate(nodes)}
    inlined_submods: list[str] = []

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
            if not with_submod:
                raise RuntimeError(
                    f"call_module is not allowed for codegen target {target}."
                )
            if target not in submod_index:
                submod_index[target] = len(submod_names)
                submod_names.append(target)
            idx = submod_index[target]
            args_str = ", ".join(_node_ref(a) for a in node.args)
            kwargs_str = ", ".join(
                f"{k}={_node_ref(v)}" for k, v in node.kwargs.items()
            )
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))
            submod = getattr(split_gm, target)
            if isinstance(submod, torch.fx.GraphModule):
                callable_name = f"__vllm_inlined_submods__{idx}"
                inlined_code, _ = generate_execution_code_with_name(
                    submod, callable_name, with_submod=False
                )
                inlined_submods.append(inlined_code)
            else:
                callable_name = f"__vllm_submods__[{idx}]"
            lines.append(f"    {node.name} = {callable_name}({all_args})")

        elif node.op == "call_function":
            if node.target is operator.getitem:
                source = _node_ref(node.args[0])
                index = node.args[1]
                assert isinstance(index, int)
                lines.append(f"    {node.name} = {source}[{index}]")
            else:
                args_str = ", ".join(_node_ref(a) for a in node.args)
                kwargs_str = ", ".join(
                    f"{k}={_node_ref(v)}" for k, v in node.kwargs.items()
                )
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                lines.append(
                    f"    {node.name} = {_get_qualified_name(node.target)}({all_args})"
                )

        elif node.op == "output":
            assert len(node.args) == 1
            ret = _node_ref(node.args[0])
            lines.append(f"    return {ret}")

        else:
            raise RuntimeError(f"Unsupported node from codegen: {node.format_node()}")

        # Emit del for variables whose last use was this node.
        if i in del_after and i < len(nodes) - 2:
            names = sorted(del_after[i])
            lines.append(f"    del {', '.join(names)}")

    assert len(param_names) > 0
    params = ", ".join(param_names)
    header = (
        f"\ndef {fn_name}({params}{', *, __vllm_submods__' if with_submod else ''}):"
    )
    return "".join(inlined_submods) + "\n".join([header] + lines) + "\n", submod_names


@dynamo_timed("vllm.generate_execution_code")
def generate_execution_code(
    split_gm: torch.fx.GraphModule,
) -> tuple[str, list[str]]:
    """Generate Python source code from a split_gm's stitching graph.

    Walks split_gm.graph.nodes and produces a function that calls
    submodules via a __vllm_submods__ list, avoiding FX GraphModule overhead
    and dict lookup cost.

    If a submodule is a plain torch.fx.GraphModule, it is inlined directly
    in the generated code and we do not need to serialize it in the artifact.

    Args:
        split_gm: The split graph module produced by split_graph().

    Returns:
        A tuple of (code, submod_names) where code is the Python source
        and submod_names is the ordered list of submodule target names
        corresponding to list indices used in the generated code.
    """

    code, submod_names = generate_execution_code_with_name(
        split_gm, "execution_fn", with_submod=True
    )
    return "import torch\nimport operator\n" + code, submod_names


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
    # Using .get() is intentional here because only piecewise backend will
    # be stored in submod_callables. The other submodules are inlined and
    # we don't need to bind them to the execution function. Instead, we
    # should use None as placeholder to ensure the list indices are preserved
    # for better debuggability.
    submods_list = [submod_callables.get(name) for name in submod_names]
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
