# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ..fx_utils import is_func

FLASHINFER_BMM_FP8_MIN_M = 64
VIEW_LIKE_OPS = (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.default,
)
LAYOUT_PRESERVING_OPS = (
    torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
)


def _get_node_arg(node: fx.Node, name: str, index: int) -> object:
    return node.kwargs.get(name, node.args[index] if len(node.args) > index else None)


def _is_view_like(node: fx.Node) -> bool:
    return any(is_func(node, op) for op in VIEW_LIKE_OPS)


def _is_passthrough(node: fx.Node) -> bool:
    return _is_view_like(node) or any(is_func(node, op) for op in LAYOUT_PRESERVING_OPS)


def _strip_view_like(node: fx.Node) -> fx.Node:
    while _is_view_like(node):
        parent = node.args[0]
        if not isinstance(parent, fx.Node):
            break
        node = parent
    return node


def _walk_reachable_users(start_nodes: list[fx.Node]) -> list[fx.Node]:
    worklist = list(start_nodes)
    visited: set[fx.Node] = set()
    reachable: list[fx.Node] = []

    while worklist:
        user = worklist.pop()
        if user in visited:
            continue
        visited.add(user)
        reachable.append(user)

        if _is_passthrough(user):
            worklist.extend(user.users)

    return reachable


def _walk_reachable_users_with_slice_scatter_state(
    start_nodes: list[fx.Node],
) -> list[tuple[fx.Node, bool]]:
    worklist: list[tuple[fx.Node, bool]] = [(user, False) for user in start_nodes]
    visited: set[tuple[fx.Node, bool]] = set()
    reachable: list[tuple[fx.Node, bool]] = []

    while worklist:
        user, saw_slice_scatter = worklist.pop()
        state = (user, saw_slice_scatter)
        if state in visited:
            continue
        visited.add(state)
        reachable.append(state)

        if _is_passthrough(user):
            worklist.extend((child, saw_slice_scatter) for child in user.users)
            continue

        if is_func(user, torch.ops.aten.slice_scatter.default):
            worklist.extend((child, True) for child in user.users)

    return reachable


def _collect_first_passthrough_matches(
    start_nodes: list[fx.Node],
    predicate,
) -> list[fx.Node]:
    worklist = list(start_nodes)
    visited: set[fx.Node] = set()
    matches: list[fx.Node] = []

    while worklist:
        user = worklist.pop()
        if user in visited:
            continue
        visited.add(user)

        if not _is_passthrough(user):
            continue

        if predicate(user):
            matches.append(user)
            continue

        worklist.extend(user.users)

    return matches


def _node_shape(node: fx.Node) -> list[object] | None:
    val = node.meta.get("val")
    if hasattr(val, "shape"):
        return list(val.shape)
    return None


def _node_first_dim(node: fx.Node) -> object | None:
    shape = _node_shape(node)
    if shape:
        return shape[0]
    return None


def _node_ndim(node: fx.Node) -> int | None:
    shape = _node_shape(node)
    if shape is None:
        return None
    return len(shape)


def _dim_is_statically_lt(dim: int | torch.SymInt, threshold: int) -> bool:
    if isinstance(dim, int):
        return dim < threshold
    try:
        return bool(statically_known_true(dim < threshold))
    except Exception:
        return False


def _passes_min_m(node: fx.Node) -> bool:
    gemm_m = _node_first_dim(node)
    if gemm_m is None or not isinstance(gemm_m, int | torch.SymInt):
        return True
    return not _dim_is_statically_lt(gemm_m, FLASHINFER_BMM_FP8_MIN_M)


def _copy_replacement_meta(src: fx.Node, dst: fx.Node) -> None:
    # Inductor may stash the original op's eager inputs in metadata. If we
    # carry those over to a replacement op with a different schema, later
    # schema normalization can fail during lowering.
    dst.meta = {
        key: value for key, value in src.meta.items() if key != "eager_input_vals"
    }


def _unwrap_bmm_fp8_arg_to_2d(arg: object) -> fx.Node | None:
    if not isinstance(arg, fx.Node):
        return None

    node = _strip_view_like(arg)
    if is_func(node, torch.ops.aten.unsqueeze.default):
        dim = _get_node_arg(node, "dim", 1)
        if dim != 0:
            return None
        src = _get_node_arg(node, "self", 0)
        if not isinstance(src, fx.Node):
            return None
        src = _strip_view_like(src)
        ndim = _node_ndim(src)
        if ndim is not None and ndim != 2:
            return None
        return src

    ndim = _node_ndim(node)
    if ndim is not None and ndim != 2:
        return None
    return node


def _parse_collective_op(
    node: fx.Node,
    op,
) -> tuple[fx.Node, object, object, object] | None:
    if not is_func(node, op):
        return None

    input_node = _get_node_arg(node, "tensor", 0)
    dim = _get_node_arg(node, "dim", 1)
    world_size = _get_node_arg(node, "world_size", 2)
    group_name = _get_node_arg(node, "group_name", 3)
    if not isinstance(input_node, fx.Node):
        return None
    return input_node, dim, world_size, group_name


def _parse_reduce_scatter(
    node: fx.Node,
) -> tuple[fx.Node, object, object, object] | None:
    return _parse_collective_op(node, torch.ops.vllm.reduce_scatter.default)


def _parse_all_gather(
    node: fx.Node,
) -> tuple[fx.Node, object, object, object] | None:
    return _parse_collective_op(node, torch.ops.vllm.all_gather.default)


def _parse_collective_group_name(
    dim: object,
    world_size: object,
    group_name: object,
) -> str | None:
    if dim != 0 or not isinstance(world_size, int) or not isinstance(group_name, str):
        return None
    return group_name


def _parse_bmm_fp8(
    node: fx.Node,
) -> tuple[fx.Node, fx.Node, object, object, object, object] | None:
    if not is_func(node, torch.ops.vllm.bmm_fp8.default):
        return None

    a = _get_node_arg(node, "A", 0)
    b = _get_node_arg(node, "B", 1)
    a_scale = _get_node_arg(node, "A_scale", 2)
    b_scale = _get_node_arg(node, "B_scale", 3)
    out_dtype = _get_node_arg(node, "dtype", 4)
    backend = _get_node_arg(node, "backend", 5)

    a_2d = _unwrap_bmm_fp8_arg_to_2d(a)
    b_2d = _unwrap_bmm_fp8_arg_to_2d(b)
    if a_2d is None or b_2d is None:
        return None
    return a_2d, b_2d, a_scale, b_scale, out_dtype, backend


@dataclass
class _FP8CollectiveGemmMatch:
    replace_nodes: list[fx.Node]
    a_2d: fx.Node
    b_2d: fx.Node
    a_scale: object
    b_scale: object
    out_dtype: object
    group_name: str


def _find_reduce_scatter_user(
    bmm_node: fx.Node,
) -> tuple[fx.Node, object, object, object] | None:
    rs_matches: list[tuple[fx.Node, object, object, object]] = []

    for user in _walk_reachable_users(list(bmm_node.users)):
        parsed_rs = _parse_reduce_scatter(user)
        if parsed_rs is not None:
            _, dim, world_size, group_name = parsed_rs
            rs_matches.append((user, dim, world_size, group_name))

    if len(rs_matches) == 1:
        return rs_matches[0]
    return None


def _is_qkv_split(node: fx.Node) -> bool:
    if not is_func(node, torch.ops.aten.split_with_sizes.default):
        return False

    split_sizes = _get_node_arg(node, "split_sizes", 1)
    dim = _get_node_arg(node, "dim", 2)
    return (
        isinstance(split_sizes, (list, tuple))
        and len(split_sizes) == 3
        and dim
        in (
            -1,
            1,
        )
    )


def _classify_qkv_branch(node: fx.Node) -> str | None:
    if any(_is_qkv_split(user) for user in _walk_reachable_users(list(node.users))):
        return "direct"
    if any(
        saw_slice_scatter and _is_qkv_split(user)
        for user, saw_slice_scatter in _walk_reachable_users_with_slice_scatter_state(
            list(node.users)
        )
    ):
        return "rotary"
    return None


def _find_ag_qkv_replace_targets(bmm_node: fx.Node) -> list[fx.Node] | None:
    replace_targets = [
        user
        for user in bmm_node.users
        if _is_passthrough(user) and _node_ndim(user) == 2
    ]
    if len(replace_targets) != 2:
        return None

    if _node_shape(replace_targets[0]) != _node_shape(replace_targets[1]):
        return None

    branch_kinds = [_classify_qkv_branch(node) for node in replace_targets]
    if set(branch_kinds) == {"direct", "rotary"}:
        return replace_targets
    return None


def _find_ag_single_replace_target(bmm_node: fx.Node) -> fx.Node | None:
    replace_targets = _collect_first_passthrough_matches(
        list(bmm_node.users),
        lambda node: _node_ndim(node) == 2,
    )

    if not replace_targets and _node_ndim(bmm_node) == 2:
        replace_targets = [bmm_node]

    if len(replace_targets) != 1:
        return None
    return replace_targets[0]


def _find_ag_replace_targets(bmm_node: fx.Node) -> list[fx.Node] | None:
    qkv_targets = _find_ag_qkv_replace_targets(bmm_node)
    if qkv_targets is not None:
        return qkv_targets

    target = _find_ag_single_replace_target(bmm_node)
    if target is None:
        return None
    return [target]


def _first_node_in_graph(graph: fx.Graph, nodes: list[fx.Node]) -> fx.Node:
    order = {node: index for index, node in enumerate(graph.nodes)}
    return min(nodes, key=order.__getitem__)


def match_bmm_rs(bmm_node: fx.Node) -> _FP8CollectiveGemmMatch | None:
    parsed_bmm = _parse_bmm_fp8(bmm_node)
    if parsed_bmm is None:
        return None

    a_2d, b_2d, a_scale, b_scale, out_dtype, _backend = parsed_bmm
    rs_match = _find_reduce_scatter_user(bmm_node)
    if rs_match is None:
        return None

    rs_node, dim, world_size, group_name = rs_match
    parsed_group_name = _parse_collective_group_name(dim, world_size, group_name)
    if parsed_group_name is None:
        return None

    return _FP8CollectiveGemmMatch(
        replace_nodes=[rs_node],
        a_2d=a_2d,
        b_2d=b_2d,
        a_scale=a_scale,
        b_scale=b_scale,
        out_dtype=out_dtype,
        group_name=parsed_group_name,
    )


def match_ag_bmm(bmm_node: fx.Node) -> _FP8CollectiveGemmMatch | None:
    parsed_bmm = _parse_bmm_fp8(bmm_node)
    if parsed_bmm is None:
        return None

    a_2d, b_2d, a_scale, b_scale, out_dtype, _backend = parsed_bmm
    ag_node = _strip_view_like(a_2d)
    parsed_ag = _parse_all_gather(ag_node)
    if parsed_ag is None:
        return None

    ag_input, dim, world_size, group_name = parsed_ag
    parsed_group_name = _parse_collective_group_name(dim, world_size, group_name)
    if parsed_group_name is None:
        return None

    targets = _find_ag_replace_targets(bmm_node)
    if targets is None:
        return None

    return _FP8CollectiveGemmMatch(
        replace_nodes=targets,
        a_2d=ag_input,
        b_2d=b_2d,
        a_scale=a_scale,
        b_scale=b_scale,
        out_dtype=out_dtype,
        group_name=parsed_group_name,
    )


def lower_bmm_rs(graph: fx.Graph, match: _FP8CollectiveGemmMatch) -> None:
    replace_node = match.replace_nodes[0]
    with graph.inserting_before(replace_node):
        replacement = graph.call_function(
            torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default,
            args=(
                match.a_2d,
                match.b_2d,
                match.a_scale,
                match.b_scale,
                "sum",
                0,
                0,
                match.group_name,
                match.out_dtype,
            ),
        )

    _copy_replacement_meta(replace_node, replacement)
    replace_node.replace_all_uses_with(replacement)
    graph.erase_node(replace_node)


def lower_ag_bmm(graph: fx.Graph, match: _FP8CollectiveGemmMatch) -> None:
    replace_node = _first_node_in_graph(graph, match.replace_nodes)
    with graph.inserting_before(replace_node):
        fused = graph.call_function(
            torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default,
            args=(
                match.a_2d,
                match.b_2d,
                match.a_scale,
                match.b_scale,
                0,
                match.group_name,
                match.out_dtype,
            ),
        )
        replacement = graph.call_function(operator.getitem, args=(fused, 1))

    _copy_replacement_meta(replace_node, replacement)
    for node in match.replace_nodes:
        node.replace_all_uses_with(replacement)
    for node in match.replace_nodes:
        graph.erase_node(node)


def rewrite_flashinfer_bmm_fp8_collective_fusion(graph: fx.Graph) -> int:
    replaced = 0
    for node in list(graph.nodes):
        if not is_func(node, torch.ops.vllm.bmm_fp8.default):
            continue

        rs_match = match_bmm_rs(node)
        if rs_match is not None and _passes_min_m(rs_match.replace_nodes[0]):
            lower_bmm_rs(graph, rs_match)
            replaced += 1
            continue
        ag_match = match_ag_bmm(node)
        if ag_match is None or not _passes_min_m(ag_match.replace_nodes[0]):
            continue

        lower_ag_bmm(graph, ag_match)
        replaced += 1

    if replaced:
        graph.eliminate_dead_code()
        graph.lint()
    return replaced
