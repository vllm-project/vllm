# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal

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


def _node_ndim(node: fx.Node) -> int | None:
    val = node.meta.get("val")
    if hasattr(val, "dim"):
        return int(val.dim())
    return None


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


def _dim_is_statically_lt(dim: int | torch.SymInt, threshold: int) -> bool:
    if isinstance(dim, int):
        return dim < threshold
    try:
        return bool(statically_known_true(dim < threshold))
    except Exception:
        return False


def _unwrap_bmm_fp8_arg_to_2d(arg: object) -> fx.Node | None:
    if not isinstance(arg, fx.Node):
        return None

    node = _strip_view_like(arg)
    if is_func(node, torch.ops.aten.unsqueeze.default):
        dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
        if dim != 0:
            return None
        src = node.args[0]
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


def _parse_reduce_scatter(
    node: fx.Node,
) -> tuple[fx.Node, object, object, object] | None:
    if not is_func(node, torch.ops.vllm.reduce_scatter.default):
        return None

    rs_input = node.kwargs.get("tensor", node.args[0] if len(node.args) > 0 else None)
    dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
    world_size = node.kwargs.get(
        "world_size", node.args[2] if len(node.args) > 2 else None
    )
    group_name = node.kwargs.get(
        "group_name", node.args[3] if len(node.args) > 3 else None
    )
    if not isinstance(rs_input, fx.Node):
        return None
    return rs_input, dim, world_size, group_name


def _parse_all_gather(
    node: fx.Node,
) -> tuple[fx.Node, object, object, object] | None:
    if not is_func(node, torch.ops.vllm.all_gather.default):
        return None

    ag_input = node.kwargs.get("tensor", node.args[0] if len(node.args) > 0 else None)
    dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
    world_size = node.kwargs.get(
        "world_size", node.args[2] if len(node.args) > 2 else None
    )
    group_name = node.kwargs.get(
        "group_name", node.args[3] if len(node.args) > 3 else None
    )
    if not isinstance(ag_input, fx.Node):
        return None
    return ag_input, dim, world_size, group_name


def _parse_bmm_fp8(
    node: fx.Node,
) -> tuple[fx.Node, fx.Node, object, object, object] | None:
    if not is_func(node, torch.ops.vllm.bmm_fp8.default):
        return None

    a = node.kwargs.get("A", node.args[0] if len(node.args) > 0 else None)
    b = node.kwargs.get("B", node.args[1] if len(node.args) > 1 else None)
    a_scale = node.kwargs.get("A_scale", node.args[2] if len(node.args) > 2 else None)
    b_scale = node.kwargs.get("B_scale", node.args[3] if len(node.args) > 3 else None)
    out_dtype = node.kwargs.get("dtype", node.args[4] if len(node.args) > 4 else None)

    a_2d = _unwrap_bmm_fp8_arg_to_2d(a)
    b_2d = _unwrap_bmm_fp8_arg_to_2d(b)
    if a_2d is None or b_2d is None:
        return None

    return a_2d, b_2d, a_scale, b_scale, out_dtype


@dataclass
class _FlashInferCollectiveGemmMatch:
    kind: Literal["ag_bmm", "bmm_rs"]
    replace_node: fx.Node
    a_2d: fx.Node
    b_2d: fx.Node
    a_scale: object
    b_scale: object
    out_dtype: torch.dtype
    collective_dim: int
    world_size: int
    group_name: str
    output_shape: list[object] | None = None


def _find_bmm_reduce_scatter(
    bmm_node: fx.Node,
) -> tuple[fx.Node, fx.Node, object, object, object] | None:
    worklist = list(bmm_node.users)
    visited: set[fx.Node] = set()
    rs_matches: list[tuple[fx.Node, fx.Node, object, object, object]] = []

    while worklist:
        user = worklist.pop()
        if user in visited:
            continue
        visited.add(user)

        parsed_rs = _parse_reduce_scatter(user)
        if parsed_rs is not None:
            rs_input, dim, world_size, group_name = parsed_rs
            rs_matches.append((user, rs_input, dim, world_size, group_name))
            continue

        if _is_passthrough(user):
            worklist.extend(user.users)

    if len(rs_matches) == 1:
        return rs_matches[0]
    return None


def _find_ag_bmm_replace_target(bmm_node: fx.Node) -> fx.Node | None:
    worklist = list(bmm_node.users)
    visited: set[fx.Node] = set()
    replace_targets: list[fx.Node] = []

    while worklist:
        user = worklist.pop()
        if user in visited:
            continue
        visited.add(user)

        if not _is_passthrough(user):
            continue

        if _node_ndim(user) == 2:
            replace_targets.append(user)
            continue

        worklist.extend(user.users)

    if not replace_targets and _node_ndim(bmm_node) == 2:
        replace_targets = [bmm_node]

    if len(replace_targets) != 1:
        return None
    return replace_targets[0]


class FlashInferCollectiveGemmRewriter:
    def _match_collective_gemm(
        self, bmm_node: fx.Node
    ) -> _FlashInferCollectiveGemmMatch | None:
        parsed_bmm = _parse_bmm_fp8(bmm_node)
        if parsed_bmm is None:
            return None

        a_2d, b_2d, a_scale, b_scale, out_dtype = parsed_bmm
        if out_dtype != torch.bfloat16:
            return None

        rs_match = _find_bmm_reduce_scatter(bmm_node)
        if rs_match is not None:
            rs_node, rs_input, dim, world_size, group_name = rs_match
            if dim == 0 and isinstance(world_size, int) and isinstance(group_name, str):
                gemm_m = _node_first_dim(rs_input)
                if (
                    gemm_m is not None
                    and isinstance(gemm_m, int | torch.SymInt)
                    and _dim_is_statically_lt(gemm_m, FLASHINFER_BMM_FP8_MIN_M)
                ):
                    return None
                output_shape = _node_shape(rs_input)
                return _FlashInferCollectiveGemmMatch(
                    kind="bmm_rs",
                    replace_node=rs_node,
                    a_2d=a_2d,
                    b_2d=b_2d,
                    a_scale=a_scale,
                    b_scale=b_scale,
                    out_dtype=out_dtype,
                    collective_dim=0,
                    world_size=world_size,
                    group_name=group_name,
                    output_shape=output_shape,
                )

        ag_node = _strip_view_like(a_2d)
        parsed_ag = _parse_all_gather(ag_node)
        if parsed_ag is None:
            return None
        ag_input, dim, world_size, group_name = parsed_ag
        if (
            dim != 0
            or not isinstance(world_size, int)
            or not isinstance(group_name, str)
        ):
            return None

        target = _find_ag_bmm_replace_target(bmm_node)
        if target is None:
            return None
        gemm_m = _node_first_dim(target)
        if (
            gemm_m is not None
            and isinstance(gemm_m, int | torch.SymInt)
            and _dim_is_statically_lt(gemm_m, FLASHINFER_BMM_FP8_MIN_M)
        ):
            return None

        return _FlashInferCollectiveGemmMatch(
            kind="ag_bmm",
            replace_node=target,
            a_2d=ag_input,
            b_2d=b_2d,
            a_scale=a_scale,
            b_scale=b_scale,
            out_dtype=out_dtype,
            collective_dim=0,
            world_size=world_size,
            group_name=group_name,
        )

    def _lower_collective_gemm(
        self, graph: fx.Graph, match: _FlashInferCollectiveGemmMatch
    ) -> None:
        if match.kind == "ag_bmm":
            with graph.inserting_before(match.replace_node):
                gathered = graph.call_function(
                    torch.ops.vllm.all_gather.default,
                    args=(
                        match.a_2d,
                        match.collective_dim,
                        match.world_size,
                        match.group_name,
                    ),
                )
                bmm_output = graph.call_function(
                    torch.ops.vllm.bmm_fp8.default,
                    args=(
                        graph.call_function(
                            torch.ops.aten.unsqueeze.default, args=(gathered, 0)
                        ),
                        graph.call_function(
                            torch.ops.aten.unsqueeze.default, args=(match.b_2d, 0)
                        ),
                        match.a_scale,
                        match.b_scale,
                        match.out_dtype,
                        "auto",
                    ),
                )
                mm_output = graph.call_function(
                    torch.ops.aten.squeeze.dim, args=(bmm_output, 0)
                )
            mm_output.meta = dict(match.replace_node.meta)
            match.replace_node.replace_all_uses_with(mm_output)
            graph.erase_node(match.replace_node)
            return

        with graph.inserting_before(match.replace_node):
            bmm_output = graph.call_function(
                torch.ops.vllm.bmm_fp8.default,
                args=(
                    graph.call_function(
                        torch.ops.aten.unsqueeze.default, args=(match.a_2d, 0)
                    ),
                    graph.call_function(
                        torch.ops.aten.unsqueeze.default, args=(match.b_2d, 0)
                    ),
                    match.a_scale,
                    match.b_scale,
                    match.out_dtype,
                    "auto",
                ),
            )
            mm_output = graph.call_function(
                torch.ops.aten.squeeze.dim, args=(bmm_output, 0)
            )
            reduced = graph.call_function(
                torch.ops.vllm.reduce_scatter.default,
                args=(
                    mm_output,
                    match.collective_dim,
                    match.world_size,
                    match.group_name,
                ),
            )
        reduced.meta = dict(match.replace_node.meta)
        match.replace_node.replace_all_uses_with(reduced)
        graph.erase_node(match.replace_node)

    def rewrite(self, graph: fx.Graph) -> int:
        replaced = 0
        for node in list(graph.nodes):
            if not is_func(node, torch.ops.vllm.bmm_fp8.default):
                continue
            match = self._match_collective_gemm(node)
            if match is None:
                continue
            self._lower_collective_gemm(graph, match)
            replaced += 1

        if replaced > 0:
            graph.eliminate_dead_code()
        return replaced
