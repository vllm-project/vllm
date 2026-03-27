# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator
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


def _is_view_like(node: fx.Node) -> bool:
    return any(is_func(node, op) for op in VIEW_LIKE_OPS)


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


def _single_user(node: fx.Node) -> fx.Node | None:
    users = list(node.users)
    if len(users) != 1:
        return None
    return users[0]


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


def _parse_exact_qkv_split(node: fx.Node) -> fx.Node | None:
    if not is_func(node, torch.ops.aten.split_with_sizes.default):
        return None

    split_input = node.kwargs.get("self", node.args[0] if len(node.args) > 0 else None)
    split_sizes = node.kwargs.get(
        "split_sizes", node.args[1] if len(node.args) > 1 else None
    )
    dim = node.kwargs.get("dim", node.args[2] if len(node.args) > 2 else None)

    if not isinstance(split_input, fx.Node):
        return None

    split_ndim = _node_ndim(split_input)
    if split_ndim is not None and split_ndim != 2:
        return None

    if dim not in (-1, 1):
        return None

    if not isinstance(split_sizes, list | tuple) or len(split_sizes) != 3:
        return None
    if any(not isinstance(size, int | torch.SymInt) for size in split_sizes):
        return None

    shape = _node_shape(split_input)
    if shape is not None and len(shape) == 2:
        last_dim = shape[1]
        if (
            isinstance(last_dim, int)
            and all(isinstance(size, int) for size in split_sizes)
            and sum(split_sizes) != last_dim
        ):
            return None

    return split_input


@dataclass
class _FlashInferCollectiveGemmMatch:
    kind: Literal["ag_bmm", "bmm_rs"]
    replace_node: fx.Node
    a_2d: fx.Node
    b_2d: fx.Node
    a_scale: object
    b_scale: object
    out_dtype: torch.dtype
    output_shape: list[object] | None = None


class FlashInferCollectiveGemmRewriter:
    def __init__(self, tp_device_group_name: str) -> None:
        self.tp_device_group_name = tp_device_group_name

    def _match_exact_bmm_reduce_scatter(
        self,
        bmm_node: fx.Node,
        a_2d: fx.Node,
        b_2d: fx.Node,
        a_scale: object,
        b_scale: object,
        out_dtype: torch.dtype,
    ) -> _FlashInferCollectiveGemmMatch | None:
        current = bmm_node
        user = _single_user(current)
        if user is not None and _is_view_like(user):
            current = user
            user = _single_user(current)

        if user is None:
            return None

        parsed_rs = _parse_reduce_scatter(user)
        if parsed_rs is None:
            return None
        rs_input, dim, world_size, group_name = parsed_rs
        if rs_input is not current:
            return None
        if (
            dim != 0
            or not isinstance(world_size, int)
            or not isinstance(group_name, str)
        ):
            return None

        gemm_m = _node_first_dim(rs_input)
        if (
            gemm_m is not None
            and isinstance(gemm_m, int | torch.SymInt)
            and _dim_is_statically_lt(gemm_m, FLASHINFER_BMM_FP8_MIN_M)
        ):
            return None

        return _FlashInferCollectiveGemmMatch(
            kind="bmm_rs",
            replace_node=user,
            a_2d=a_2d,
            b_2d=b_2d,
            a_scale=a_scale,
            b_scale=b_scale,
            out_dtype=out_dtype,
            output_shape=_node_shape(rs_input),
        )

    def _match_exact_ag_qkv(
        self,
        bmm_node: fx.Node,
        ag_input: fx.Node,
        b_2d: fx.Node,
        a_scale: object,
        b_scale: object,
        out_dtype: torch.dtype,
    ) -> _FlashInferCollectiveGemmMatch | None:
        qkv_output = _single_user(bmm_node)
        if qkv_output is None or not _is_view_like(qkv_output):
            return None
        if _node_ndim(qkv_output) != 2:
            return None

        split_node = _single_user(qkv_output)
        if split_node is None:
            return None

        split_input = _parse_exact_qkv_split(split_node)
        if split_input is not qkv_output:
            return None

        gemm_m = _node_first_dim(qkv_output)
        if (
            gemm_m is not None
            and isinstance(gemm_m, int | torch.SymInt)
            and _dim_is_statically_lt(gemm_m, FLASHINFER_BMM_FP8_MIN_M)
        ):
            return None

        return _FlashInferCollectiveGemmMatch(
            kind="ag_bmm",
            replace_node=qkv_output,
            a_2d=ag_input,
            b_2d=b_2d,
            a_scale=a_scale,
            b_scale=b_scale,
            out_dtype=out_dtype,
        )

    def _match_collective_gemm(
        self, bmm_node: fx.Node
    ) -> _FlashInferCollectiveGemmMatch | None:
        parsed_bmm = _parse_bmm_fp8(bmm_node)
        if parsed_bmm is None:
            return None

        a_2d, b_2d, a_scale, b_scale, out_dtype = parsed_bmm
        if out_dtype != torch.bfloat16:
            return None

        rs_match = self._match_exact_bmm_reduce_scatter(
            bmm_node, a_2d, b_2d, a_scale, b_scale, out_dtype
        )
        if rs_match is not None:
            return rs_match

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

        return self._match_exact_ag_qkv(
            bmm_node,
            ag_input,
            b_2d,
            a_scale,
            b_scale,
            out_dtype,
        )

    def _lower_collective_gemm(
        self, graph: fx.Graph, match: _FlashInferCollectiveGemmMatch
    ) -> None:
        if match.kind == "ag_bmm":
            with graph.inserting_before(match.replace_node):
                fused = graph.call_function(
                    torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default,
                    args=(
                        match.a_2d,
                        match.b_2d,
                        match.a_scale,
                        match.b_scale,
                        0,
                        self.tp_device_group_name,
                        match.out_dtype,
                    ),
                )
                mm_output = graph.call_function(
                    operator.getitem,
                    args=(fused, 1),
                )
            mm_output.meta = dict(match.replace_node.meta)
            match.replace_node.replace_all_uses_with(mm_output)
            graph.erase_node(match.replace_node)
            return

        with graph.inserting_before(match.replace_node):
            fused = graph.call_function(
                torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default,
                args=(
                    match.a_2d,
                    match.b_2d,
                    match.a_scale,
                    match.b_scale,
                    "sum",
                    0,
                    0,
                    self.tp_device_group_name,
                    match.output_shape,
                    match.out_dtype,
                ),
            )
        fused.meta = dict(match.replace_node.meta)
        match.replace_node.replace_all_uses_with(fused)
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
