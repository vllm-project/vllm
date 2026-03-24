# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator
from dataclasses import dataclass
from typing import Literal

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..fx_utils import is_func
from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

FP8_DTYPE = current_platform.fp8_dtype()

logger = init_logger(__name__)

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


class BasePattern:
    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()


class GEMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        mul = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mm_weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [mul, mm_weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul,
                mm_weight,
                "sum",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherGEMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

            return torch.ops.aten.mm.default(all_gather, weight)

        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                x,
                [weight],
                gather_dim=0,
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)
        return [input, mm_weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            scaled_mm = torch.ops.aten._scaled_mm.default(
                input,
                mat2=mat2,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                scaled_mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        return [x, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            return torch.ops.aten._scaled_mm.default(
                all_gather,
                mat2=weight,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class CutlassScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        cutlass_mm_output = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        return [input, mm_weight, scale_a, scale_b, cutlass_mm_output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=cutlass_mm_output,
                a=input,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )

            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                cutlass_scaled_mm[1],
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherCutlassScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        s2 = weight.shape[1]
        output = torch.empty([s1, s2], device=self.device, dtype=self.dtype)

        return [x, weight, scale_a, scale_b, output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=output,
                a=all_gather,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )
            return cutlass_scaled_mm[1]

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class FlashInferBMMFP8ReduceScatterPattern(BasePattern):
    """Matches unsqueeze → unsqueeze → bmm_fp8 → view → reduce_scatter
    and replaces with fused_flashinfer_scaled_matmul_reduce_scatter."""

    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        weight = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return [input, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            input_3d = input.unsqueeze(0)
            weight_3d = weight.unsqueeze(0)
            bmm_result = torch.ops.vllm.bmm_fp8.default(
                input_3d, weight_3d, scale_a, scale_b, self.dtype, "auto"
            )
            mm_result = bmm_result.view(input.shape[0], weight.shape[1])
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm_result,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            output_shape = [*input.shape[:-1], weight.shape[1]]
            scatter_dim = 0
            return torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter(
                input,
                weight,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,
                scatter_dim,
                self.tp.device_group.group_name,
                output_shape,
                self.dtype,
            )

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherFlashInferBMMFP8Pattern(BasePattern):
    """Matches all_gather → unsqueeze → unsqueeze → bmm_fp8 → view
    and replaces with fused_all_gather_flashinfer_scaled_matmul."""

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return [x, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            ag_3d = all_gather.unsqueeze(0)
            weight_3d = weight.unsqueeze(0)
            bmm_result = torch.ops.vllm.bmm_fp8.default(
                ag_3d, weight_3d, scale_a, scale_b, self.dtype, "auto"
            )
            return bmm_result.view(all_gather.shape[0], weight.shape[1])

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_output = (
                torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul(
                    x,
                    weight,
                    scale_a,
                    scale_b,
                    0,  # gather_dim
                    self.tp.device_group.group_name,
                    self.dtype,
                )
            )
            return mm_output

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AsyncTPPass(VllmPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        # Enable symmetric memory for the TP process group
        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="async_tp_pass"
        )
        GEMMReduceScatterPattern(self.model_dtype, self.device).register(self.patterns)

        AllGatherGEMMPattern(self.model_dtype, self.device).register(self.patterns)

        # These fusions are enabled only for bfloat16 models because
        # `scaled_mm` or `cutlass_scaled_mm` with per-token (row-wise) scaling
        # only supports bfloat16 as the output dtype.
        if self.model_dtype == torch.bfloat16:
            ScaledMMReduceScatterPattern(self.model_dtype, self.device).register(
                self.patterns
            )
            AllGatherScaledMMPattern(self.model_dtype, self.device).register(
                self.patterns
            )

            CutlassScaledMMReduceScatterPattern(self.model_dtype, self.device).register(
                self.patterns
            )
            AllGatherCutlassScaledMMPattern(self.model_dtype, self.device).register(
                self.patterns
            )

        self.dump_patterns(config, self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # This pass is applied on top of the sequence parallelism pass.
        # It inherits the same applicability condition as `SequenceParallelismPass`.
        # See `SequenceParallelismPass.is_applicable` for more details.
        if (
            not self.compilation_config.splitting_ops
            or self.compilation_config.use_inductor_graph_partition
        ):
            return True
        tp_size = get_tensor_model_parallel_world_size()
        return bool(compile_range.is_single_size() and compile_range.end % tp_size == 0)

    def _match_flashinfer_collective_gemm(
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
                output_shape = _node_shape(rs_input)
                return _FlashInferCollectiveGemmMatch(
                    kind="bmm_rs",
                    replace_node=rs_node,
                    a_2d=a_2d,
                    b_2d=b_2d,
                    a_scale=a_scale,
                    b_scale=b_scale,
                    out_dtype=out_dtype,
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

        return _FlashInferCollectiveGemmMatch(
            kind="ag_bmm",
            replace_node=target,
            a_2d=ag_input,
            b_2d=b_2d,
            a_scale=a_scale,
            b_scale=b_scale,
            out_dtype=out_dtype,
            group_name=group_name,
        )

    def _lower_flashinfer_collective_gemm(
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
                        match.group_name,
                        match.out_dtype,
                    ),
                )
                mm_output = graph.call_function(operator.getitem, args=(fused, 1))
            mm_output.meta = dict(match.replace_node.meta)
            match.replace_node.replace_all_uses_with(mm_output)
            graph.erase_node(match.replace_node)
            return

        output_shape = match.output_shape
        if output_shape is None:
            output_shape = [
                match.a_2d.meta["val"].shape[0],
                match.b_2d.meta["val"].shape[1],
            ]

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
                    match.group_name,
                    output_shape,
                    match.out_dtype,
                ),
            )
        fused.meta = dict(match.replace_node.meta)
        match.replace_node.replace_all_uses_with(fused)
        graph.erase_node(match.replace_node)

    def _rewrite_flashinfer_collective_gemm(self, graph: fx.Graph) -> int:
        replaced = 0
        for node in list(graph.nodes):
            if not is_func(node, torch.ops.vllm.bmm_fp8.default):
                continue
            match = self._match_flashinfer_collective_gemm(node)
            if match is None:
                continue
            self._lower_flashinfer_collective_gemm(graph, match)
            replaced += 1

        if replaced > 0:
            graph.eliminate_dead_code()
        return replaced

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        matched_by_patterns = self.patterns.apply(graph)
        matched_flashinfer = 0
        if self.model_dtype == torch.bfloat16 and hasattr(torch.ops.vllm, "bmm_fp8"):
            matched_flashinfer = self._rewrite_flashinfer_collective_gemm(graph)
        self.matched_count = matched_by_patterns + matched_flashinfer
        logger.debug("Replaced %s patterns", self.matched_count)
