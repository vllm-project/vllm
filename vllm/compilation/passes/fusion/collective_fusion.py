# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..vllm_inductor_pass import VllmFusionPatternMatcherPass, VllmPatternReplacement

FP8_DTYPE = current_platform.fp8_dtype()

logger = init_logger(__name__)


class BasePattern(VllmPatternReplacement):
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

    @property
    def pattern(self):
        def _pattern(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        return _pattern

    @property
    def replacement(self):
        def _replacement(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul,
                mm_weight,
                "sum",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )
            return gemm_rs

        return _replacement


class AllGatherGEMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [x, weight]

    @property
    def pattern(self):
        def _pattern(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return torch.ops.aten.mm.default(all_gather, weight)

        return _pattern

    @property
    def replacement(self):
        def _replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                x,
                [weight],
                gather_dim=0,
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        return _replacement


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

    @property
    def pattern(self):
        def _pattern(
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

        return _pattern

    @property
    def replacement(self):
        def _replacement(
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

        return _replacement


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

    @property
    def pattern(self):
        def _pattern(
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

        return _pattern

    @property
    def replacement(self):
        def _replacement(
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

        return _replacement


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

    @property
    def pattern(self):
        def _pattern(
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

        return _pattern

    @property
    def replacement(self):
        def _replacement(
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

        return _replacement


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

    @property
    def pattern(self):
        def _pattern(
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

        return _pattern

    @property
    def replacement(self):
        def _replacement(
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

        return _replacement


class AsyncTPPass(VllmFusionPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "async_tp_pass")

        # Enable symmetric memory for the TP process group
        enable_symm_mem_for_group(get_tp_group().device_group.group_name)

        self.register(GEMMReduceScatterPattern(self.model_dtype, self.device))
        self.register(AllGatherGEMMPattern(self.model_dtype, self.device))

        # These fusions are enabled only for bfloat16 models because
        # `scaled_mm` or `cutlass_scaled_mm` with per-token (row-wise) scaling
        # only supports bfloat16 as the output dtype.
        if self.model_dtype == torch.bfloat16:
            self.register(ScaledMMReduceScatterPattern(self.model_dtype, self.device))
            self.register(AllGatherScaledMMPattern(self.model_dtype, self.device))
            self.register(
                CutlassScaledMMReduceScatterPattern(self.model_dtype, self.device)
            )
            self.register(
                AllGatherCutlassScaledMMPattern(self.model_dtype, self.device)
            )

        self.dump_patterns(config, self.pm_pass)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # This pass is applied on top of the sequence parallelism pass,
        # which is only supported in fullgraph compilation mode.
        assert (
            self.compilation_config.use_inductor_graph_partition
            or not self.compilation_config.splitting_ops
        ), "AsyncTPPass requires full-graph compilation"
        return True
