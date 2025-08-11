# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class _RMSNormAndQuantOpHelper:
    """Base helper for RMSNorm and RMSNorm + Quantization functionalization."""

    def __init__(self,
                 epsilon: float,
                 dtype: torch.dtype,
                 device: str,
                 quant_op: Optional[torch._ops.OpOverload] = None,
                 **kwargs):
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.quant_op = quant_op

    def _functional_rmsnorm(self, result_buffer, input_tensor, weight_tensor):
        return torch.ops.higher_order.auto_functionalized(
            torch.ops._C.rms_norm.default,
            result=result_buffer,
            input=input_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon)

    def _functional_fused_add_rmsnorm(self, input_tensor, residual_tensor,
                                      weight_tensor):
        return torch.ops.higher_order.auto_functionalized(
            torch.ops._C.fused_add_rms_norm.default,
            input=input_tensor,
            residual=residual_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon)

    def _functional_rmsnorm_then_quant(self, rmsnorm_result_buffer,
                                       quant_result_buffer, input_tensor,
                                       weight_tensor, scale_tensor):
        if self.quant_op is None:
            raise RuntimeError(
                "_RMSNormAndQuantOpHelper was not initialized with a quant_op."
            )
        rmsnorm_out_tuple = self._functional_rmsnorm(rmsnorm_result_buffer,
                                                     input_tensor,
                                                     weight_tensor)
        quant_out_tuple = torch.ops.higher_order.auto_functionalized(
            self.quant_op,
            result=quant_result_buffer,
            input=rmsnorm_out_tuple[1],
            scale=scale_tensor)
        return quant_out_tuple

    def _functional_fused_add_rmsnorm_then_quant(self, quant_result_buffer,
                                                 input_tensor, residual_tensor,
                                                 weight_tensor, scale_tensor):
        if self.quant_op is None:
            raise RuntimeError(
                "_RMSNormAndQuantOpHelper was not initialized with a quant_op."
            )
        fused_add_rmsnorm_out_tuple = self._functional_fused_add_rmsnorm(
            input_tensor, residual_tensor, weight_tensor)
        quant_out_tuple = torch.ops.higher_order.auto_functionalized(
            self.quant_op,
            result=quant_result_buffer,
            input=fused_add_rmsnorm_out_tuple[1],
            scale=scale_tensor)
        return quant_out_tuple, fused_add_rmsnorm_out_tuple[2]


class _SequenceParallelPatternHelper(_RMSNormAndQuantOpHelper):
    """Helper for sequence parallelism patterns."""

    def __init__(self,
                 epsilon: float,
                 dtype: torch.dtype,
                 device: str,
                 quant_op: Optional[torch._ops.OpOverload] = None,
                 **kwargs):
        super().__init__(epsilon, dtype, device, quant_op=quant_op, **kwargs)
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(x)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.reduce_scatter.default(
            x,
            dim=0,
            world_size=self.tp_size,
            group_name=self.tp_group.unique_name)

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.all_gather.default(
            x,
            dim=0,
            world_size=self.tp_size,
            group_name=self.tp_group.unique_name)


class FirstAllReduceRMSNormPattern(_SequenceParallelPatternHelper):

    def get_inputs(self):
        input = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        permute = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        arg3_1 = torch.empty([4], device=self.device, dtype=self.dtype)

        return [input, permute, arg3_1]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            input: torch.Tensor,
            permute: torch.Tensor,
            arg3_1: torch.Tensor,
        ):
            all_reduce = self._all_reduce(input)
            rmsnorm = self._functional_rmsnorm(permute, all_reduce, arg3_1)

            return rmsnorm[1], all_reduce

        def replacement(
            input: torch.Tensor,
            permute: torch.Tensor,
            arg3_1: torch.Tensor,
        ):
            reduce_scatter = self._reduce_scatter(input)

            rmsnorm_result = torch.empty_like(reduce_scatter)
            rmsnorm = self._functional_rmsnorm(rmsnorm_result, reduce_scatter,
                                               arg3_1)

            all_gather = self._all_gather(rmsnorm[1])

            return all_gather, reduce_scatter

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class MiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights)
            return rmsnorm[1], rmsnorm[2]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights)
            all_gather = self._all_gather(rmsnorm[1])
            return all_gather, rmsnorm[2]

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class LastAllReduceRMSNormPattern(_SequenceParallelPatternHelper):

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights)
            return rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights)
            normalized = self._all_gather(rmsnorm[1])
            return normalized

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


FP8_DTYPE = current_platform.fp8_dtype()


class FirstAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 op: torch._ops.OpOverload):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        input = torch.zeros([1, 8, 4], device=self.device, dtype=self.dtype)
        rmsnorm_result = torch.empty([1, 8, 4],
                                     device=self.device,
                                     dtype=self.dtype)
        quant_result = torch.empty([1, 8, 4],
                                   device=self.device,
                                   dtype=FP8_DTYPE)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)
        scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return [input, rmsnorm_result, quant_result, weight, scale]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            input: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            all_reduce = self._all_reduce(input)
            static_fp8 = self._functional_rmsnorm_then_quant(
                rmsnorm_result, quant_result, all_reduce, weight, scale)
            return static_fp8[1], all_reduce

        def replacement(
            input: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            reduce_scatter = self._reduce_scatter(input)

            rmsnorm_result = torch.empty_like(reduce_scatter,
                                              dtype=rmsnorm_result.dtype)
            quant_result = torch.empty_like(
                rmsnorm_result,  # Output of RMSNorm
                dtype=quant_result.dtype)
            static_fp8 = self._functional_rmsnorm_then_quant(
                rmsnorm_result, quant_result, reduce_scatter, weight, scale)
            all_gather = self._all_gather(static_fp8[1])

            return all_gather, reduce_scatter

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class MiddleAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 op: torch._ops.OpOverload):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)
        result = torch.empty([4, 4], device=self.device, dtype=FP8_DTYPE)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)

        return [
            result,
            residual,
            mm_1,
            rms_norm_weights,
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            static_fp8, rmsnorm_residual_out = self._functional_fused_add_rmsnorm_then_quant(  # noqa: E501
                result, all_reduce, residual, rms_norm_weights, scale)
            return static_fp8[1], rmsnorm_residual_out

        def replacement(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            quant_result_buf = torch.empty_like(reduce_scatter,
                                                dtype=result.dtype)
            static_fp8, rmsnorm_residual_out = self._functional_fused_add_rmsnorm_then_quant(  # noqa: E501
                quant_result_buf, reduce_scatter, residual, rms_norm_weights,
                scale)
            all_gather = self._all_gather(static_fp8[1])
            return all_gather, rmsnorm_residual_out

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class LastAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 op: torch._ops.OpOverload):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)
        result = torch.empty([4, 4], device=self.device, dtype=FP8_DTYPE)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)

        return [
            result,
            residual,
            mm_1,
            rms_norm_weights,
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            static_fp8, _ = self._functional_fused_add_rmsnorm_then_quant(
                result, all_reduce, residual, rms_norm_weights, scale)
            return static_fp8[1]

        def replacement(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            quant_result_buf = torch.empty_like(reduce_scatter,
                                                dtype=result.dtype)
            static_fp8, _ = self._functional_fused_add_rmsnorm_then_quant(
                quant_result_buf, reduce_scatter, residual, rms_norm_weights,
                scale)
            normalized = self._all_gather(static_fp8[1])
            return normalized

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class SequenceParallelismPass(VllmInductorPass):
    """
    This pass enables sequence parallelism for models.
    It identifies patterns where an AllReduce operation is followed by
    an RMSNorm (or RMSNorm and then Quantization) operation.
    These patterns are replaced with a ReduceScatter operation, followed by
    a local RMSNorm/Quantization, and then an AllGather operation.

    The general transformation is:
    Input -> AllReduce -> RMSNorm -> Output
    becomes
    Input -> ReduceScatter -> RMSNorm -> AllGather -> Output

    While this pass itself does not directly yield performance improvements,
    it lays the groundwork for subsequent fusion passes, such as
    GEMM + ReduceScatter and AllGather + GEMM fusions. These fusions can
    significantly reduce communication overhead and improve overall model
    performance.
    """

    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_pass")

        for epsilon in [1e-5, 1e-6]:
            # RMSNorm + Static FP8 quantization patterns
            fp8_quant_op = torch.ops._C.static_scaled_fp8_quant.default
            FirstAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device,
                fp8_quant_op).register(self.patterns)
            MiddleAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device,
                fp8_quant_op).register(self.patterns)
            LastAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device,
                fp8_quant_op).register(self.patterns)

            # Normal RMSNorm patterns
            FirstAllReduceRMSNormPattern(epsilon, self.model_dtype,
                                         self.device).register(self.patterns)

            MiddleAllReduceRMSNormPattern(epsilon, self.model_dtype,
                                          self.device).register(self.patterns)

            LastAllReduceRMSNormPattern(epsilon, self.model_dtype,
                                        self.device).register(self.patterns)

            # WARNING: This is a hack to clear the pattern matcher cache
            # and allow multiple values of epsilon.
            torch._inductor.pattern_matcher._seen_patterns.clear()

    def is_applicable_for_shape(self, shape: Optional[int]) -> bool:
        tp_size = get_tensor_model_parallel_world_size()
        return shape is not None and shape % tp_size == 0

    def __call__(self, graph: fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_sequence_parallelism_pass")
        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns with sequence parallelism", count)
        self.dump_graph(graph, "after_sequence_parallelism_pass")
        self.end_and_log()
