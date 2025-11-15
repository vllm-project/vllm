# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

# Min hidden size per device capability for sequence parallelism
# Only apply sequence parallelism for models with hidden_size >= threshold
SP_MIN_HIDDEN_SIZE: dict[int, int] = {
    90: 8192,  # H100: only for models with hidden_size >= 8192
    100: 8192,  # Blackwell: only for models with hidden_size >= 8192
}

# Min size per GPU per device capability for sequence parallelism
# Total min size = min_per_gpu_size * tp_size
# This ensures the threshold scales appropriately with tensor parallelism
SP_MIN_PER_GPU_SIZE_MB: dict[int, float] = {
    90: 8,  # 8MB per GPU for H100
    100: 64,  # 64MB per GPU for Blackwell
}


def get_sequence_parallelism_threshold(
    hidden_size: int,
    tp_size: int,
    element_size: int,
) -> int | None:
    """
    Calculate the minimum token threshold for applying sequence parallelism.

    Returns None if sequence parallelism should not be applied based on model size.

    Branching logic based on device capability:
    - Check if hidden_size >= SP_MIN_HIDDEN_SIZE[device_capability]
    - If not, returns None (SP disabled for small models on this device)
    - If yes, calculates threshold based on per-GPU size

    Formula: min_token_num = (min_per_gpu_size_mb * tp_size * MiB) //
             (hidden_size * element_size)
    """
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return None

    device_capability = current_platform.get_device_capability().to_int()

    # Check if device has configured thresholds
    min_hidden_size = SP_MIN_HIDDEN_SIZE.get(device_capability)
    min_per_gpu_size_mb = SP_MIN_PER_GPU_SIZE_MB.get(device_capability)

    if min_hidden_size is None or min_per_gpu_size_mb is None:
        return None

    # Only apply sequence parallelism for models meeting the size threshold
    if hidden_size < min_hidden_size:
        return None

    MiB = 1024 * 1024
    min_size = min_per_gpu_size_mb * MiB * tp_size
    return int(min_size // (hidden_size * element_size))


class _RMSNormAndQuantOpHelper:
    """Base helper for RMSNorm and RMSNorm + Quantization functionalization."""

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
        quant_op: torch._ops.OpOverload | None = None,
        **kwargs,
    ):
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
            epsilon=self.epsilon,
        )

    def _functional_fused_add_rmsnorm(
        self, input_tensor, residual_tensor, weight_tensor
    ):
        return torch.ops.higher_order.auto_functionalized(
            torch.ops._C.fused_add_rms_norm.default,
            input=input_tensor,
            residual=residual_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon,
        )

    def _functional_rmsnorm_then_quant(
        self,
        rmsnorm_result_buffer,
        quant_result_buffer,
        input_tensor,
        weight_tensor,
        scale_tensor,
    ):
        if self.quant_op is None:
            raise RuntimeError(
                "_RMSNormAndQuantOpHelper was not initialized with a quant_op."
            )
        rmsnorm_out_tuple = self._functional_rmsnorm(
            rmsnorm_result_buffer, input_tensor, weight_tensor
        )
        quant_out_tuple = torch.ops.higher_order.auto_functionalized(
            self.quant_op,
            result=quant_result_buffer,
            input=rmsnorm_out_tuple[1],
            scale=scale_tensor,
        )
        return quant_out_tuple

    def _functional_fused_add_rmsnorm_then_quant(
        self,
        quant_result_buffer,
        input_tensor,
        residual_tensor,
        weight_tensor,
        scale_tensor,
    ):
        if self.quant_op is None:
            raise RuntimeError(
                "_RMSNormAndQuantOpHelper was not initialized with a quant_op."
            )
        fused_add_rmsnorm_out_tuple = self._functional_fused_add_rmsnorm(
            input_tensor, residual_tensor, weight_tensor
        )
        quant_out_tuple = torch.ops.higher_order.auto_functionalized(
            self.quant_op,
            result=quant_result_buffer,
            input=fused_add_rmsnorm_out_tuple[1],
            scale=scale_tensor,
        )
        return quant_out_tuple, fused_add_rmsnorm_out_tuple[2]


class _SequenceParallelPatternHelper(_RMSNormAndQuantOpHelper):
    """Helper for sequence parallelism patterns."""

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
        quant_op: torch._ops.OpOverload | None = None,
        **kwargs,
    ):
        super().__init__(epsilon, dtype, device, quant_op=quant_op, **kwargs)
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(x)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.reduce_scatter.default(
            x, dim=0, world_size=self.tp_size, group_name=self.tp_group.unique_name
        )

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.all_gather.default(
            x, dim=0, world_size=self.tp_size, group_name=self.tp_group.unique_name
        )


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
            rmsnorm = self._functional_rmsnorm(rmsnorm_result, reduce_scatter, arg3_1)

            all_gather = self._all_gather(rmsnorm[1])

            return all_gather, reduce_scatter

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)

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
                all_reduce, residual, rms_norm_weights
            )
            return rmsnorm[1], rmsnorm[2]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights
            )
            all_gather = self._all_gather(rmsnorm[1])
            return all_gather, rmsnorm[2]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class LastAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)

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
                all_reduce, residual, rms_norm_weights
            )
            return rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights
            )
            normalized = self._all_gather(rmsnorm[1])
            return normalized

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


FP8_DTYPE = current_platform.fp8_dtype()


class FirstAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str, op: torch._ops.OpOverload
    ):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        input = torch.zeros([1, 8, 4], device=self.device, dtype=self.dtype)
        rmsnorm_result = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        quant_result = torch.empty([1, 8, 4], device=self.device, dtype=FP8_DTYPE)
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
                rmsnorm_result, quant_result, all_reduce, weight, scale
            )
            return static_fp8[1], all_reduce

        def replacement(
            input: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            reduce_scatter = self._reduce_scatter(input)

            rmsnorm_result = torch.empty_like(
                reduce_scatter, dtype=rmsnorm_result.dtype
            )
            quant_result = torch.empty_like(
                rmsnorm_result,  # Output of RMSNorm
                dtype=quant_result.dtype,
            )
            static_fp8 = self._functional_rmsnorm_then_quant(
                rmsnorm_result, quant_result, reduce_scatter, weight, scale
            )
            all_gather = self._all_gather(static_fp8[1])

            return all_gather, reduce_scatter

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiddleAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str, op: torch._ops.OpOverload
    ):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)
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
            static_fp8, rmsnorm_residual_out = (
                self._functional_fused_add_rmsnorm_then_quant(  # noqa: E501
                    result, all_reduce, residual, rms_norm_weights, scale
                )
            )
            return static_fp8[1], rmsnorm_residual_out

        def replacement(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            quant_result_buf = torch.empty_like(reduce_scatter, dtype=result.dtype)
            static_fp8, rmsnorm_residual_out = (
                self._functional_fused_add_rmsnorm_then_quant(  # noqa: E501
                    quant_result_buf, reduce_scatter, residual, rms_norm_weights, scale
                )
            )
            all_gather = self._all_gather(static_fp8[1])
            return all_gather, rmsnorm_residual_out

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class LastAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str, op: torch._ops.OpOverload
    ):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)
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
                result, all_reduce, residual, rms_norm_weights, scale
            )
            return static_fp8[1]

        def replacement(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            quant_result_buf = torch.empty_like(reduce_scatter, dtype=result.dtype)
            static_fp8, _ = self._functional_fused_add_rmsnorm_then_quant(
                quant_result_buf, reduce_scatter, residual, rms_norm_weights, scale
            )
            normalized = self._all_gather(static_fp8[1])
            return normalized

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class SequenceParallelismPass(VllmPatternMatcherPass):
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

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        # Get min_token_num threshold
        self.min_token_num = None
        if config.model_config is not None:
            pass_config = config.compilation_config.pass_config

            # Check if user provided explicit token override
            # User override works regardless of hidden_size
            if pass_config.sequence_parallelism_min_token_num is not None:
                self.min_token_num = pass_config.sequence_parallelism_min_token_num
            else:
                # Otherwise calculate using helper function with branching logic
                tp_size = get_tensor_model_parallel_world_size()
                hidden_size = config.model_config.get_hidden_size()
                element_size = self.model_dtype.itemsize
                self.min_token_num = get_sequence_parallelism_threshold(
                    hidden_size, tp_size, element_size
                )

            if self.min_token_num is not None:
                # take the min to avoid exceeding max_num_batched_tokens
                max_batched = config.scheduler_config.max_num_batched_tokens
                if max_batched is not None:
                    self.min_token_num = min(self.min_token_num, max_batched)
                logger.debug_once(
                    f"Sequence parallelism min token threshold: {self.min_token_num}",
                    scope="global",
                )

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_pass"
        )

        for epsilon in [1e-5, 1e-6]:
            # RMSNorm + Static FP8 quantization patterns
            fp8_quant_op = torch.ops._C.static_scaled_fp8_quant.default
            FirstAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device, fp8_quant_op
            ).register(self.patterns)
            MiddleAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device, fp8_quant_op
            ).register(self.patterns)
            LastAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device, fp8_quant_op
            ).register(self.patterns)

            # Normal RMSNorm patterns
            FirstAllReduceRMSNormPattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)

            MiddleAllReduceRMSNormPattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)

            LastAllReduceRMSNormPattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)
        self.dump_patterns(config, self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # When sequence parallelism is enabled, the residual tensor from RMSNorm
        # needs to be split along the sequence dimension. However, this dimension
        # is symbolic during piecewise compilation, and splitting symbolic shapes
        # is not supported.
        #
        # This pass is therefore only applied when the sequence dimension is
        # concrete:
        # 1. In full-graph compilation mode (no Dynamo splitting ops are used).
        #   For this case we always pad num_tokens to be a multiple of
        #   tensor_parallel_size, so there's no need to check shape % tp_size == 0.
        # 2. For specific shape provided during compilation (e.g., from
        #    `compile_sizes`), which must be divisible by the tensor-parallel
        #    size.
        if (
            not self.compilation_config.splitting_ops
            or self.compilation_config.use_inductor_graph_partition
        ):
            apply = True
        else:
            tp_size = get_tensor_model_parallel_world_size()
            apply = (compile_range.is_single_size()) and (
                compile_range.end % tp_size == 0
            )

        # Additional check: only apply if range is above minimum threshold
        # Sequence parallelism is only beneficial for larger batch sizes
        if apply and self.min_token_num is not None:
            return compile_range.start >= self.min_token_num

        return apply

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
