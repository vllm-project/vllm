# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)

from ..inductor_pass import enable_fake_mode
from ..utility.noop_elimination import NoOpEliminationPass
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherQuantFP8

logger = init_logger(__name__)

if hasattr(torch.ops._C, "scaled_fp4_quant"):
    SCALED_FP4_QUANT_OUT_OVERLOAD = torch.ops._C.scaled_fp4_quant.out
    SCALED_FP4_QUANT_DEFAULT_OVERLOAD = torch.ops._C.scaled_fp4_quant.default

# Min hidden size per device capability for sequence parallelism
# Only apply sequence parallelism for models with hidden_size >= threshold
SP_MIN_HIDDEN_SIZE: dict[int, int] = {
    90: 8192,  # H100: only for models with hidden_size >= 8192
    100: 8192,  # Blackwell family: only for models with hidden_size >= 8192
}

# Min size per GPU per device capability for sequence parallelism
# Total min size = min_per_gpu_size * tp_size
# This ensures the threshold scales appropriately with tensor parallelism
SP_MIN_PER_GPU_SIZE_MB: dict[int, float] = {
    90: 8,  # 8MB per GPU for H100
    # Use a more conservative threshold on Blackwell so TP8 starts later.
    100: 32,
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

    capability = current_platform.get_device_capability()
    if capability is None:
        return None

    # Collapse Blackwell variants (sm100/sm103/...) into one policy bucket.
    if current_platform.is_device_capability_family(100):
        device_capability = 100
    else:
        device_capability = capability.to_int()

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


def get_first_out_wrapper(
    fn: Callable[..., Sequence[torch.Tensor]],
) -> Callable[..., torch.Tensor]:
    @functools.wraps(fn)
    def wrapper(*args: Any) -> torch.Tensor:
        return fn(*args)[0]

    return wrapper


class _SequenceParallelPatternHelper:
    """Helper for sequence parallelism patterns."""

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
    ) -> None:
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

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

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=self.dtype, device=self.device, **kwargs)

    def empty_f32(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.float32, device=self.device, **kwargs)


class FirstAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def __init__(self, epsilon: float, dtype: torch.dtype, device: str | None) -> None:
        super().__init__(epsilon, dtype, device)

    def get_inputs(self) -> list[torch.Tensor]:
        # input, weight
        return [self.empty([1, 8, 4]), self.empty([4])]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(input)
            rmsnorm = vllm.ir.ops.rms_norm(all_reduce, weight, self.epsilon)

            return rmsnorm, all_reduce

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(input)

            rmsnorm = vllm.ir.ops.rms_norm(reduce_scatter, weight, self.epsilon)
            all_gather = self._all_gather(rmsnorm)
            return all_gather, reduce_scatter

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def __init__(self, epsilon: float, dtype: torch.dtype, device: str | None) -> None:
        super().__init__(epsilon, dtype, device)

    def get_inputs(self) -> list[torch.Tensor]:
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = vllm.ir.ops.fused_add_rms_norm(
                all_reduce, residual, rms_norm_weights, self.epsilon
            )
            return rmsnorm[0], rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # The pattern matcher replaces from the end of the graph
            # (last layer first). At the time each match is replaced,
            # the preceding layer has NOT been replaced yet, so
            # `residual` is still full-size and the slice below is
            # correct. Once the preceding layer IS replaced, its
            # residual output shrinks to [local_len, H], and this
            # slice becomes semantically incorrect (e.g. for rank > 0,
            # the indices would be out of bounds). However, since the
            # symbolic output shape equals the input shape,
            # NoOpEliminationPass (called at the end of
            # SequenceParallelismPass.__call__) removes these slices
            # before the graph is ever executed or compiled.
            reduce_scatter = self._reduce_scatter(mm_1)
            local_len = reduce_scatter.size(0)
            # when the preceding VocabParallelEmbedding is excluded
            # from the FX graph (e.g., passing `inputs_embeds` directly in VLMs),
            # the FirstAllReduceRMSNorm pattern is never matched. we must
            # perform a proper TP-aware slice here. simply using `[0:local_len]`
            # would incorrectly cause all ranks to process rank 0's chunk.
            residual = residual[
                self.tp_rank * local_len : self.tp_rank * local_len + local_len, ...
            ]
            rmsnorm = vllm.ir.ops.fused_add_rms_norm(
                reduce_scatter, residual, rms_norm_weights, self.epsilon
            )
            all_gather = self._all_gather(rmsnorm[0])
            # residual output is now [local_len, H]; the next layer's
            # slice on it is semantically incorrect until
            # NoOpEliminationPass removes it.
            return all_gather, rmsnorm[1]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )
        pm.register_replacement(
            get_first_out_wrapper(pattern),
            get_first_out_wrapper(replacement),
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
        )


class FirstAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
    ) -> None:
        super().__init__(epsilon, dtype, device)
        self.quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)

    def get_inputs(self) -> list[torch.Tensor]:
        # input, weight, scale
        return [self.empty([1, 8, 4]), self.empty([4]), self.empty_f32([1, 1])]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(input)
            rms = vllm.ir.ops.rms_norm(all_reduce, weight, self.epsilon)
            quant, _ = self.quant_matcher(rms, scale)
            return quant, all_reduce

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(input)
            rms = vllm.ir.ops.rms_norm(reduce_scatter, weight, self.epsilon)
            quant, _ = self.quant_matcher(rms, scale)
            all_gather = self._all_gather(quant)

            return all_gather, reduce_scatter

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiddleAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    def __init__(self, epsilon: float, dtype: torch.dtype, device: str | None) -> None:
        super().__init__(epsilon, dtype, device)
        self.quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)

    def get_inputs(self) -> list[torch.Tensor]:
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)

        return [residual, mm_1, rms_norm_weights, scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rms, residual_out = vllm.ir.ops.fused_add_rms_norm(
                all_reduce, residual, rms_norm_weights, self.epsilon
            )
            quant, _ = self.quant_matcher(rms, scale)
            return quant, residual_out

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # See MiddleAllReduceRMSNormPattern.replacement for a
            # detailed explanation of the temporary slice below:
            # it is correct when first inserted, becomes semantically
            # incorrect after the preceding layer is replaced, and is
            # removed by NoOpEliminationPass before the graph is compiled.
            reduce_scatter = self._reduce_scatter(mm_1)
            local_len = reduce_scatter.size(0)
            # when the preceding VocabParallelEmbedding is excluded
            # from the FX graph (e.g., passing `inputs_embeds` directly in VLMs),
            # the FirstAllReduceRMSNorm pattern is never matched. we must
            # perform a proper TP-aware slice here. simply using `[0:local_len]`
            # would incorrectly cause all ranks to process rank 0's chunk.
            residual = residual[
                self.tp_rank * local_len : self.tp_rank * local_len + local_len, ...
            ]
            rms, residual_out = vllm.ir.ops.fused_add_rms_norm(
                reduce_scatter, residual, rms_norm_weights, self.epsilon
            )
            quant, _ = self.quant_matcher(rms, scale)
            all_gather = self._all_gather(quant)
            # residual output is now [local_len, H]; the next layer's
            # slice on it is semantically incorrect until
            # NoOpEliminationPass removes it.
            return all_gather, residual_out

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

        pm.register_replacement(
            get_first_out_wrapper(pattern),
            get_first_out_wrapper(replacement),
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
        )


class FirstAllReduceRMSNormStaticNVFP4Pattern(_SequenceParallelPatternHelper):
    def get_inputs(self) -> list[torch.Tensor]:
        input = self.empty([8, 16])
        weight = self.empty([16])
        input_global_scale = self.empty_f32([1, 1])
        quant_output = torch.empty([8, 8], device=self.device, dtype=torch.uint8)
        output_scale = torch.empty([128, 4], device=self.device, dtype=torch.int32)
        return [input, weight, input_global_scale, quant_output, output_scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            input_global_scale: torch.Tensor,
            quant_output: torch.Tensor,
            output_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(input)
            rms = vllm.ir.ops.rms_norm(all_reduce, weight, self.epsilon)
            quant = auto_functionalized(
                SCALED_FP4_QUANT_OUT_OVERLOAD,
                input=rms,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
                output=quant_output,
                output_scale=output_scale,
            )
            return quant[1], all_reduce, quant[2]

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            input_global_scale: torch.Tensor,
            quant_output: torch.Tensor,
            output_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(input)
            rms = vllm.ir.ops.rms_norm(reduce_scatter, weight, self.epsilon)
            rms = torch.ops.aten.view.default(rms, [-1, rms.shape[-1]])
            quant = SCALED_FP4_QUANT_DEFAULT_OVERLOAD(
                rms,
                input_global_scale,
                True,
            )
            return (
                self._all_gather(quant[0]),
                reduce_scatter,
                self._all_gather(quant[1]),
            )

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiddleAllReduceRMSNormStaticNVFP4Pattern(_SequenceParallelPatternHelper):
    def get_inputs(self) -> list[torch.Tensor]:
        mm_1 = self.empty([8, 16])
        residual = self.empty([8, 16])
        rms_norm_weights = self.empty([16])
        input_global_scale = self.empty_f32([1, 1])
        quant_output = torch.empty([8, 8], device=self.device, dtype=torch.uint8)
        output_scale = torch.empty([128, 4], device=self.device, dtype=torch.int32)
        return [
            residual,
            mm_1,
            rms_norm_weights,
            input_global_scale,
            quant_output,
            output_scale,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            input_global_scale: torch.Tensor,
            quant_output: torch.Tensor,
            output_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rms, residual_out = vllm.ir.ops.fused_add_rms_norm(
                all_reduce, residual, rms_norm_weights, self.epsilon
            )
            quant = auto_functionalized(
                SCALED_FP4_QUANT_OUT_OVERLOAD,
                input=rms,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
                output=quant_output,
                output_scale=output_scale,
            )
            return quant[1], residual_out, quant[2]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            input_global_scale: torch.Tensor,
            quant_output: torch.Tensor,
            output_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Keep this slice in sync with the non-quantized SP replacement:
            # once the previous SP pattern fires, it becomes a no-op.
            reduce_scatter = self._reduce_scatter(mm_1)
            residual = residual[0 : reduce_scatter.size(0), ...]
            rms, residual_out = vllm.ir.ops.fused_add_rms_norm(
                reduce_scatter, residual, rms_norm_weights, self.epsilon
            )
            rms = torch.ops.aten.view.default(rms, [-1, rms.shape[-1]])
            quant = SCALED_FP4_QUANT_DEFAULT_OVERLOAD(
                rms,
                input_global_scale,
                True,
            )
            return self._all_gather(quant[0]), residual_out, self._all_gather(quant[1])

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

    This pass is only supported when compiling the whole graph (fullgraph
    mode, i.e. using Inductor graph partition or empty splitting_ops).
    Piecewise compilation is not supported because the residual tensor
    gets split across TP ranks, causing size mismatches at subgraph
    boundaries.

    This pass splits up the residual tensor across TP ranks and hence divides
    its size. The pattern matcher starts at the end of the graph (last layer
    first), so when each replacement inserts a residual slice, the preceding
    layer has not been replaced yet and the slice is correct. Once the
    preceding layer IS replaced, its residual output shrinks and the slice
    becomes semantically incorrect (out-of-bounds indices for rank > 0).
    The graph is never executed in this intermediate state —
    NoOpEliminationPass removes these slices based on symbolic shape equality
    (input shape == output shape) before the graph is compiled.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        # Get min_token_num threshold
        # Read min_token_num from config (calculated during config init)
        self.min_token_num = None
        if config.model_config is not None:
            pass_config = config.compilation_config.pass_config
            self.min_token_num = pass_config.sp_min_token_num

            if self.min_token_num is not None:
                # Take the min to avoid exceeding max_num_batched_tokens
                max_batched = config.scheduler_config.max_num_batched_tokens
                if max_batched is not None:
                    self.min_token_num = min(self.min_token_num, max_batched)
                logger.debug_once(
                    f"Sequence parallelism min token threshold: {self.min_token_num}",
                    scope="global",
                )

        # Used to clean up redundant views created temporarily
        # to circumvent residual shape change issues
        self.noop_cleanup = NoOpEliminationPass(config)
        self.noop_cleanup.pass_name = f"{self.pass_name}.{self.noop_cleanup.pass_name}"

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_pass"
        )

        for epsilon in [1e-5, 1e-6]:
            # RMSNorm + Static FP8 quantization patterns
            FirstAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)
            MiddleAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)

            if "SCALED_FP4_QUANT_OUT_OVERLOAD" in globals():
                FirstAllReduceRMSNormStaticNVFP4Pattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)
                MiddleAllReduceRMSNormStaticNVFP4Pattern(
                    epsilon, self.model_dtype, self.device
                ).register(self.patterns)

            # Normal RMSNorm patterns
            FirstAllReduceRMSNormPattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)

            MiddleAllReduceRMSNormPattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Determines if sequence parallelism should be applied for the given
        compile range.

        SP is only beneficial for larger batch sizes where the communication
        overhead is amortized. For small batches, the overhead of splitting
        and gathering tensors across TP ranks outweighs the benefits.

        Returns False (SP disabled) when:
        - min_token_num is None (SP disabled for this device/config)
        - The compile range starts below the minimum token threshold
        """
        assert (
            self.compilation_config.use_inductor_graph_partition
            or not self.compilation_config.splitting_ops
        ), "SequenceParallelismPass requires full-graph compilation"

        # min_token_num is None when SP is disabled for this device/config
        # (e.g., non-CUDA platform, unsupported GPU, or small hidden_size)
        if self.min_token_num is None:
            return False

        # Only apply SP when batch size meets the minimum threshold
        return compile_range.start >= self.min_token_num

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        # Clean up reshape nodes
        self.noop_cleanup(graph)
