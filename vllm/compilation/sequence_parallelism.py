# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherFusedAddRMSNorm, MatcherQuantFP8, MatcherRMSNorm
from .noop_elimination import NoOpEliminationPass
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


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
    def __init__(self, epsilon: float, dtype: torch.dtype, device: str | None) -> None:
        super().__init__(epsilon, dtype, device)
        self.rmsnorm_matcher = MatcherRMSNorm(epsilon)

    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        arg3_1 = torch.empty([4], device=self.device, dtype=self.dtype)

        return [input, arg3_1]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            arg3_1: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(input)
            rmsnorm = self.rmsnorm_matcher(all_reduce, arg3_1)

            return rmsnorm, all_reduce

        def replacement(
            input: torch.Tensor,
            arg3_1: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(input)

            rmsnorm = self.rmsnorm_matcher(reduce_scatter, arg3_1)
            all_gather = self._all_gather(rmsnorm)
            return all_gather, reduce_scatter

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    def __init__(self, epsilon: float, dtype: torch.dtype, device: str | None) -> None:
        super().__init__(epsilon, dtype, device)
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)

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
            rmsnorm = self.rmsnorm_matcher(all_reduce, rms_norm_weights, residual)
            return rmsnorm[0], rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # pattern matcher replaces from top-to-bottom,
            # so residual is still the full size here.
            # once the seqpar pattern with the previous rmsnorm is replaced
            reduce_scatter = self._reduce_scatter(mm_1)
            residual = residual[0 : reduce_scatter.size(0), ...]
            rmsnorm = self.rmsnorm_matcher(reduce_scatter, rms_norm_weights, residual)
            all_gather = self._all_gather(rmsnorm[0])
            # shape of residual changes but that's fine,
            # next node is already slicing it, now becomes a noop
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


FP8_DTYPE = current_platform.fp8_dtype()


class FirstAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
    ) -> None:
        super().__init__(epsilon, dtype, device)
        self.rmsnorm_matcher = MatcherRMSNorm(epsilon)
        self.quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)

    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.zeros([1, 8, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)
        scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return [input, weight, scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(input)
            rms = self.rmsnorm_matcher(all_reduce, weight)
            quant, _ = self.quant_matcher(rms, scale)
            return quant, all_reduce

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(input)
            rms = self.rmsnorm_matcher(reduce_scatter, weight)
            quant, _ = self.quant_matcher(rms, scale)
            all_gather = self._all_gather(quant)

            return all_gather, reduce_scatter

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiddleAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    def __init__(self, epsilon: float, dtype: torch.dtype, device: str | None) -> None:
        super().__init__(epsilon, dtype, device)
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)
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
            rms, residual_out = self.rmsnorm_matcher(
                all_reduce, rms_norm_weights, residual
            )
            quant, _ = self.quant_matcher(rms, scale)
            return quant, residual_out

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # pattern matcher replaces from top-to-bottom,
            # so residual is still the full size here.
            # add a temporary slice which will become a noop
            # once the seqpar pattern with the previous rmsnorm is replaced
            reduce_scatter = self._reduce_scatter(mm_1)
            residual = residual[0 : reduce_scatter.size(0), ...]
            rms, residual_out = self.rmsnorm_matcher(
                reduce_scatter, rms_norm_weights, residual
            )
            quant, _ = self.quant_matcher(rms, scale)
            all_gather = self._all_gather(quant)
            # shape of residual changes but that's fine,
            # next node is already slicing it, now becomes a noop
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


    This pass splits up the residual tensor across TP ranks and hence divides its size.
    Because the pattern matcher starts at the end of the graph, the replacement
    contains a slice that temporarily conforms the input residual to the correct size.
    After all patterns have been matched, we use a NoOpEliminationPass to clean up
    what have now become no-op slices.

    Note that an older version of the pass did not need this as it operated only on
    custom rms_norm and fused_rms_norm_add custom ops which did not complain about
    mismatched shapes during replacement. So this approach has the same assumption that
    correctness is only maintained if all rms_norm operations are split across ranks.

    Correctness-wise, this is approach strictly better than before - before,
    the graph was incorrect semantically and shape-wise during the pass.
    With this approach there's only semantic incorrectness during the pass.
    Both approaches restore a correct graph once all patterns are matched.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

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

            # Normal RMSNorm patterns
            FirstAllReduceRMSNormPattern(
                epsilon, self.model_dtype, self.device
            ).register(self.patterns)

            MiddleAllReduceRMSNormPattern(
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
            return True
        tp_size = get_tensor_model_parallel_world_size()
        return (compile_range.is_single_size()) and (compile_range.end % tp_size == 0)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        # Clean up reshape nodes
        self.noop_cleanup(graph)
